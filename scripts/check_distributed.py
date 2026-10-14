"""Launch with torchrun: validate synchronous PPO, per-rank RNG and resume."""
import argparse
from pathlib import Path
import sys
import json
import pickle
import torch
import torch.distributed as dist

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pfn_network import distributed
from pfn_network.env.dgp import TaskPrior, sample_task
from pfn_network.models import PFNPolicy, ModelConfig
from pfn_network.train_eval.train import PPOConfig, PPOTrainer, RolloutBuffer, collect_episode

def assert_tree_equal(left, right, label='state'):
    """Checkpoint restoration itself must be exact, including optimizer moments."""
    if isinstance(left, torch.Tensor):
        if not isinstance(right, torch.Tensor) or left.dtype != right.dtype:
            raise AssertionError(f'{label}: tensor type/dtype changed')
        torch.testing.assert_close(left.detach().cpu(), right.detach().cpu(), rtol=0, atol=0, msg=label)
    elif isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            raise AssertionError(f'{label}: keys changed')
        for key in left:
            assert_tree_equal(left[key], right[key], f'{label}.{key}')
    elif isinstance(left, (list, tuple)):
        if not isinstance(right, type(left)) or len(left) != len(right):
            raise AssertionError(f'{label}: sequence changed')
        for index, (a, b) in enumerate(zip(left, right)):
            assert_tree_equal(a, b, f'{label}[{index}]')
    elif left != right:
        raise AssertionError(f'{label}: value changed')

def rank_parameter_difference(model, device):
    vector = torch.cat([p.detach().flatten() for p in model.parameters()])
    leader = vector.clone()
    dist.broadcast(leader, src=0)
    return distributed.max_value((vector - leader).abs().max().item(), device)

def validate_restored_state(trainer, other):
    assert_tree_equal(trainer.model.state_dict(), other.model.state_dict(), 'model')
    assert_tree_equal(trainer.optimizer.state_dict(), other.optimizer.state_dict(), 'optimizer')
    assert_tree_equal(trainer.action_generator.get_state(), other.action_generator.get_state(), 'action_rng')
    assert_tree_equal(trainer.rng.bit_generator.state, other.rng.bit_generator.state, 'minibatch_rng')
    assert_tree_equal((trainer.real_interactions, trainer.gradient_steps),
                      (other.real_interactions, other.gradient_steps), 'counters')

def compare_resumed_update(model, other, buffer):
    """Fixed FP32 tolerances, not adjusted from the observed error.

    State restoration is checked separately with zero tolerance. Further device
    computations are compared numerically, including gradients and outputs.
    """
    errors, parameters = [], []
    gradient_max = probability_max = value_max = 0.0
    for (name, p), (_, q) in zip(model.named_parameters(), other.named_parameters()):
        difference = (p.detach()-q.detach()).abs().max().item()
        parameters.append({'name': name, 'max_abs_difference': difference})
        try:
            torch.testing.assert_close(p, q, atol=1e-5, rtol=1e-4, msg=f'resume parameter {name}')
            if (p.grad is None) != (q.grad is None):
                raise AssertionError(f'{name}: gradient presence differs')
            if p.grad is not None:
                gradient_max = max(gradient_max, (p.grad-q.grad).abs().max().item())
                torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-4, msg=f'resume gradient {name}')
        except AssertionError as error:
            errors.append(str(error))
    with torch.no_grad():
        for row in range(len(buffer)):
            obs = buffer.observations[row]
            history = [buffer.records[i] for i in buffer.history_indices[row, :buffer.history_lengths[row]]]
            a, av = model(obs, history, int(buffer.episodes[row]))
            b, bv = other(obs, history, int(buffer.episodes[row]))
            probability_max = max(probability_max, (a.probs-b.probs).abs().max().item())
            value_max = max(value_max, (av-bv).abs().item())
            try:
                torch.testing.assert_close(a.probs, b.probs, atol=2e-5, rtol=2e-4, msg=f'resume probabilities row {row}')
                torch.testing.assert_close(av, bv, atol=2e-5, rtol=2e-4, msg=f'resume value row {row}')
            except AssertionError as error:
                errors.append(str(error))
    return {'errors': errors, 'max_parameter_difference': max(p['max_abs_difference'] for p in parameters),
            'max_gradient_difference': gradient_max, 'max_probability_difference': probability_max,
            'max_value_difference': value_max,
            'worst_parameters': sorted(parameters, key=lambda p:p['max_abs_difference'], reverse=True)[:5],
            'tolerances': {'parameters': {'atol':1e-5, 'rtol':1e-4},
                           'gradients_and_outputs': {'atol':2e-5, 'rtol':2e-4}}}

def collect(model, ranks, config):
    prior = TaskPrior(n=5, budget=2, catalog_size=6, horizon=2)
    buffer = RolloutBuffer(2 * len(ranks), 4)
    for index in ranks:
        bundle = sample_task(prior, 19, 'train', index, 'FC' if index % 2 == 0 else 'CD')
        buffer, _ = collect_episode(model, bundle.env(index), index, index, [], None,
                                    config, deterministic=True, buffer=buffer)
    return buffer

def merge_rollouts(buffers):
    """Preserve actual actions, old log probabilities, targets and history links."""
    context = buffers[0].context_length
    merged = RolloutBuffer(sum(map(len, buffers)), context,
                           prefix_capacity=sum(b.record_count for b in buffers))
    row_offset = record_offset = 0
    for buffer in buffers:
        assert buffer.context_length == context
        end = row_offset + len(buffer)
        merged.observations[row_offset:end] = buffer.observations[:len(buffer)]
        for name in ('actions','episodes','times','history_lengths','treated_counts',
                     'rewards','old_log_probs','values','advantages','targets','done'):
            getattr(merged, name)[row_offset:end] = getattr(buffer, name)[:len(buffer)]
        merged.history_indices[row_offset:end] = buffer.history_indices[:len(buffer)] + record_offset
        stop = record_offset + buffer.record_count
        merged.records[record_offset:stop] = buffer.records[:buffer.record_count]
        merged.record_observations[record_offset:stop] = buffer.record_observations[:buffer.record_count]
        row_offset, record_offset = end, stop
    merged.size, merged.record_count = row_offset, record_offset
    merged._record_ids = {id(record): i for i, record in enumerate(merged.records[:record_offset])}
    return merged

def tensor_check(left, right, label, atol, rtol):
    left, right = left.detach().cpu(), right.detach().cpu()
    difference = (left-right).abs()
    report = {'name': label, 'max_abs_difference': difference.max().item(), 'atol':atol, 'rtol':rtol}
    try:
        torch.testing.assert_close(left, right, atol=atol, rtol=rtol)
        report['passed'] = True
    except AssertionError as error:
        report.update(passed=False, error=str(error))
    return report

def output_checks(left, right, buffer, prefix, atol, rtol):
    reports = []
    with torch.no_grad():
        for row in range(len(buffer)):
            obs = buffer.observations[row]
            history = [buffer.records[i] for i in buffer.history_indices[row, :buffer.history_lengths[row]]]
            a, av = left(obs, history, int(buffer.episodes[row]))
            b, bv = right(obs, history, int(buffer.episodes[row]))
            reports.append(tensor_check(a.probs,b.probs,f'{prefix}.row{row}.probabilities',atol,rtol))
            reports.append(tensor_check(av,bv,f'{prefix}.row{row}.value',atol,rtol))
    return reports

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='musa')
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    device = distributed.initialize(args.device)
    rank, size = distributed.rank(), distributed.world_size()
    if size < 2:
        raise ValueError('Launch at least two ranks with torchrun')
    torch.set_num_threads(2)
    if rank == 0:
        args.output.mkdir(parents=True, exist_ok=False)
    distributed.barrier()
    torch.manual_seed(20)
    model = PFNPolicy(ModelConfig(width=8, heads=2, context_length=4, transformer_layers=1)).to(device)
    config = PPOConfig(epochs=1, minibatch=2*size)
    trainer = PPOTrainer(model, config, seed=31)
    initial = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    buffer = collect(model, [rank], config)
    # Own freshly generated fixture only; never load externally supplied pickle.
    with (args.output / f'fixture_rank{rank}.pkl').open('wb') as stream:
        pickle.dump(buffer, stream)
    trainer.update(buffer)
    after_first = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    first_grads = {k: p.grad.detach().cpu().clone() for k, p in model.named_parameters() if p.grad is not None}
    rank_difference = rank_parameter_difference(model, device)
    assert rank_difference < 1e-6, rank_difference
    assert trainer.real_interactions == 2*size
    path = args.output / 'resume.pt'
    trainer.checkpoint(path, {'fixture': True})
    other = PPOTrainer(PFNPolicy(model.config).to(device), config, seed=100)
    assert other.restore(path) == {'fixture': True}
    restore_errors = []
    try:
        validate_restored_state(trainer, other)
    except AssertionError as error:
        restore_errors.append(str(error))
    restore_reports = distributed.gather_json({'rank': rank, 'errors': restore_errors}, device)
    if rank == 0:
        (args.output / 'restore_exact.json').write_text(json.dumps(restore_reports, indent=2), encoding='utf-8')
    if any(report['errors'] for report in restore_reports):
        raise AssertionError('Checkpoint state changed during restore; see restore_exact.json')
    trainer.update(buffer)
    other.update(buffer)
    diagnostics = compare_resumed_update(model, other.model, buffer)
    diagnostics['rank'] = rank
    diagnostics['continued_rank_parameter_difference'] = rank_parameter_difference(model, device)
    diagnostics['restored_rank_parameter_difference'] = rank_parameter_difference(other.model, device)
    if max(diagnostics['continued_rank_parameter_difference'], diagnostics['restored_rank_parameter_difference']) >= 1e-6:
        diagnostics['errors'].append('Cross-rank model synchronization failed after continued/resumed update')
    resume_reports = distributed.gather_json(diagnostics, device)
    resume_difference = max(report['max_parameter_difference'] for report in resume_reports)
    if rank == 0:
        (args.output / 'resume_diagnostics.json').write_text(json.dumps(resume_reports, indent=2), encoding='utf-8')
        print(json.dumps({'checkpoint_restore_exact': 'passed', 'max_resume_parameter_difference': resume_difference,
                          'resume_diagnostics': str(args.output / 'resume_diagnostics.json')}, indent=2), flush=True)
    if any(report['errors'] for report in resume_reports):
        raise AssertionError('Resumed update comparison failed; see resume_diagnostics.json')
    distributed.barrier()
    distributed.finish()
    if rank != 0:
        return
    buffers = []
    for index in range(size):
        with (args.output / f'fixture_rank{index}.pkl').open('rb') as stream:
            buffers.append(pickle.load(stream))
    global_buffer = merge_rollouts(buffers)
    # Compare DDP to one process on the SAME device and the exact same samples.
    reference = PFNPolicy(model.config).to(device)
    reference.load_state_dict(initial)
    serial = PPOTrainer(reference, config, seed=31)
    serial.update(global_buffer)
    maximum_update_difference = 0.0
    maximum_gradient_difference = 0.0
    checks = []
    for name, parameter in reference.named_parameters():
        maximum_update_difference = max(maximum_update_difference, (after_first[name]-parameter.detach().cpu()).abs().max().item())
        if parameter.grad is not None:
            maximum_gradient_difference = max(maximum_gradient_difference, (first_grads[name]-parameter.grad.cpu()).abs().max().item())
            checks.append(tensor_check(first_grads[name], parameter.grad, f'ddp_vs_serial.gradient.{name}',2e-5,2e-4))
        else:
            checks.append(tensor_check(after_first[name], parameter, f'unused_parameter.{name}',0,0))
    updated = PFNPolicy(model.config).to(device)
    updated.load_state_dict(after_first)
    checks.extend(output_checks(updated,reference,global_buffer,'ddp_vs_serial.updated',2e-5,2e-4))
    # Isolate optimizer correctness from CPU/GPU gradient differences: replay the
    # exact captured DDP gradients through one CPU Adam step, including tiny ones.
    replay = PFNPolicy(model.config)
    replay.load_state_dict(initial)
    optimizer = torch.optim.Adam(replay.parameters(),lr=config.learning_rate,foreach=False,fused=False)
    for name, parameter in replay.named_parameters():
        parameter.grad = first_grads.get(name)
    optimizer.step()
    for name, parameter in replay.named_parameters():
        checks.append(tensor_check(after_first[name],parameter,f'adam_same_gradient.{name}',2e-5,2e-4))
    # Keep CPU-vs-device validation separate, at identical initial weights.
    cpu_initial = PFNPolicy(model.config)
    cpu_initial.load_state_dict(initial)
    device_initial = PFNPolicy(model.config).to(device)
    device_initial.load_state_dict(initial)
    checks.extend(output_checks(device_initial,cpu_initial,global_buffer,'cpu_vs_device.initial',2e-4,2e-3))
    cpu_trainer = PPOTrainer(cpu_initial,config,seed=31)
    cpu_trainer.update(global_buffer)
    for name, parameter in cpu_initial.named_parameters():
        if parameter.grad is not None:
            checks.append(tensor_check(first_grads[name],parameter.grad,f'cpu_vs_ddp.gradient.{name}',5e-4,5e-3))
    failures = [check for check in checks if not check['passed']]
    (args.output/'reference_diagnostics.json').write_text(json.dumps({
        'reference_device':str(device), 'samples':'exact rank rollouts, not recollected',
        'checks':checks,'failures':failures,
        'ddp_vs_serial_parameter_difference_informational':maximum_update_difference},indent=2),encoding='utf-8')
    state = torch.load(path, map_location='cpu', weights_only=True)
    assert len(state['rank_rngs']) == size
    assert len({json.dumps(r['numpy_rng'], sort_keys=True) for r in state['rank_rngs']}) == size
    result = {'world_size': size, 'device': str(device), 'backend': 'mccl' if device.type == 'musa' else 'gloo',
              'check_version': 3, 'checkpoint_restore_exact': 'passed',
              'resumed_update_numerical_comparison': 'passed',
              'max_rank_parameter_difference': rank_difference, 'max_resume_parameter_difference': resume_difference,
              'same_data_single_process_gradient_and_update_comparison': 'failed' if failures else 'passed',
              'max_single_process_update_difference': maximum_update_difference,
              'max_single_process_gradient_difference': maximum_gradient_difference,
              'update_comparison_note': 'All Adam parameters checked using identical captured gradients; DDP/serial gradients and updated outputs checked on same device; CPU comparison separate',
              'reference_failure_count':len(failures),
              'global_interactions': 2*size, 'distinct_rank_rngs': size, 'status': 'failed' if failures else 'passed'}
    (args.output / 'check.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
    if failures:
        raise AssertionError('Reference checks failed; see reference_diagnostics.json for all mismatches')

if __name__ == '__main__':
    try:
        main()
    finally:
        distributed.finish()
