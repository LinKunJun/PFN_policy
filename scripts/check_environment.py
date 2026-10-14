"""Fail-fast stack check plus actual project forward/backward/update/save/load."""
import argparse
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='musa')
    parser.add_argument('--stack-only', action='store_true')
    args = parser.parse_args()
    import torch
    from pfn_network.runtime import resolve_device, runtime_info, synchronize
    if args.device.startswith('musa'):
        if sys.version_info[:2] != (3, 10):
            raise RuntimeError('Target image must use Python 3.10')
        import importlib.metadata
        for package in ('torch', 'torch_musa'):
            version = importlib.metadata.version(package)
            if version.split('+')[0].split('.post')[0] != '2.7.1':
                raise RuntimeError(f'{package}={version}; expected vendor 2.7.1 stack')
    device = resolve_device(args.device)
    probe = torch.randn(8, 8, device=device, requires_grad=True)
    (probe @ probe.T).square().mean().backward()
    synchronize(device)
    assert torch.isfinite(probe.grad).all().item()
    if args.stack_only:
        print('Vendor stack import, device access, matmul and backward passed')
        return
    import numpy as np
    from pfn_network.env.dgp import TaskPrior, sample_task
    from pfn_network.models import ModelConfig, PFNPolicy
    from pfn_network.train_eval.train import PPOConfig, PPOTrainer, collect_tasks
    torch.set_num_threads(2)
    model = PFNPolicy(ModelConfig(width=16, heads=2, transformer_layers=1, context_length=4)).to(device)
    trainer = PPOTrainer(model, PPOConfig(epochs=1, minibatch=4), 17)
    prior = TaskPrior(n=6, budget=2, catalog_size=8, horizon=2)
    tasks = [(sample_task(prior, 17, 'train', i, d).env(i), [i, i+10])
             for i, d in enumerate(('FC', 'CD'))]
    rollout = collect_tasks(model, tasks, 2, trainer.action_generator, trainer.config)
    # Compare real project inference and gradients against identical CPU weights.
    reference = PFNPolicy(model.config).eval()
    reference.load_state_dict({k: v.detach().cpu() for k, v in model.state_dict().items()})
    row = len(rollout) - 1
    obs = rollout.observations[row]
    history = [rollout.records[i] for i in rollout.history_indices[row, :rollout.history_lengths[row]]]
    episode = int(rollout.episodes[row])
    actual_dist, actual_value = model(obs, history, episode)
    expected_dist, expected_value = reference(obs, history, episode)
    torch.testing.assert_close(actual_dist.probs.cpu(), expected_dist.probs, atol=2e-4, rtol=2e-3)
    torch.testing.assert_close(actual_value.cpu(), expected_value, atol=2e-4, rtol=2e-3)
    (actual_value.square() - actual_dist.entropy()).backward()
    (expected_value.square() - expected_dist.entropy()).backward()
    for p, q in zip(model.parameters(), reference.parameters()):
        if q.grad is not None:
            torch.testing.assert_close(p.grad.cpu(), q.grad, atol=5e-4, rtol=5e-3)
    model.zero_grad(set_to_none=True)
    metrics = trainer.update(rollout)
    assert np.isfinite(metrics['loss'])
    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / 'resume.pt'
        trainer.checkpoint(path, {'check': True})
        other = PPOTrainer(PFNPolicy(model.config).to(device), trainer.config)
        assert other.restore(path) == {'check': True}
        for x, y in zip(model.parameters(), other.model.parameters()):
            torch.testing.assert_close(x, y)
        torch.testing.assert_close(trainer.action_generator.get_state(), other.action_generator.get_state())
        continued = other.update(rollout)
        assert np.isfinite(continued['loss'])
    synchronize(device)
    print(json.dumps({'runtime': runtime_info(device), 'smoke_metrics': metrics,
                      'status': 'project rollout, PPO, checkpoint and resumed update passed'}, indent=2))

if __name__ == '__main__':
    main()
