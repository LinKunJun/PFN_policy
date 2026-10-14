"""Single-node synchronous PPO support, MCCL for MUSA and Gloo for CPU tests."""
from datetime import timedelta
import json
import os
import numpy as np
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from .runtime import resolve_device

def active():
    return dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1

def world_size():
    return dist.get_world_size() if active() else 1

def rank():
    return dist.get_rank() if active() else 0

def is_main():
    return rank() == 0

def barrier():
    if active():
        if dist.get_backend() == 'mccl':
            # Explicit device also fixes the first barrier before any tensor collective.
            dist.barrier(device_ids=[torch.musa.current_device()])
        else:
            dist.barrier()

def initialize(device):
    size = int(os.environ.get('WORLD_SIZE', '1'))
    if size > 1:
        if str(device) not in ('musa', 'cpu'):
            raise ValueError('Under torchrun use --device musa or cpu without an index')
        local_rank = int(os.environ['LOCAL_RANK'])
        device = resolve_device(f'musa:{local_rank}' if str(device) == 'musa' else 'cpu')
        dist.init_process_group('mccl' if device.type == 'musa' else 'gloo',
                                timeout=timedelta(seconds=int(os.environ.get('PFN_DISTRIBUTED_TIMEOUT_SECONDS', '1800'))))
        return device
    return resolve_device(device)

def finish():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()

def sum_values(values, device):
    tensor = torch.as_tensor(values, dtype=torch.float32, device=device)
    if active():
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor

def max_value(value, device):
    tensor = torch.tensor(float(value), dtype=torch.float32, device=device)
    if active():
        dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
    return tensor.item()


def normalize_advantages(values, device):
    if not active():
        return (values - values.mean()) / max(values.std(), 1e-8)
    count, total = sum_values([len(values), float(values.sum())], device).cpu().tolist()
    mean = total / count
    squared = sum_values([float(np.square(values - mean).sum())], device).item()
    return (values - mean) / max((squared / count) ** 0.5, 1e-8)

def gather_json(value, device):
    """All-gather small metadata using only ordinary FP32 collectives.

    Avoid backend-dependent pickle/object collective device selection. UTF-8 byte
    values 0..255 and these small lengths are exactly representable in FP32.
    """
    if not active():
        return [value]
    data = json.dumps(value, separators=(',', ':')).encode('utf-8')
    size = int(max_value(len(data), device))
    encoded = torch.zeros(size + 1, dtype=torch.float32, device=device)
    encoded[0] = len(data)
    encoded[1:len(data)+1] = torch.tensor(list(data), dtype=torch.float32, device=device)
    gathered = [torch.empty_like(encoded) for _ in range(world_size())]
    dist.all_gather(gathered, encoded)
    decoded = []
    for tensor in gathered:
        array = tensor.cpu().to(torch.int64).tolist()
        decoded.append(json.loads(bytes(array[1:array[0]+1]).decode('utf-8')))
    return decoded

class PreparedPolicy(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, data, plan):
        return self.policy.forward_prepared(data, plan, return_logits=True)

def wrap_policy(policy):
    if not active():
        return None
    adapter = PreparedPolicy(policy)
    kwargs = dict(broadcast_buffers=False, find_unused_parameters=False,
                  gradient_as_bucket_view=True)
    # forward_prepared adds exact-zero parameter edges for conditional heads.
    # PPOTrainer preserves legacy Adam behavior for rollout-wide inactive heads.
    if policy.device.type != 'cpu':
        kwargs.update(device_ids=[policy.device.index], output_device=policy.device.index)
    return DistributedDataParallel(adapter, **kwargs)

def validate_training(settings, ppo):
    size = world_size()
    if settings['tasks_per_update'] < size or settings['tasks_per_update'] % size:
        raise ValueError('Global tasks_per_update must be positive and divisible by world size')
    if ppo.minibatch < size or ppo.minibatch % size:
        raise ValueError('Global PPO minibatch must be positive and divisible by world size')
