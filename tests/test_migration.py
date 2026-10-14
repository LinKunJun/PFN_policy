import copy
import numpy as np
import pytest
import torch
from pfn_network.models.portable import GCNConv, normalized_adjacency, PortableEncoderLayer
from pfn_network.runtime import resolve_device, action_generator
from pfn_network.models import PFNPolicy, ModelConfig
from pfn_network.env.core import Dynamics
from pfn_network.env.dgp import sample_task, TaskPrior
from pfn_network.train_eval.train import PPOConfig, PPOTrainer, collect_tasks

@pytest.mark.parametrize('empty', [False, True])
def test_gcn_dense_reference_gradients(empty):
    edges = torch.empty((2, 0), dtype=torch.long) if empty else torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    conv = GCNConv(3, 5)
    ref = copy.deepcopy(conv)
    x = torch.randn(4, 3, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a = torch.eye(4)
    a[edges[1], edges[0]] = 1
    d = a.sum(1).rsqrt()
    a = d[:, None] * a * d[None, :]
    expected = a @ ref.lin(x) + ref.bias
    actual = conv(y, normalized_adjacency(edges, 4))
    torch.testing.assert_close(actual, expected)
    expected.square().sum().backward()
    actual.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad)
    for p, q in zip(conv.parameters(), ref.parameters()):
        torch.testing.assert_close(p.grad, q.grad)

def test_attention_matches_torch_forward_backward():
    portable = PortableEncoderLayer(8, 2, 32, dropout=0, batch_first=True, norm_first=True).train()
    standard = torch.nn.TransformerEncoderLayer(8, 2, 32, dropout=0, batch_first=True, norm_first=True).train()
    standard.load_state_dict(portable.state_dict())
    x = torch.randn(2, 5, 8, requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    causal = torch.ones(5, 5, dtype=torch.bool).triu(1)
    padding = torch.tensor([[False]*5, [False, False, False, True, True]])
    a = portable(x, src_mask=causal, src_key_padding_mask=padding)
    b = standard(y, src_mask=causal, src_key_padding_mask=padding)
    torch.testing.assert_close(a, b, atol=2e-6, rtol=2e-5)
    a.square().sum().backward(); b.square().sum().backward()
    torch.testing.assert_close(x.grad, y.grad, atol=1e-5, rtol=2e-5)
    for p, q in zip(portable.parameters(), standard.parameters()):
        torch.testing.assert_close(p.grad, q.grad, atol=2e-5, rtol=2e-5)

def test_cpu_resume_next_update_equivalence(tmp_path):
    torch.manual_seed(8)
    model = PFNPolicy(ModelConfig(width=8, heads=2, context_length=4, transformer_layers=1))
    trainer = PPOTrainer(model, PPOConfig(epochs=1, minibatch=4), seed=11)
    prior = TaskPrior(n=5, budget=2, catalog_size=6, horizon=2)
    tasks = [(sample_task(prior, 12, 'train', i, d).env(i), [1, 2]) for i, d in enumerate(['FC', 'CD'])]
    rollout = collect_tasks(model, tasks, 2, trainer.action_generator, trainer.config)
    trainer.update(rollout)
    path = tmp_path / 'resume.pt'
    trainer.checkpoint(path, {'next_update': 1})
    other = PPOTrainer(PFNPolicy(model.config), trainer.config)
    assert other.restore(path) == {'next_update': 1}
    assert torch.equal(trainer.action_generator.get_state(), other.action_generator.get_state())
    left, right = trainer.update(rollout), other.update(rollout)
    assert left['loss'] == pytest.approx(right['loss'])
    for p, q in zip(model.parameters(), other.model.parameters()):
        torch.testing.assert_close(p, q)
    legacy = torch.load(path, weights_only=True)
    del legacy['action_rng_backend']
    torch.save(legacy, path)
    with pytest.raises(ValueError, match='Legacy'):
        other.restore(path)

def test_device_and_enum_contract():
    assert str(resolve_device('cpu')) == 'cpu'
    with pytest.raises(ValueError):
        resolve_device('cuda')
    assert str(Dynamics.FC) == 'FC' and Dynamics.FC == 'FC'
    assert repr(Dynamics.FC) == "<Dynamics.FC: 'FC'>"
    assert action_generator().device.type == 'cpu'
