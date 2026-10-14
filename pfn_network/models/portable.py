"""GCN and attention using ordinary PyTorch tensor operations.

GCN state keys match PyG GCNConv (lin.weight, bias). Graphs in this project
are unweighted simple undirected graphs; normalization occurs once per plan.
No dense block-diagonal adjacency, sparse extensions or fused attention required.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F

@dataclass
class EdgeAdjacency:
    row: torch.Tensor
    col: torch.Tensor
    value: torch.Tensor
    count: int

    def coo(self):
        return self.row, self.col, self.value

def normalized_adjacency(edges, count):
    # Set diagonal to 1 exactly as gcn_norm(SparseTensor) does.
    row, col = edges[1], edges[0]
    keep = row != col
    loops = torch.arange(count, device=edges.device)
    row, col = torch.cat((row[keep], loops)), torch.cat((col[keep], loops))
    values = torch.ones(row.numel(), dtype=torch.float32, device=edges.device)
    degree = values.new_zeros(count).index_add(0, row, values)
    inv = degree.rsqrt()
    return EdgeAdjacency(row, col, inv[row] * inv[col], count)

class GCNConv(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_channels))
        nn.init.xavier_uniform_(self.lin.weight)

    def forward(self, x, adjacency):
        x = self.lin(x)
        row, col, weight = adjacency.coo()
        messages = x[col] * weight[:, None]
        return x.new_zeros((adjacency.count, x.shape[1])).index_add(0, row, messages) + self.bias

class PortableEncoderLayer(nn.TransformerEncoderLayer):
    """Preserve encoder parameters and pre-norm logic, avoid fused/SDPA kernels."""
    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        if not self.norm_first or self.dropout.p != 0:
            raise ValueError("This project uses pre-norm attention with zero dropout")
        x = self.norm1(src)
        batch, length, width = x.shape
        heads = self.self_attn.num_heads
        dim = width // heads
        q, k, v = F.linear(x, self.self_attn.in_proj_weight, self.self_attn.in_proj_bias).chunk(3, -1)
        q, k, v = [t.reshape(batch, length, heads, dim).transpose(1, 2) for t in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(dim)
        if src_mask is not None:
            scores = scores.masked_fill(src_mask[None, None], float("-inf"))
        elif is_causal:
            mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
            scores = scores.masked_fill(mask, float("-inf"))
        if src_key_padding_mask is not None:
            scores = scores.masked_fill(src_key_padding_mask[:, None, None, :], float("-inf"))
        context = (scores.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, width)
        x = src + self.dropout1(self.self_attn.out_proj(context))
        return x + self._ff_block(self.norm2(x))
