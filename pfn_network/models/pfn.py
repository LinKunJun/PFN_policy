"""PFN policy with TF32-compatible recovery for selected MUSA fusion paths.

This is the production model module. Its public model/configuration interfaces,
checkpoint state and ordinary console behavior match the original module.
"""

import os

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import math
import weakref

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Categorical
from .portable import GCNConv, EdgeAdjacency, normalized_adjacency, PortableEncoderLayer
from .incremental_history import IncrementalHistoryCache
from ._ppo_unique import unique_pairs_packed, unique_states_visited
if __package__ and '.' in __package__:
    from ..runtime import resolve_device
else:
    from runtime import resolve_device


@dataclass(frozen=True)
class ModelConfig:
    feature_dim: int = 4
    width: int = 64
    graph_layers: int = 2
    transformer_layers: int = 2
    heads: int = 4
    context_length: int = 24
    joint_head: bool = True
    history: bool = True
    gcn_backend: str = "auto"
    inference_cache_mb: float = 512.0


@dataclass(frozen=True)
class FactualRecord:
    state: np.ndarray
    global_state: np.ndarray
    allocation: np.ndarray
    exposure: np.ndarray
    reward: float
    next_state: np.ndarray
    next_global: np.ndarray
    done: bool
    episode: int
    time: int

    @classmethod
    def from_step(cls, obs, action, reward, next_obs, done, episode):
        return cls(obs["pfn_state"], obs["global_state"], obs["catalog"][action],
                   obs["catalog_exact_exposure"][action], float(reward),
                   next_obs["pfn_state"], next_obs["global_state"], bool(done),
                   episode, obs["episode_time"])


def _stack_catalog(entries, name, max_n, max_m):
    """Preserve catalog stacking order, shape, dtype and padding semantics."""
    tensors = []
    for entry in entries:
        if name == 'scalars':
            pad = (0, 0, 0, max_m - entry['m'])
        else:
            pad = (0, max_n - entry['n'], 0, max_m - entry['m'])
        tensor = entry[name]
        if any(pad):
            tensor = F.pad(tensor, pad)
        tensors.append(tensor)
    return torch.stack(tensors)


# --- Production fused operations. ---

# --- Optional FP32 Triton kernels, with first-order autograd. ---
# All tensor code in the unfused variants remains ordinary PyTorch.
try:
    import triton
    import triton.language as tl
    TRITON_AVAILABLE = True
    TRITON_IMPORT_ERROR = None
except Exception as _triton_import_error:
    triton = tl = None
    TRITON_AVAILABLE = False
    TRITON_IMPORT_ERROR = str(_triton_import_error)

_TRITON_TARGETS = {}


def triton_usable(tensor):
    """Select MUSA kernels without changing the model's console output.

    Unsupported runtimes use ordinary PyTorch. Actual JIT/kernel failures are
    propagated; the separate comparison script reports backend availability.
    """
    if tensor.dtype != torch.float32:
        return False
    if (TRITON_AVAILABLE and os.environ.get("TRITON_INTERPRET") == "1"
            and os.environ.get("PFN_TRITON_INTERPRET_TEST") == "1"):
        return tensor.device.type == "cpu"
    if tensor.device.type != "musa":
        return False
    key = str(tensor.device)
    if key not in _TRITON_TARGETS:
        enabled = False
        if TRITON_AVAILABLE:
            try:
                from triton.runtime.driver import driver
                target = driver.active.get_current_target()
                enabled = str(target.backend).lower() in {"musa", "mtgpu"}
            except Exception:
                pass
        _TRITON_TARGETS[key] = enabled
    return _TRITON_TARGETS[key]


# TF32 support in torch_musa 2.7.x is controlled by the MUSA muDNN flag.
# Reading it is a host-side metadata operation; this module never changes it.
def _musa_tf32_enabled(tensor):
    if tensor.device.type != 'musa':
        return False
    backend = getattr(torch.backends, 'mudnn', None)
    return backend is not None and bool(getattr(backend, 'allow_tf32', False))


def fusion_execution_policy(tensor):
    """Describe actual routing, distinct from Triton backend availability.

    The TF32 MUSA comparison passed with the public triton_usable selector
    disabled. That selector controls softmax, residual norms, candidate GELU,
    and creation of training-history reverse plans. The separate history
    selector remains active in frozen inference. Reproduce precisely that
    route under TF32, while retaining TF32 for every ordinary matrix product.
    No precision flag, activation, parameter or comparison tolerance changes.
    """
    recovery = _musa_tf32_enabled(tensor)
    return {
        'musa_tf32_enabled': recovery,
        'policy': 'tf32_torch_recovery' if recovery else 'original_backend_dispatch',
        'softmax': 'torch' if recovery else 'original_dispatch',
        'residual_norm': 'torch' if recovery else 'original_dispatch',
        'gcn_bias_relu_norm': 'torch' if recovery else 'original_dispatch',
        'candidate_gelu': 'torch' if recovery else 'original_dispatch',
        'training_history_reverse_plan': 'disabled' if recovery else 'original_dispatch',
        'training_history_pool': 'torch_bmm_autograd' if recovery else 'original_dispatch',
        'frozen_history_pool': 'original_dispatch_including_triton',
        'final_query_pruning': 'retained',
        'node_and_history_projection_split': 'retained',
        'candidate_projection_reordering': 'retained',
        'inference_and_catalog_caches': 'retained',
    }


def _mask_metadata(scores, source_mask, key_padding):
    b, h, q, k = scores.shape
    if source_mask is not None:
        if source_mask.dtype != torch.bool:
            raise TypeError("The project's attention masks are boolean, True = blocked")
        if source_mask.device != scores.device:
            raise ValueError("source_mask and scores must be on the same device")
        if source_mask.ndim == 2:
            source_mask = source_mask[None, None]
        if source_mask.ndim != 4:
            raise ValueError("Mask must be [Q,K] or broadcastable [B,H,Q,K]")
        source_mask = source_mask.expand(b, h, q, k)
        strides = tuple(source_mask.stride())
    else:
        strides = (0, 0, 0, 0)
    if key_padding is not None:
        if key_padding.dtype != torch.bool or tuple(key_padding.shape) != (b, k):
            raise ValueError("key_padding must be bool [B,K]")
        if key_padding.device != scores.device:
            raise ValueError("key_padding and scores must be on the same device")
        ps = tuple(key_padding.stride())
    else:
        ps = (0, 0)
    return source_mask, strides, ps


def torch_scaled_masked_softmax(scores, source_mask, key_padding, is_causal, scale):
    result = scores * scale
    if source_mask is not None:
        mask = source_mask[None, None] if source_mask.ndim == 2 else source_mask
        result = result.masked_fill(mask, float("-inf"))
    elif is_causal:
        mask = torch.ones(scores.shape[-2:], device=scores.device,
                          dtype=torch.bool).triu(1)
        result = result.masked_fill(mask, float("-inf"))
    if key_padding is not None:
        result = result.masked_fill(key_padding[:, None, None, :], float("-inf"))
    return result.softmax(-1)


# Replacement source fragment for the softmax portion of fusion_fragment.py.
# Requires torch, triton, tl, TRITON_AVAILABLE, triton_usable,
# _mask_metadata and torch_scaled_masked_softmax in the containing module.
# Real Q/K and all changing mask strides are runtime scalars. Triton must not
# specialize even their alignment/equal-to-one properties: a history growing
# from length 1 to 385 otherwise compiles hundreds of near-identical kernels.
# Only BLOCK, H, scale, and the few mask-mode flags are compile-time constants.


if TRITON_AVAILABLE:
    @triton.jit(do_not_specialize=["Q", "K", "M0", "M1", "M2", "M3", "P0", "P1"])
    def _softmax_fwd_kernel(S, P, MASK, PAD,
                            H: tl.constexpr, Q, K,
                            M0, M1, M2, M3, P0, P1,
                            HAS_MASK: tl.constexpr, HAS_PAD: tl.constexpr,
                            CAUSAL: tl.constexpr, SCALE: tl.constexpr,
                            BLOCK: tl.constexpr):
        # A two-dimensional grid avoids integer division by dynamic Q.
        query = tl.program_id(0).to(tl.int64)
        batch_head = tl.program_id(1).to(tl.int64)
        head = batch_head % H
        batch = batch_head // H
        row = batch_head * Q + query
        # Promote the varying stride products, including key*mask_stride, to
        # int64 before multiplication; large views must not overflow int32.
        keys = tl.arange(0, BLOCK).to(tl.int64)
        legal = keys < K
        if HAS_MASK:
            blocked = tl.load(MASK + batch*M0 + head*M1 + query*M2 + keys*M3,
                              keys < K, other=1)
            legal = legal & (blocked == 0)
        elif CAUSAL:
            legal = legal & (keys <= query)
        if HAS_PAD:
            blocked = tl.load(PAD + batch*P0 + keys*P1, keys < K, other=1)
            legal = legal & (blocked == 0)
        values = tl.load(S + row*K + keys, keys < K, other=0).to(tl.float32)
        values = tl.where(legal, values * SCALE, -float("inf"))
        maximum = tl.max(values, axis=0)
        numer = tl.exp(values - maximum)
        denom = tl.sum(numer, axis=0)
        tl.store(P + row*K + keys, numer / denom, keys < K)

    @triton.jit(do_not_specialize=["Q", "K", "M0", "M1", "M2", "M3", "P0", "P1"])
    def _softmax_bwd_kernel(DP, P, DS, MASK, PAD,
                            H: tl.constexpr, Q, K,
                            M0, M1, M2, M3, P0, P1,
                            HAS_MASK: tl.constexpr, HAS_PAD: tl.constexpr,
                            CAUSAL: tl.constexpr, SCALE: tl.constexpr,
                            BLOCK: tl.constexpr):
        query = tl.program_id(0).to(tl.int64)
        batch_head = tl.program_id(1).to(tl.int64)
        head = batch_head % H
        batch = batch_head // H
        row = batch_head * Q + query
        keys = tl.arange(0, BLOCK).to(tl.int64)
        legal = keys < K
        if HAS_MASK:
            blocked = tl.load(MASK + batch*M0 + head*M1 + query*M2 + keys*M3,
                              keys < K, other=1)
            legal = legal & (blocked == 0)
        elif CAUSAL:
            legal = legal & (keys <= query)
        if HAS_PAD:
            blocked = tl.load(PAD + batch*P0 + keys*P1, keys < K, other=1)
            legal = legal & (blocked == 0)
        probabilities = tl.load(P + row*K + keys, keys < K, other=0).to(tl.float32)
        grad = tl.load(DP + row*K + keys, keys < K, other=0).to(tl.float32)
        dot = tl.sum(probabilities * grad, axis=0)
        result = SCALE * probabilities * (grad - dot)
        # Retain masked_fill's exactly-zero gradient at every blocked element,
        # including the all-blocked row whose softmax probabilities are NaN.
        result = tl.where(legal, result, 0.0)
        tl.store(DS + row*K + keys, result, keys < K)


class _ScaledMaskedSoftmax(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, source_mask, key_padding, is_causal, scale):
        scores = scores.contiguous()
        source_mask, ms, ps = _mask_metadata(scores, source_mask, key_padding)
        b, h, q, k = scores.shape
        out = torch.empty_like(scores)
        if b and h and q:
            _softmax_fwd_kernel[(q, b*h)](
                scores, out, scores if source_mask is None else source_mask,
                scores if key_padding is None else key_padding,
                h, q, k, *ms, *ps, source_mask is not None, key_padding is not None,
                bool(is_causal and source_mask is None), float(scale),
                BLOCK=triton.next_power_of_2(k), num_warps=4)
        ctx.save_for_backward(out, source_mask, key_padding)
        ctx.meta = (h, q, k, ms, ps, bool(is_causal), float(scale))
        return out

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_out):
        out, mask, padding = ctx.saved_tensors
        h, q, k, ms, ps, causal, scale = ctx.meta
        result = torch.empty_like(out)
        if out.numel():
            _softmax_bwd_kernel[(q, out.shape[0]*h)](
                grad_out.contiguous(), out, result,
                out if mask is None else mask, out if padding is None else padding,
                h, q, k, *ms, *ps, mask is not None, padding is not None,
                bool(causal and mask is None), scale,
                BLOCK=triton.next_power_of_2(k), num_warps=4)
        return result, None, None, None, None


def fused_scaled_masked_softmax(scores, source_mask=None, key_padding=None,
                                is_causal=False, scale=1.0):
    if _musa_tf32_enabled(scores) or not triton_usable(scores):
        return torch_scaled_masked_softmax(scores, source_mask, key_padding,
                                           is_causal, scale)
    if scores.ndim != 4:
        raise ValueError("scores must be [B,H,Q,K]")
    if scores.shape[-1] == 0 or scores.shape[-1] > 16384:
        # Kernel capacity must not impose a new model context-length limit.
        return torch_scaled_masked_softmax(scores, source_mask, key_padding,
                                           is_causal, scale)
    return _ScaledMaskedSoftmax.apply(scores, source_mask, key_padding,
                                      is_causal, float(scale))


if TRITON_AVAILABLE:
    @triton.jit
    def _resnorm_fwd_kernel(A, R, BIAS, W, B, SUMMED, Y, MEAN, INV,
                            WIDTH: tl.constexpr, EPS: tl.constexpr,
                            RELU_BIAS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        ok = cols < WIDTH
        a = tl.load(A + row*WIDTH + cols, ok, other=0).to(tl.float32)
        r = tl.load(R + row*WIDTH + cols, ok, other=0).to(tl.float32)
        if RELU_BIAS:
            bias = tl.load(BIAS + cols, ok, other=0).to(tl.float32)
            a = tl.maximum(a + bias, 0.)
        summed = a + r
        mean = tl.sum(tl.where(ok, summed, 0.), axis=0) / WIDTH
        centered = tl.where(ok, summed - mean, 0.)
        var = tl.sum(centered * centered, axis=0) / WIDTH
        inv = tl.rsqrt(var + EPS)
        weight = tl.load(W + cols, ok, other=0).to(tl.float32)
        beta = tl.load(B + cols, ok, other=0).to(tl.float32)
        y = centered * inv * weight + beta
        tl.store(SUMMED + row*WIDTH + cols, summed, ok)
        tl.store(Y + row*WIDTH + cols, y, ok)
        tl.store(MEAN + row, mean)
        tl.store(INV + row, inv)

    @triton.jit
    def _resnorm_dx_kernel(DY, DSUM, SUMMED, A, BIAS, W, MEAN, INV, DA, DR,
                           WIDTH: tl.constexpr, HAS_DSUM: tl.constexpr,
                           RELU_BIAS: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        ok = cols < WIDTH
        x = tl.load(SUMMED + row*WIDTH + cols, ok, other=0).to(tl.float32)
        mean = tl.load(MEAN + row)
        inv = tl.load(INV + row)
        xhat = tl.where(ok, (x - mean) * inv, 0.)
        dy = tl.load(DY + row*WIDTH + cols, ok, other=0).to(tl.float32)
        weight = tl.load(W + cols, ok, other=0).to(tl.float32)
        weighted = dy * weight
        average = tl.sum(weighted, axis=0) / WIDTH
        covariance = tl.sum(weighted * xhat, axis=0) / WIDTH
        ds = (weighted - average - xhat*covariance) * inv
        if HAS_DSUM:
            ds += tl.load(DSUM + row*WIDTH + cols, ok, other=0).to(tl.float32)
        da = ds
        if RELU_BIAS:
            a = tl.load(A + row*WIDTH + cols, ok, other=0).to(tl.float32)
            bias = tl.load(BIAS + cols, ok, other=0).to(tl.float32)
            da = tl.where(a + bias > 0., ds, 0.)
        tl.store(DA + row*WIDTH + cols, da, ok)
        tl.store(DR + row*WIDTH + cols, ds, ok)

    @triton.jit(do_not_specialize=["ROWS"])
    def _resnorm_partial_kernel(DY, SUMMED, MEAN, INV, DA, PW, PB, PA,
                                ROWS, WIDTH: tl.constexpr,
                                RELU_BIAS: tl.constexpr,
                                BR: tl.constexpr, BC: tl.constexpr):
        group = tl.program_id(0).to(tl.int64)
        rows = group*BR + tl.arange(0, BR)
        cols = tl.program_id(1)*BC + tl.arange(0, BC)
        ok = (rows[:, None] < ROWS) & (cols[None, :] < WIDTH)
        x = tl.load(SUMMED + rows[:, None]*WIDTH + cols[None, :], ok, other=0).to(tl.float32)
        dy = tl.load(DY + rows[:, None]*WIDTH + cols[None, :], ok, other=0).to(tl.float32)
        mean = tl.load(MEAN + rows, rows < ROWS, other=0)
        inv = tl.load(INV + rows, rows < ROWS, other=0)
        xhat = (x - mean[:, None])*inv[:, None]
        pw = tl.sum(dy * xhat, axis=0)
        pb = tl.sum(dy, axis=0)
        tl.store(PW + group*WIDTH + cols, pw, cols < WIDTH)
        tl.store(PB + group*WIDTH + cols, pb, cols < WIDTH)
        if RELU_BIAS:
            da = tl.load(DA + rows[:, None]*WIDTH + cols[None, :], ok, other=0).to(tl.float32)
            pa = tl.sum(da, axis=0)
            tl.store(PA + group*WIDTH + cols, pa, cols < WIDTH)

    @triton.jit(do_not_specialize=["GROUPS"])
    def _resnorm_reduce_kernel(PW, PB, PA, DW, DB, DBA,
                               GROUPS, WIDTH: tl.constexpr,
                               RELU_BIAS: tl.constexpr,
                               BG: tl.constexpr, BC: tl.constexpr):
        group_offsets = tl.arange(0, BG).to(tl.int64)
        cols = tl.program_id(0)*BC + tl.arange(0, BC)
        accum_w = tl.full((BC,), 0.0, tl.float32)
        accum_b = tl.full((BC,), 0.0, tl.float32)
        accum_a = tl.full((BC,), 0.0, tl.float32)
        # BG stays min(256, next_power_of_2(GROUPS)) at the existing launch.
        # This bounded loop covers all runtime groups without growing either
        # the register footprint or the number of compiled variants with S.
        for group_start in range(0, GROUPS, BG):
            groups = group_start + group_offsets
            ok = (groups[:, None] < GROUPS) & (cols[None, :] < WIDTH)
            pw = tl.load(PW + groups[:, None]*WIDTH + cols[None, :], ok, other=0)
            pb = tl.load(PB + groups[:, None]*WIDTH + cols[None, :], ok, other=0)
            accum_w += tl.sum(pw, axis=0)
            accum_b += tl.sum(pb, axis=0)
            if RELU_BIAS:
                pa = tl.load(PA + groups[:, None]*WIDTH + cols[None, :], ok, other=0)
                accum_a += tl.sum(pa, axis=0)
        tl.store(DW + cols, accum_w, cols < WIDTH)
        tl.store(DB + cols, accum_b, cols < WIDTH)
        if RELU_BIAS:
            tl.store(DBA + cols, accum_a, cols < WIDTH)


class _FusedResidualNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, aggregate, residual, bias, weight, beta, eps, relu_bias):
        aggregate, residual = aggregate.contiguous(), residual.contiguous()
        # Parameters may be one-dimensional views with a stride greater than
        # one. The device kernels index them densely; copies retain the same
        # element ordering and autograd returns gradients to the original views.
        weight, beta = weight.contiguous(), beta.contiguous()
        bias = None if bias is None else bias.contiguous()
        width, rows = aggregate.shape[-1], aggregate.numel() // aggregate.shape[-1]
        summed, normalized = torch.empty_like(aggregate), torch.empty_like(aggregate)
        mean = torch.empty(rows, dtype=torch.float32, device=aggregate.device)
        inv = torch.empty_like(mean)
        _resnorm_fwd_kernel[(rows,)](
            aggregate, residual, weight if bias is None else bias, weight, beta,
            summed, normalized, mean, inv, width, float(eps), bool(relu_bias),
            BLOCK=triton.next_power_of_2(width), num_warps=4)
        ctx.save_for_backward(aggregate, summed, bias, weight, mean, inv)
        ctx.relu_bias = bool(relu_bias)
        ctx.set_materialize_grads(False)
        return summed, normalized

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_summed, grad_normalized):
        aggregate, summed, bias, weight, mean, inv = ctx.saved_tensors
        if grad_normalized is None:
            # If only SUMMED is consumed, LayerNorm is absent from the used
            # graph: its parameter gradients must remain None (not zeros),
            # otherwise Adam/weight-decay could update an inactive parameter.
            if grad_summed is None:
                return (None,) * 7
            ds = grad_summed.contiguous()
            da = ds
            bias_grad = None
            if ctx.relu_bias:
                da = ds * (aggregate + bias > 0).to(ds.dtype)
                bias_grad = da.reshape(-1, da.shape[-1]).sum(0)
            return da, ds, bias_grad, None, None, None, None
        grad_normalized = grad_normalized.contiguous()
        grad_summed = None if grad_summed is None else grad_summed.contiguous()
        width, rows = summed.shape[-1], summed.numel() // summed.shape[-1]
        da, dr = torch.empty_like(summed), torch.empty_like(summed)
        _resnorm_dx_kernel[(rows,)](
            grad_normalized, summed if grad_summed is None else grad_summed,
            summed, aggregate, weight if bias is None else bias,
            weight, mean, inv, da, dr, width, grad_summed is not None, ctx.relu_bias,
            BLOCK=triton.next_power_of_2(width), num_warps=4)
        groups = triton.cdiv(rows, 128)
        partial_w = torch.empty((groups, width), device=summed.device, dtype=summed.dtype)
        partial_b = torch.empty_like(partial_w)
        partial_a = torch.empty_like(partial_w) if ctx.relu_bias else partial_b
        dw, db = torch.empty_like(weight), torch.empty_like(weight)
        da_bias = torch.empty_like(weight) if ctx.relu_bias else db
        _resnorm_partial_kernel[(groups, triton.cdiv(width, 32))](
            grad_normalized, summed, mean, inv, da, partial_w, partial_b, partial_a,
            rows, width, ctx.relu_bias, BR=128, BC=32, num_warps=4)
        _resnorm_reduce_kernel[(triton.cdiv(width, 8),)](
            partial_w, partial_b, partial_a, dw, db, da_bias,
            groups, width, ctx.relu_bias,
            BG=min(256, triton.next_power_of_2(groups)), BC=8, num_warps=4)
        return da, dr, da_bias if ctx.relu_bias else None, dw, db, None, None


def _check_resnorm(aggregate, residual, weight, beta):
    if aggregate.ndim == 0:
        raise ValueError("Residual LayerNorm requires at least one dimension")
    if aggregate.shape != residual.shape or aggregate.dtype != residual.dtype:
        raise ValueError("Residual and branch must have identical shape and dtype")
    if aggregate.shape[-1] > 4096 or aggregate.numel() == 0:
        raise ValueError("Fused residual LayerNorm requires nonempty width <=4096")
    if (weight.dtype != aggregate.dtype or beta.dtype != aggregate.dtype
            or tuple(weight.shape) != (aggregate.shape[-1],) or beta.shape != weight.shape):
        raise ValueError("LayerNorm parameters must match width/dtype")
    if any(t.device != aggregate.device for t in (residual, weight, beta)):
        raise ValueError("Residual LayerNorm inputs and parameters must share a device")


def fused_residual_layer_norm(branch, residual, weight, bias, eps=1e-5):
    if (_musa_tf32_enabled(branch) or not triton_usable(branch) or branch.numel() == 0
            or (branch.ndim and branch.shape[-1] > 4096)):
        summed = residual + branch
        return summed, F.layer_norm(summed, (summed.shape[-1],), weight, bias, eps)
    _check_resnorm(branch, residual, weight, bias)
    return _FusedResidualNorm.apply(branch, residual, None, weight, bias, eps, False)


def fused_bias_relu_residual_layer_norm(aggregate, bias, residual,
                                        weight, norm_bias, eps=1e-5):
    if (_musa_tf32_enabled(aggregate) or not triton_usable(aggregate) or aggregate.numel() == 0
            or (aggregate.ndim and aggregate.shape[-1] > 4096)):
        return F.layer_norm(residual + F.relu(aggregate + bias),
                            (aggregate.shape[-1],), weight, norm_bias, eps)
    _check_resnorm(aggregate, residual, weight, norm_bias)
    if bias.shape != weight.shape or bias.dtype != aggregate.dtype:
        raise ValueError("GCN bias must match width/dtype")
    if bias.device != aggregate.device:
        raise ValueError("GCN bias and aggregate must share a device")
    return _FusedResidualNorm.apply(aggregate, residual, bias, weight, norm_bias,
                                    eps, True)[1]


class OptimizedPortableEncoderLayer(nn.TransformerEncoderLayer):
    """Same parameters and pre-norm equations as the project encoder.

    ``explicit`` is an eager FP32/FP64 comparison path; ``fused_softmax``
    dispatches to this file's verified Triton softmax and optional residual-LN;
    ``sdpa`` calls PyTorch SDPA, whose actual device kernel is backend-dependent.
    This class never casts inputs or weights to a lower precision.

    forward_last is exact output pruning for a known selected query position.
    It is safe in the final layer when the caller only consumes that position.
    It must not replace an earlier layer whose historical outputs are needed.
    """

    def _get_name(self):
        return "PortableEncoderLayer"

    def __init__(self, *args, attention_backend="explicit",
                 fuse_residual_norm=False, **kwargs):
        super().__init__(*args, **kwargs)
        if attention_backend not in ("explicit", "fused_softmax", "sdpa"):
            raise ValueError("attention_backend must be explicit/fused_softmax/sdpa")
        self.attention_backend = attention_backend
        self.fuse_residual_norm = bool(fuse_residual_norm)

    def _check_configuration(self, src, src_mask, src_key_padding_mask):
        if not self.norm_first or self.dropout.p != 0:
            raise ValueError("This project uses pre-norm attention with zero dropout")
        if src.ndim != 3 or not self.self_attn.batch_first:
            raise ValueError("Expected a batch-first [B, L, D] tensor")
        if src_mask is not None:
            if src_mask.dtype != torch.bool or src_mask.ndim != 2:
                raise TypeError("Project src_mask must be a 2D boolean blocked-position mask")
            if tuple(src_mask.shape) != (src.shape[1], src.shape[1]):
                raise ValueError("src_mask must have shape [L, L]")
        if src_key_padding_mask is not None:
            if src_key_padding_mask.dtype != torch.bool:
                raise TypeError("Project src_key_padding_mask must be boolean")
            if tuple(src_key_padding_mask.shape) != tuple(src.shape[:2]):
                raise ValueError("src_key_padding_mask must have shape [B, L]")

    @staticmethod
    def _broadcast_blocked(src_mask):
        if src_mask is None:
            return None
        return src_mask[None, None] if src_mask.ndim == 2 else src_mask

    def _attention(self, q, k, v, src_mask, key_padding, is_causal):
        dim = q.shape[-1]
        if self.attention_backend == "fused_softmax":
            # Score scaling and mask application are inside the single softmax
            # kernel. No global-memory-sized scaled/masked score temporaries.
            scores = q @ k.transpose(-2, -1)
            probs = fused_scaled_masked_softmax(
                scores, src_mask, key_padding, is_causal,
                1.0 / math.sqrt(dim))
            return probs @ v

        blocked = self._broadcast_blocked(src_mask)
        if blocked is None and is_causal:
            blocked = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool,
                                 device=q.device).triu(1)[None, None]

        if self.attention_backend == "explicit":
            # Preserve the original division, operation ordering, and two mask
            # applications in the pure eager reference-like optimized path.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(dim)
            if blocked is not None:
                scores = scores.masked_fill(blocked, float("-inf"))
            if key_padding is not None:
                scores = scores.masked_fill(key_padding[:, None, None, :], float("-inf"))
            return scores.softmax(-1) @ v

        # SDPA boolean masks have the opposite polarity: True means allowed.
        # A source mask overrides is_causal, exactly as the original elif does.
        if src_mask is None and key_padding is None:
            return F.scaled_dot_product_attention(
                q, k, v, dropout_p=0.0, is_causal=is_causal)
        if key_padding is not None:
            padding = key_padding[:, None, None, :]
            blocked = padding if blocked is None else (blocked | padding)
        allowed = None if blocked is None else ~blocked
        context = F.scaled_dot_product_attention(
            q, k, v, attn_mask=allowed, dropout_p=0.0, is_causal=False)
        if allowed is not None:
            # The project never produces all-blocked rows. For generic callers,
            # retain the original softmax(-inf,...,-inf) NaN semantics instead
            # of silently changing them to SDPA's all-zero result.
            context = context.masked_fill(~allowed.any(-1, keepdim=True), float("nan"))
        return context

    def _finish(self, residual, context):
        branch = self.dropout1(self.self_attn.out_proj(context))
        if self.fuse_residual_norm:
            summed, normalized = fused_residual_layer_norm(
                branch, residual, self.norm2.weight, self.norm2.bias, self.norm2.eps)
        else:
            summed = residual + branch
            normalized = self.norm2(summed)
        return summed + self._ff_block(normalized)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, is_causal=False):
        self._check_configuration(src, src_mask, src_key_padding_mask)
        x = self.norm1(src)
        batch, length, width = x.shape
        heads, dim = self.self_attn.num_heads, width // self.self_attn.num_heads
        q, k, v = F.linear(x, self.self_attn.in_proj_weight,
                           self.self_attn.in_proj_bias).chunk(3, -1)
        q, k, v = [t.reshape(batch, length, heads, dim).transpose(1, 2)
                   for t in (q, k, v)]
        context = self._attention(q, k, v, src_mask, src_key_padding_mask,
                                  is_causal and src_mask is None)
        context = context.transpose(1, 2).reshape(batch, length, width)
        return self._finish(src, context)

    def forward_last(self, src, lengths, src_mask=None,
                     src_key_padding_mask=None, is_causal=False):
        """Return [B,D] for src[b,lengths[b]] without computing unused rows.

        ``lengths`` are query indices (history-record counts), NOT total token
        counts. Every index must be in [0, L-1]. The caller prepares these valid indices
        before uploading tensors; no device-to-host sync occurs here.
        All K/V positions still participate in autograd. No frozen-state cache,
        truncation, detached tensor, or approximation is introduced.
        """
        self._check_configuration(src, src_mask, src_key_padding_mask)
        if lengths.ndim != 1 or lengths.shape[0] != src.shape[0]:
            raise ValueError("lengths must have shape [B]")
        if lengths.dtype != torch.long:
            raise TypeError("lengths must be torch.int64 query indices")
        batch, length, width = src.shape
        heads, dim = self.self_attn.num_heads, width // self.self_attn.num_heads
        batch_index = torch.arange(batch, device=src.device)
        # LayerNorm acts independently on each token; historical K/V need norm1
        # for the full sequence, but Q/out-projection/norm2/FFN need one row only.
        normalized = self.norm1(src)
        query_input = normalized[batch_index, lengths]
        weights, biases = self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        q = F.linear(query_input, weights[:width],
                     None if biases is None else biases[:width])
        k, v = F.linear(normalized, weights[width:],
                        None if biases is None else biases[width:]).chunk(2, -1)
        q = q.reshape(batch, 1, heads, dim).transpose(1, 2)
        k, v = [t.reshape(batch, length, heads, dim).transpose(1, 2)
                for t in (k, v)]
        query_mask = None
        if src_mask is not None:
            query_mask = src_mask.index_select(0, lengths)[:, None, None, :]
        elif is_causal:
            query_mask = (torch.arange(length, device=src.device)[None, :]
                          > lengths[:, None])[:, None, None, :]
        # For a selected query, is_causal=True on non-square SDPA would align
        # to row zero. The exact original row mask above avoids that error.
        context = self._attention(q, k, v, query_mask, src_key_padding_mask, False)
        context = context.transpose(1, 2).reshape(batch, width)
        residual = src[batch_index, lengths]
        return self._finish(residual, context)

_HISTORY_TRITON_AVAILABLE = TRITON_AVAILABLE
_history_triton_usable = triton_usable

_HISTORY_INDEX_DTYPES = (torch.int32, torch.int64)

_HISTORY_COUNT_DTYPES = (
    torch.float32, torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64,
)

def prepare_history_reverse(before_cpu, state_count, device=None):
    """Prepare ``(rowptr, record_indices)`` outside training/capture.

    ``before_cpu`` is the same one-dimensional integer CPU/NumPy array used
    to construct the minibatch's ``before`` tensor. For state ``s``, its
    records are ``record_indices[rowptr[s]:rowptr[s+1]]`` in their original
    order. Duplicate state references are preserved. The output contains
    normal tensors even if this helper is called under ``inference_mode``.

    Accelerator input is rejected rather than silently synchronized. Pass
    the already available CPU plan, and transfer the resulting small index
    tensors once using ``device``.
    """
    if isinstance(before_cpu, torch.Tensor) and before_cpu.device.type != "cpu":
        raise ValueError("prepare_history_reverse requires existing CPU indices")
    state_count = int(state_count)
    if state_count < 0:
        raise ValueError("state_count must be non-negative")
    target = torch.device("cpu") if device is None else torch.device(device)
    with torch.inference_mode(False), torch.no_grad():
        indices = torch.as_tensor(before_cpu, device="cpu")
        if indices.ndim != 1:
            raise ValueError("before_cpu must be one-dimensional")
        if indices.numel() and indices.dtype not in _HISTORY_INDEX_DTYPES:
            raise TypeError("before_cpu must contain integer indices")
        indices = indices.to(dtype=torch.int64).clone()
        if indices.numel() and (int(indices.min()) < 0
                                or int(indices.max()) >= state_count):
            raise ValueError("before_cpu contains an out-of-range state index")
        records = torch.argsort(indices, stable=True)
        degree = torch.bincount(indices, minlength=state_count)
        rowptr = torch.zeros(state_count + 1, dtype=torch.int64)
        rowptr[1:] = degree.cumsum(0)
        return rowptr.to(target).contiguous(), records.to(target).contiguous()

def _validate_history_inputs(node_bank, before, allocations, exposures, counts,
                             reverse_plan):
    if node_bank.ndim != 3:
        raise ValueError("node_bank must have shape [states,nodes,width]")
    if before.ndim != 1 or before.dtype not in _HISTORY_INDEX_DTYPES:
        raise ValueError("before must be a one-dimensional int32/int64 tensor")
    records, nodes = before.numel(), node_bank.shape[1]
    if tuple(allocations.shape) != (records, nodes):
        raise ValueError("allocations must have shape [records,nodes]")
    if tuple(exposures.shape) != (records, nodes):
        raise ValueError("exposures must have shape [records,nodes]")
    if tuple(counts.shape) != (records,):
        raise ValueError("counts must have shape [records]")
    if any(t.device != node_bank.device
           for t in (before, allocations, exposures, counts)):
        raise ValueError("all history pooling tensors must use the same device")
    if reverse_plan is not None:
        if not isinstance(reverse_plan, (tuple, list)) or len(reverse_plan) != 2:
            raise TypeError("reverse_plan must be (rowptr, record_indices)")
        rowptr, record_indices = reverse_plan
        if (rowptr.ndim != 1 or rowptr.numel() != node_bank.shape[0] + 1
                or record_indices.ndim != 1 or record_indices.numel() != records):
            raise ValueError("reverse_plan dimensions do not match this history bank")
        if (rowptr.dtype not in _HISTORY_INDEX_DTYPES
                or record_indices.dtype not in _HISTORY_INDEX_DTYPES):
            raise TypeError("reverse_plan must contain integer index tensors")
        if rowptr.device != node_bank.device or record_indices.device != node_bank.device:
            raise ValueError("reverse_plan must be on node_bank.device")

def history_pool_backend_name(node_bank, before, allocations, exposures, counts,
                              reverse_plan=None):
    """Identify the actual execution path, including the training fallback."""
    _validate_history_inputs(node_bank, before, allocations, exposures, counts,
                             reverse_plan)
    constants = (allocations, exposures, counts)
    needs_reverse = torch.is_grad_enabled() and node_bank.requires_grad
    if (_history_triton_usable(node_bank)
            and allocations.dtype == torch.float32
            and exposures.dtype == torch.float32
            and counts.dtype in _HISTORY_COUNT_DTYPES
            and not any(t.requires_grad for t in constants)
            and (not needs_reverse or reverse_plan is not None)):
        return "triton_indexed_history"
    return "torch_indexed_history"

def torch_indexed_history_pools(node_bank, before, allocations, exposures, counts):
    """Ordinary autograd, with batched pooling for matching floating dtypes.

    The matching-dtype route materializes the gathered bank but avoids the
    two record-by-node-by-feature products. Mixed dtypes retain the original
    elementwise expression so PyTorch's promotion rules remain unchanged.
    """
    past = node_bank[before]
    if (node_bank.is_floating_point()
            and node_bank.dtype == allocations.dtype == exposures.dtype):
        weights = torch.stack((allocations, exposures), dim=1)
        return torch.bmm(weights, past) / counts[:, None, None]
    treated = (past * allocations[..., None]).sum(1) / counts[:, None]
    exposed = (past * exposures[..., None]).sum(1) / counts[:, None]
    return torch.stack((treated, exposed), dim=1)

if _HISTORY_TRITON_AVAILABLE:
    @triton.jit
    def _indexed_history_pool_fwd(
        X, Before, A, E, Count, Out,
        NODES: tl.constexpr, WIDTH: tl.constexpr,
        X0: tl.constexpr, X1: tl.constexpr, X2: tl.constexpr,
        B0: tl.constexpr, A0: tl.constexpr, A1: tl.constexpr,
        E0: tl.constexpr, E1: tl.constexpr, C0: tl.constexpr,
        NODE_BLOCK: tl.constexpr, FEATURE_BLOCK: tl.constexpr,
    ):
        record = tl.program_id(0).to(tl.int64)
        features = (tl.program_id(1).to(tl.int64) * FEATURE_BLOCK
                    + tl.arange(0, FEATURE_BLOCK))
        node_offsets = tl.arange(0, NODE_BLOCK).to(tl.int64)
        state = tl.load(Before + record * B0).to(tl.int64)
        treated = tl.full((FEATURE_BLOCK,), 0.0, tl.float32)
        exposed = tl.full((FEATURE_BLOCK,), 0.0, tl.float32)
        for node_start in range(0, NODES, NODE_BLOCK):
            nodes = node_start + node_offsets
            valid = nodes < NODES
            values = tl.load(
                X + state * X0 + nodes[:, None] * X1 + features[None, :] * X2,
                valid[:, None] & (features[None, :] < WIDTH), other=0.0,
            ).to(tl.float32)
            allocation = tl.load(A + record * A0 + nodes * A1,
                                 valid, other=0.0).to(tl.float32)
            exposure = tl.load(E + record * E0 + nodes * E1,
                               valid, other=0.0).to(tl.float32)
            treated += tl.sum(values * allocation[:, None], axis=0)
            exposed += tl.sum(values * exposure[:, None], axis=0)
        count = tl.load(Count + record * C0).to(tl.float32)
        tl.store(Out + record * (2 * WIDTH) + features,
                 treated / count, features < WIDTH)
        tl.store(Out + record * (2 * WIDTH) + WIDTH + features,
                 exposed / count, features < WIDTH)

    @triton.jit
    def _indexed_history_pool_bwd(
        Grad, A, E, Count, RowPtr, Records, DX,
        NODES: tl.constexpr, WIDTH: tl.constexpr,
        G0: tl.constexpr, G1: tl.constexpr, G2: tl.constexpr,
        A0: tl.constexpr, A1: tl.constexpr,
        E0: tl.constexpr, E1: tl.constexpr, C0: tl.constexpr,
        RP0: tl.constexpr, RI0: tl.constexpr,
        NODE_BLOCK: tl.constexpr, FEATURE_BLOCK: tl.constexpr,
        RECORD_BLOCK: tl.constexpr,
    ):
        state = tl.program_id(0).to(tl.int64)
        nodes = (tl.program_id(1).to(tl.int64) * NODE_BLOCK
                 + tl.arange(0, NODE_BLOCK))
        features = (tl.program_id(2).to(tl.int64) * FEATURE_BLOCK
                    + tl.arange(0, FEATURE_BLOCK))
        position = tl.load(RowPtr + state * RP0).to(tl.int64)
        end = tl.load(RowPtr + (state + 1) * RP0).to(tl.int64)
        record_offsets = tl.arange(0, RECORD_BLOCK).to(tl.int64)
        acc = tl.full((NODE_BLOCK, FEATURE_BLOCK), 0.0, tl.float32)
        # A bounded record/node/feature tile works for arbitrary record counts.
        # Four records per iteration amortize index loads and loop overhead,
        # with at most 4*8*64 FP32 elements per product on the MUSA31 schedule.
        # Each state is handled by disjoint output tiles: no scatter atomics.
        while position < end:
            positions = position + record_offsets
            valid_record = positions < end
            record = tl.load(Records + positions * RI0, valid_record,
                             other=0).to(tl.int64)
            count = tl.load(Count + record * C0, valid_record,
                            other=1.0).to(tl.float32)
            grad_treated = tl.load(
                Grad + record[:, None] * G0 + features[None, :] * G2,
                valid_record[:, None] & (features[None, :] < WIDTH), other=0.0,
            ).to(tl.float32) / count[:, None]
            grad_exposed = tl.load(
                Grad + record[:, None] * G0 + G1 + features[None, :] * G2,
                valid_record[:, None] & (features[None, :] < WIDTH), other=0.0,
            ).to(tl.float32) / count[:, None]
            allocation = tl.load(
                A + record[:, None] * A0 + nodes[None, :] * A1,
                valid_record[:, None] & (nodes[None, :] < NODES), other=0.0,
            ).to(tl.float32)
            exposure = tl.load(
                E + record[:, None] * E0 + nodes[None, :] * E1,
                valid_record[:, None] & (nodes[None, :] < NODES), other=0.0,
            ).to(tl.float32)
            acc += tl.sum(allocation[:, :, None] * grad_treated[:, None, :]
                          + exposure[:, :, None] * grad_exposed[:, None, :], axis=0)
            position += RECORD_BLOCK
        tl.store(DX + (state * NODES + nodes[:, None]) * WIDTH + features[None, :],
                 acc, (nodes[:, None] < NODES) & (features[None, :] < WIDTH))

def _run_indexed_history_pool_fwd(node_bank, before, allocations, exposures, counts):
    records, nodes, width = before.numel(), node_bank.shape[1], node_bank.shape[2]
    output = torch.empty((records, 2, width), dtype=torch.float32, device=node_bank.device)
    if records == 0 or width == 0:
        return output
    feature_block = min(64, triton.next_power_of_2(width))
    node_block = triton.next_power_of_2(max(1, nodes)) if nodes <= 64 else 32
    _indexed_history_pool_fwd[(records, triton.cdiv(width, feature_block))](
        node_bank, before, allocations, exposures, counts, output,
        NODES=nodes, WIDTH=width,
        X0=node_bank.stride(0), X1=node_bank.stride(1), X2=node_bank.stride(2),
        B0=before.stride(0), A0=allocations.stride(0), A1=allocations.stride(1),
        E0=exposures.stride(0), E1=exposures.stride(1), C0=counts.stride(0),
        NODE_BLOCK=node_block, FEATURE_BLOCK=feature_block, num_warps=4,
    )
    return output

class _IndexedHistoryPools(torch.autograd.Function):
    @staticmethod
    def forward(ctx, node_bank, before, allocations, exposures, counts, reverse_plan):
        rowptr, records = reverse_plan
        ctx.save_for_backward(allocations, exposures, counts, rowptr, records)
        ctx.bank_shape = tuple(node_bank.shape)
        return _run_indexed_history_pool_fwd(node_bank, before, allocations, exposures, counts)

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        allocations, exposures, counts, rowptr, records = ctx.saved_tensors
        states, nodes, width = ctx.bank_shape
        grad_nodes = torch.empty(ctx.bank_shape, dtype=torch.float32, device=grad_output.device)
        if states and nodes and width:
            feature_block = min(64, triton.next_power_of_2(width))
            node_block = min(8, triton.next_power_of_2(nodes))
            _indexed_history_pool_bwd[(states, triton.cdiv(nodes, node_block),
                                      triton.cdiv(width, feature_block))](
                grad_output, allocations, exposures, counts, rowptr, records, grad_nodes,
                NODES=nodes, WIDTH=width,
                G0=grad_output.stride(0), G1=grad_output.stride(1), G2=grad_output.stride(2),
                A0=allocations.stride(0), A1=allocations.stride(1),
                E0=exposures.stride(0), E1=exposures.stride(1), C0=counts.stride(0),
                RP0=rowptr.stride(0), RI0=records.stride(0),
                NODE_BLOCK=node_block, FEATURE_BLOCK=feature_block,
                RECORD_BLOCK=4, num_warps=4,
            )
        return grad_nodes, None, None, None, None, None

def indexed_history_pools(node_bank, before, allocations, exposures, counts,
                          reverse_plan=None):
    """Return ``[records,2,width]`` treated/exposed means with exact PFN semantics.

    For each record ``r``, the two outputs are the sums over all stored nodes
    of ``node_bank[before[r]] * allocation/exposure[r]``, divided by ``counts[r]``.
    There is no extra masking, count clamping, deduplication, or normalization.
    Padded nodes must have the same zeros/weights as in the original PFN path.

    FP32 MUSA uses a fused gather and dual reduction. In training, pass the
    reverse index produced from the *same* ``before`` plan to fuse its backward
    without atomics. Without that reverse index, ordinary PyTorch supplies
    training gradients; inference still uses the fused kernel. If allocation,
    exposure, or counts require gradients, or dtypes/devices are unsupported,
    the reference path supplies their ordinary autograd behavior. Shape and
    device checks use metadata only. Index values must already be valid in the
    CPU plan, as checked by ``prepare_history_reverse``.
    """
    route = history_pool_backend_name(node_bank, before, allocations, exposures,
                                      counts, reverse_plan)
    if route == "triton_indexed_history":
        if torch.is_grad_enabled() and node_bank.requires_grad:
            return _IndexedHistoryPools.apply(node_bank, before, allocations,
                                             exposures, counts, reverse_plan)
        return _run_indexed_history_pool_fwd(node_bank, before, allocations, exposures, counts)
    return torch_indexed_history_pools(node_bank, before, allocations, exposures, counts)

# PFN candidate-head epilogue: three pools, normalization, shared context,
# scalar projection and exact (erf) GELU. All matrix products stay in PyTorch.

if TRITON_AVAILABLE:
    @triton.jit
    def _pfn_candidate_gelu_fwd(P0, P1, P2, SHARED, SCALAR, COUNTS,
                                PREACT, Y, TOTAL: tl.constexpr,
                                M: tl.constexpr, H: tl.constexpr,
                                SAVE: tl.constexpr, BLOCK: tl.constexpr):
        ids = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = ids < TOTAL
        batch = ids // (M * H)
        col = ids % H
        a = tl.load(P0 + ids, ok, other=0).to(tl.float32)
        b = tl.load(P1 + ids, ok, other=0).to(tl.float32)
        c = tl.load(P2 + ids, ok, other=0).to(tl.float32)
        n = tl.load(COUNTS + batch, ok, other=1).to(tl.float32)
        shared = tl.load(SHARED + batch * H + col, ok, other=0).to(tl.float32)
        scalar = tl.load(SCALAR + ids, ok, other=0).to(tl.float32)
        value = ((a + b) + c) / n
        value = (value + shared) + scalar
        gelu = (0.5 * value) * (1.0 + tl.erf(value * 0.7071067811865476))
        if SAVE:
            tl.store(PREACT + ids, value, ok)
        tl.store(Y + ids, gelu, ok)

    @triton.jit
    def _pfn_candidate_gelu_bwd(DY, PREACT, COUNTS, DPOOL, DPREACT,
                                TOTAL: tl.constexpr, M: tl.constexpr,
                                H: tl.constexpr, BLOCK: tl.constexpr):
        ids = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        ok = ids < TOTAL
        batch = ids // (M * H)
        value = tl.load(PREACT + ids, ok, other=0).to(tl.float32)
        dy = tl.load(DY + ids, ok, other=0).to(tl.float32)
        n = tl.load(COUNTS + batch, ok, other=1).to(tl.float32)
        derivative = (0.5 * (1.0 + tl.erf(value * 0.7071067811865476))
                      + value * 0.3989422804014327 * tl.exp(-0.5 * value * value))
        du = dy * derivative
        tl.store(DPREACT + ids, du, ok)
        tl.store(DPOOL + ids, du / n, ok)

    @triton.jit
    def _pfn_candidate_shared_bwd(DPREACT, DSHARED, M: tl.constexpr,
                                  H: tl.constexpr, BM: tl.constexpr,
                                  BH: tl.constexpr):
        batch = tl.program_id(0).to(tl.int64)
        cols = tl.program_id(1) * BH + tl.arange(0, BH)
        rows = tl.arange(0, BM)
        total = tl.full((BM, BH), 0, tl.float32)
        for start in range(0, M, BM):
            m = start + rows
            values = tl.load(DPREACT + (batch * M + m[:, None]) * H + cols[None, :],
                             (m[:, None] < M) & (cols[None, :] < H), other=0)
            total += values
        result = tl.sum(total, axis=0)
        tl.store(DSHARED + batch * H + cols, result, cols < H)


class _PFNCandidateGELU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, selected, exposed, majority, shared, scalar, counts):
        b, m, h = selected.shape
        output = torch.empty_like(selected)
        save = any(ctx.needs_input_grad[:5])
        preactivation = torch.empty_like(selected) if save else output
        _pfn_candidate_gelu_fwd[(triton.cdiv(selected.numel(), 256),)](
            selected, exposed, majority, shared, scalar, counts,
            preactivation, output, selected.numel(), m, h, save, 256,
            num_warps=4, enable_fp_fusion=False)
        if save:
            ctx.save_for_backward(preactivation, counts)
        ctx.shape = (b, m, h)
        return output

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_output):
        preactivation, counts = ctx.saved_tensors
        b, m, h = ctx.shape
        grad_output = grad_output.contiguous()
        grad_pool = torch.empty_like(preactivation)
        grad_pre = torch.empty_like(preactivation)
        _pfn_candidate_gelu_bwd[(triton.cdiv(preactivation.numel(), 256),)](
            grad_output, preactivation, counts, grad_pool, grad_pre,
            preactivation.numel(), m, h, 256, num_warps=4, enable_fp_fusion=False)
        grad_shared = None
        if ctx.needs_input_grad[3]:
            grad_shared = preactivation.new_empty((b, h))
            _pfn_candidate_shared_bwd[(b, triton.cdiv(h, 32))](
                grad_pre, grad_shared, m, h,
                min(128, triton.next_power_of_2(m)), 32, num_warps=4)
        return (grad_pool if ctx.needs_input_grad[0] else None,
                grad_pool if ctx.needs_input_grad[1] else None,
                grad_pool if ctx.needs_input_grad[2] else None,
                grad_shared, grad_pre if ctx.needs_input_grad[4] else None, None)


def fused_candidate_gelu(selected, exposed, majority, shared, scalar, counts):
    """Return GELU(((selected+exposed)+majority)/n + shared + scalar).

    Inputs are projected candidate pools [B,M,H], shared history/total-node
    context [B,H], scalar projection [B,M,H] and fixed public node counts [B].
    Learned inputs retain their first-order gradients; shared context's
    broadcast gradient is reduced explicitly. The three equal pool gradients
    share read-only storage, as ordinary expand/add backward tensors can do.
    No parameter values are cached across optimizer steps.
    """
    tensors = (selected, exposed, majority, shared, scalar)
    supported = (not _musa_tf32_enabled(selected) and triton_usable(selected) and selected.ndim == 3
                 and selected.numel() > 0 and not counts.requires_grad
                 and all(t.dtype == torch.float32 and t.device == selected.device
                         and t.is_contiguous() for t in tensors)
                 and exposed.shape == selected.shape == majority.shape == scalar.shape
                 and shared.shape == (selected.shape[0], selected.shape[2])
                 and counts.shape == (selected.shape[0],)
                 # The kernel returns FP32. In particular, FP64 counts would
                 # promote eager division to FP64 and must use that path.
                 and counts.dtype in (torch.float32, torch.int32, torch.int64)
                 and counts.device == selected.device and counts.is_contiguous())
    if not supported:
        candidate = ((selected + exposed) + majority) / counts[:, None, None]
        return F.gelu((candidate + shared[:, None]) + scalar, approximate='none')
    return _PFNCandidateGELU.apply(selected, exposed, majority, shared, scalar, counts)


def optimized_fuse_nodes(fusion, current_nodes, history):
    """Project the shared history once per graph instead of once per node."""
    first = fusion[0]
    width = current_nodes.shape[-1]
    node_term = F.linear(current_nodes, first.weight[:, :width], first.bias)
    history_term = F.linear(history, first.weight[:, width:])
    return fusion[2](fusion[1](node_term + history_term[:, None]))

def optimized_action_logits(node_logit, action_head, nodes, history, z, e, s,
                            scalars, node_mask, counts, valid_mask,
                            joint_head=True, strategy="auto"):
    """Algebraically identical logits, including the shared Bernoulli term.

    log(sigmoid(r)) - log(sigmoid(-r)) = r eliminates B*M*N broadcast
    products. The shared term remains, so this returns the original logits
    themselves, not merely an equivalent categorical distribution.

    For the joint head, unselected-node pooling equals total-node pooling
    minus selected-node pooling. Its first Linear can be distributed over
    the remaining three pools. History and total-node projections are
    shared by all candidates. When N < M, projecting three sets of node
    features before pooling replaces projections of M candidate features.
    Otherwise one packed pool GEMM and one candidate GEMM are used.

    No learned projection is cached: sliced/combined weights remain in the
    autograd graph and update normally after every optimizer step. Float
    rounding differs from the reference; this is not bitwise equivalence.
    """
    if strategy not in ("auto", "node_project", "pool_project"):
        raise ValueError("strategy must be auto, node_project or pool_project")
    raw = node_logit(nodes).squeeze(-1)
    available = node_mask.to(z.dtype)
    logits = torch.bmm(z, raw.unsqueeze(-1)).squeeze(-1)
    logits = logits + (available * F.logsigmoid(-raw)).sum(-1, keepdim=True)
    if joint_head:
        width = nodes.shape[-1]
        first = action_head[0]
        weight = first.weight
        selected_w = weight[:, :width]
        unselected_w = weight[:, width:2 * width]
        exposure_w = weight[:, 2 * width:3 * width]
        majority_w = weight[:, 3 * width:4 * width]
        history_w = weight[:, 4 * width:5 * width]
        scalar_w = weight[:, 5 * width:]
        selected_delta_w = selected_w - unselected_w
        total_nodes = torch.bmm(available.unsqueeze(1), nodes).squeeze(1)
        total_mean = total_nodes / counts[:, None]
        shared = F.linear(total_mean, unselected_w) + F.linear(history, history_w)
        use_nodes = (strategy == "node_project" or
                     (strategy == "auto" and nodes.shape[1] < z.shape[1]))
        if use_nodes:
            # Each [B,N,H] split is a valid strided matrix, without a transpose
            # or a B*3 reshape that would require an extra contiguous copy.
            projected = F.linear(nodes, torch.cat(
                (selected_delta_w, exposure_w, majority_w), dim=0))
            selected_nodes, exposed_nodes, majority_nodes = projected.split(width, dim=-1)
            selected = torch.bmm(z, selected_nodes)
            exposed = torch.bmm(e, exposed_nodes)
            majority = torch.bmm(s, majority_nodes)
            scalar_projection = F.linear(scalars, scalar_w, first.bias)
            if isinstance(action_head[1], nn.GELU) and action_head[1].approximate == 'none':
                # The three pool sums, normalization, shared history, scalar
                # addition and exact GELU share one device pass. Backward
                # reduces the shared-context gradient without atomics.
                activated = fused_candidate_gelu(
                    selected, exposed, majority, shared, scalar_projection, counts)
            else:
                candidate = ((selected + exposed) + majority) / counts[:, None, None]
                activated = action_head[1](candidate + shared[:, None] + scalar_projection)
        else:
            pools = torch.bmm(torch.cat((z, e, s), dim=1), nodes)
            selected_pool, exposed_pool, majority_pool = pools.split(z.shape[1], dim=1)
            candidate = F.linear(
                torch.cat((selected_pool, exposed_pool, majority_pool), dim=-1) /
                counts[:, None, None],
                torch.cat((selected_delta_w, exposure_w, majority_w), dim=1))
            first_output = (candidate + shared[:, None] +
                            F.linear(scalars, scalar_w, first.bias))
            activated = action_head[1](first_output)
        joint = action_head[2](activated).squeeze(-1)
        logits = logits + joint
    return logits.masked_fill(~valid_mask, -1e9)

def _enable_model_musa_tf32(model):
    """Apply the validated MUSA precision default without changing model APIs."""
    parameter = next(model.parameters(), None)
    if parameter is None or parameter.device.type != 'musa':
        return
    backend = getattr(torch.backends, 'mudnn', None)
    if backend is None or not hasattr(backend, 'allow_tf32'):
        raise RuntimeError('MUSA TF32 requires torch.backends.mudnn.allow_tf32')
    backend.allow_tf32 = True


class PFNPolicy(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        if config.gcn_backend not in ('auto', 'dense', 'edge'):
            raise ValueError('gcn_backend must be auto, dense, or edge')
        if not math.isfinite(config.inference_cache_mb) or config.inference_cache_mb <= 0:
            raise ValueError('inference_cache_mb must be positive and finite')
        self._static_cache = OrderedDict()
        self._inference_active = False
        self._inference_bank = None
        self._inference_key_memo = None
        self._inference_content_ids = None
        self._inference_content_next_id = 0
        self._incremental_history_cache = None
        self._position_cache = None
        self._causal_cache = None
        self._inference_cache_limit = 512
        self._static_cache_limit = 32
        h = config.width
        self.input = nn.Sequential(nn.Linear(config.feature_dim + 11, h), nn.ReLU(), nn.Linear(h, h))
        self.global_input = nn.Linear(7, h)
        self.gnn = nn.ModuleList([GCNConv(h, h, cached=False, normalize=False, add_self_loops=False) for _ in range(config.graph_layers)])
        self.norms = nn.ModuleList([nn.LayerNorm(h) for _ in range(config.graph_layers)])
        self.record_token = nn.Sequential(
            nn.Linear(4 * h + 4, h), nn.GELU(), nn.Linear(h, h))
        self.query_token = nn.Linear(2 * h + 2, h)
        # Separate construction initializes each layer independently. No dropout:
        # old/new PPO probabilities differ only through parameter updates.
        self.transformer = nn.ModuleList([
            PortableEncoderLayer(h, config.heads, 4 * h, dropout=0.0,
                                       batch_first=True, norm_first=True)
            for _ in range(config.transformer_layers)])
        self.fusion = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.LayerNorm(h))
        self.node_logit = nn.Linear(h, 1)
        self.action_head = nn.Sequential(nn.Linear(5 * h + 3, h), nn.GELU(), nn.Linear(h, 1))
        self.value_head = nn.Sequential(nn.Linear(2 * h, h), nn.GELU(), nn.Linear(h, 1))
        for layer in self.transformer:
            layer.__class__ = OptimizedPortableEncoderLayer
            layer.attention_backend = 'fused_softmax'
            layer.fuse_residual_norm = True
        _enable_model_musa_tf32(self)

    @contextmanager
    def _rollout_range(self, name):
        if getattr(self, '_rollout_trace_active', False):
            with torch.profiler.record_function(name):
                yield
        else:
            yield

    @property
    def device(self):
        return next(self.parameters()).device

    def _tensor(self, value, dtype=torch.float32):
        # as_tensor avoids a second host copy; immutable inputs are never mutated.
        if isinstance(value, np.ndarray) and not value.flags.writeable:
            value = value.copy()
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _apply(self, fn, recurse=True):
        result = super()._apply(fn, recurse=recurse)
        _enable_model_musa_tf32(self)
        self._static_cache.clear()
        self._inference_active = False
        self._inference_bank = None
        self._inference_key_memo = None
        self._inference_content_ids = None
        self._inference_content_next_id = 0
        self._incremental_history_cache = None
        self._position_cache = None
        self._causal_cache = None
        return result

    @contextmanager
    def cached_inference(self):
        """Cache parameter-independent inference outputs only while weights are frozen."""
        previous_active = self._inference_active
        previous_bank = self._inference_bank
        previous_keys = self._inference_key_memo
        previous_content_ids = self._inference_content_ids
        previous_content_next_id = self._inference_content_next_id
        previous_history = self._incremental_history_cache
        self._inference_active = True
        self._inference_bank = None
        self._inference_key_memo = {}
        self._inference_content_ids = {}
        self._inference_content_next_id = 0
        self._incremental_history_cache = (
            IncrementalHistoryCache(self.config.context_length)
            if self.config.history and self.config.context_length > 0 else None)
        try:
            yield
        finally:
            self._inference_active = previous_active
            self._inference_bank = previous_bank
            self._inference_key_memo = previous_keys
            self._inference_content_ids = previous_content_ids
            self._inference_content_next_id = previous_content_next_id
            self._incremental_history_cache = previous_history

    def _normalized_adjacency(self, edges, count):
        return normalized_adjacency(edges, count)

    @torch.inference_mode(False)
    @torch.no_grad()
    def _static(self, obs):
        # These constants survive rollout scopes and may be saved by autograd
        # during a later PPO update. Keep normal tensors even when the caller
        # currently runs under inference_mode; preserve all cache identities.
        # Catalog references prevent Python object-id reuse while an entry is cached.
        # Only public, parameter-independent tensors are retained between forwards.
        key = (id(obs["catalog"]), obs["edge_index"].tobytes())
        if key in self._static_cache:
            self._static_cache.move_to_end(key)
            return key, self._static_cache[key]
        z = np.asarray(obs["catalog"], dtype=np.float32)
        e = np.asarray(obs["catalog_exact_exposure"], dtype=np.float32)
        s = np.asarray(obs["catalog_majority_exposure"], dtype=np.float32)
        edges = obs["edge_index"]
        edge_fraction = (z[:, edges[0]] * z[:, edges[1]]).sum(1) / max(1, edges.shape[1])
        entry = {"key": key, "reference": obs["catalog"], "n": z.shape[1], "m": z.shape[0],
                 "edges": self._tensor(edges, torch.long),
                 "z": self._tensor(z), "e": self._tensor(e), "s": self._tensor(s),
                 "scalars": self._tensor(np.stack((z.mean(1), e.mean(1), edge_fraction), axis=-1))}
        coo = normalized_adjacency(torch.from_numpy(np.asarray(edges).copy()), entry['n'])
        entry['adj_t'] = EdgeAdjacency(coo.row.to(self.device), coo.col.to(self.device),
                                       coo.value.to(self.device), coo.count)
        if self._dense_graphs(entry['n']):
            adjacency = np.eye(entry['n'], dtype=np.float32)
            adjacency[edges[1], edges[0]] = 1
            inv = adjacency.sum(1).astype(np.float32) ** -0.5
            entry['dense_cpu'] = inv[:, None] * adjacency * inv[None, :]
            entry['dense'] = self._tensor(entry['dense_cpu'])
        self._static_cache[key] = entry
        while len(self._static_cache) > self._static_cache_limit:
            self._static_cache.popitem(last=False)
        return key, entry

    def _dense_graphs(self, max_n):
        return self.config.gcn_backend == 'dense' or (self.config.gcn_backend == 'auto' and max_n <= 128)

    def _encode_unique(self, states, globals_, graph_slots, static, key_plan=None):
        """Use one contiguous inference GPU bank; never cache gradient-bearing outputs.

        For variable-node graphs or an insufficient inference budget, directly
        compute the same features without caching. This is a shape/budget
        constraint, not a selectable legacy optimization implementation.
        """
        if not self._inference_active or torch.is_grad_enabled():
            return self._encode_uncached(states, globals_, graph_slots, static)
        packed = self._encode_unique_bank(states, globals_, graph_slots, static, key_plan)
        return packed if packed is not None else self._encode_uncached(
            states, globals_, graph_slots, static)

    def _intern_inference_content(self, key):
        """Intern exact content tuples once; bank lookups then use integer IDs.

        Tuple equality still resolves all hash collisions. IDs only live inside
        the frozen inference scope and do not enter model/checkpoint state.
        """
        content_ids = self._inference_content_ids
        if content_ids is None:
            return key
        value = content_ids.get(key)
        if value is None:
            value = self._inference_content_next_id
            self._inference_content_next_id += 1
            content_ids[key] = value
        return value

    def _prune_inference_content(self):
        """Bound CPU metadata while preserving every live GPU-bank identity.

        Pruning occurs between forwards, never halfway through a live-key list.
        IDs are monotonic, so pruning cannot alias an older occupied bank slot.
        """
        content_ids = self._inference_content_ids
        limit = max(65536, 4 * self._inference_cache_limit)
        if content_ids is None or len(content_ids) <= limit:
            return
        bank = self._inference_bank
        live = set(bank['lru']) if bank is not None else set()
        self._inference_content_ids = {
            key: value for key, value in content_ids.items() if value in live}
        if self._inference_key_memo is not None:
            self._inference_key_memo.clear()

    def _bank_content_key(self, state, global_state, static_key):
        """Reuse IDs only for owning read-only snapshots; re-read other inputs."""
        memo = self._inference_key_memo
        if (memo is None or not isinstance(state, np.ndarray)
                or not isinstance(global_state, np.ndarray)
                or state.flags.writeable or global_state.flags.writeable
                or not state.flags.owndata or not global_state.flags.owndata):
            return self._intern_inference_content(
                (static_key, state.tobytes(), global_state.tobytes()))
        identity = (static_key, id(state), id(global_state))
        existing = memo.get(identity)
        if (existing is not None and existing[0]() is state
                and existing[1]() is global_state):
            return existing[2]
        key = self._intern_inference_content(
            (static_key, state.tobytes(), global_state.tobytes()))
        memo[identity] = (weakref.ref(state), weakref.ref(global_state), key)
        if len(memo) > max(65536, 4 * self._inference_cache_limit):
            memo.clear()
        return key

    def _encode_unique_bank(self, states, globals_, graph_slots, static, key_plan=None):
        """Reuse one contiguous inference-only GPU bank instead of concatenating
        thousands of tiny per-state cache tensors on *every* rollout step.

        Fixed-size graphs are the common case. Irregular shapes use the
        general non-cached encoder instead of a fixed-node bank.
        Model weights must stay fixed throughout cached_inference().
        """
        if not states:
            return None
        nodes = key_plan['nodes'] if key_plan is not None else len(states[0])
        if (nodes == 0 or (key_plan is None and any(len(state) != nodes for state in states))
                or self._inference_cache_limit <= 0):
            return None

        self._prune_inference_content()
        if key_plan is not None and key_plan['epoch'] is self._inference_content_ids:
            keys, first = key_plan['keys'], key_plan['first']
        else:
            # Pruning can replace the content table after the plan was built.
            # Fall back for this call; the next plan rebuilds against the new table.
            # Stable content identities allow safe reuse while parameters are frozen.
            first = {}
            keys = []
            append_key, set_first = keys.append, first.setdefault
            content_key = self._bank_content_key
            memo = self._inference_key_memo
            memo_get = memo.get if memo is not None else None
            array_type = np.ndarray
            for index, (state, global_state, slot) in enumerate(zip(states, globals_, graph_slots)):
                static_key = static[slot]['key']
                existing = None
                # Inline the common memo hit without one Python method call per
                # historical state. Re-read flags and verify BOTH weakrefs each time.
                # Subclasses and all other inputs keep the exact content-key path.
                if (memo_get is not None and type(state) is array_type
                        and type(global_state) is array_type):
                    state_flags, global_flags = state.flags, global_state.flags
                    if (not state_flags.writeable and not global_flags.writeable
                            and state_flags.owndata and global_flags.owndata):
                        existing = memo_get((static_key, id(state), id(global_state)))
                        if (existing is not None and
                                (existing[0]() is not state or existing[1]() is not global_state)):
                            existing = None
                key = existing[2] if existing is not None else content_key(state, global_state, static_key)
                append_key(key)
                set_first(key, index)
        if len(first) > self._inference_cache_limit:
            return None  # All live keys must fit so an eviction cannot corrupt a gather.

        bank = self._inference_bank
        if bank is None:
            capacity = int(self._inference_cache_limit)
            width = self.config.width
            dtype = next(self.parameters()).dtype
            # The same frozen-weight cache can span inference_mode and no_grad
            # calls. A normal backing tensor is safe to update in either mode.
            with torch.inference_mode(False), torch.no_grad():
                bank = {
                    'nodes_per_state': nodes,
                    'capacity': capacity,
                    'nodes': torch.empty((capacity, nodes, width),
                                         dtype=dtype, device=self.device),
                    'pooled': torch.empty((capacity, width),
                                          dtype=dtype, device=self.device),
                    'lru': OrderedDict(),
                    'free': list(range(capacity - 1, -1, -1)),
                }
            self._inference_bank = bank
        if bank['nodes_per_state'] != nodes or len(first) > bank['capacity']:
            return None

        lru = bank['lru']
        # Move all live hits to the newest end *before* choosing eviction slots.
        for key in first:
            if key in lru:
                lru.move_to_end(key)
        missing = [key for key in first if key not in lru]
        if missing:
            required = set(first)
            slots = []
            for _ in missing:
                if bank['free']:
                    slot = bank['free'].pop()
                else:
                    old_key, slot = lru.popitem(last=False)
                    if old_key in required:
                        # Defensive recovery if a future change breaks the
                        # capacity invariant; never return corrupted features.
                        self._inference_bank = None
                        return None
                slots.append(slot)
            ids = [first[key] for key in missing]
            new_nodes, new_pooled, _, _ = self._encode_uncached(
                [states[i] for i in ids], [globals_[i] for i in ids],
                [graph_slots[i] for i in ids], static)
            slot_tensor = self._tensor(np.asarray(slots, dtype=np.int64), torch.long)
            bank['nodes'].index_copy_(
                0, slot_tensor, new_nodes.reshape(len(ids), nodes, self.config.width))
            bank['pooled'].index_copy_(0, slot_tensor, new_pooled)
            for key, slot in zip(missing, slots):
                lru[key] = slot

        indices = self._tensor(
            np.fromiter((lru[key] for key in keys), dtype=np.int64, count=len(keys)),
            torch.long)
        flat = bank['nodes'].index_select(0, indices).reshape(-1, self.config.width)
        pooled = bank['pooled'].index_select(0, indices)
        counts = np.full(len(keys), nodes, dtype=np.int64)
        offsets = np.arange(len(keys), dtype=np.int64) * nodes
        return flat, pooled, offsets, counts

    def _encode_uncached(self, states, globals_, graph_slots, static):
        counts_np = np.array([len(state) for state in states], dtype=np.int64)
        offsets_np = np.r_[0, np.cumsum(counts_np)[:-1]]
        # Small batched GEMMs avoid edge-message tensors and atomic scatter
        # reductions. No dense block diagonal is built.
        if self._dense_graphs(int(counts_np.max())):
            max_n = int(counts_np.max())
            raw = np.zeros((len(states), max_n, states[0].shape[1]), dtype=np.float32)
            for row, state in enumerate(states):
                raw[row, :len(state)] = state
            adjacency = torch.stack([
                static[slot]['dense'] if static[slot]['n'] == max_n
                else F.pad(static[slot]['dense'],
                           (0, max_n-static[slot]['n'], 0, max_n-static[slot]['n']))
                for slot in graph_slots])
            x = self.input(self._tensor(raw))
            for conv, norm in zip(self.gnn, self.norms):
                aggregate = torch.bmm(adjacency, conv.lin(x))
                x = fused_bias_relu_residual_layer_norm(
                    aggregate, conv.bias, x, norm.weight, norm.bias, norm.eps)
            if np.all(counts_np == max_n):
                pooled = x.mean(1)
                flat = x.reshape(-1, self.config.width)
            else:
                mask = self._tensor(np.arange(max_n)[None] < counts_np[:, None], torch.bool)
                pooled = (x * mask[:, :, None]).sum(1) / self._tensor(counts_np)[:, None]
                indices = np.concatenate([np.arange(size) + row * max_n for row, size in enumerate(counts_np)])
                flat = x.reshape(-1, self.config.width)[self._tensor(indices, torch.long)]
            pooled = pooled + self.global_input(self._tensor(np.stack(globals_)))
            return flat, pooled, offsets_np, counts_np
        offsets = self._tensor(offsets_np, torch.long)
        counts = self._tensor(counts_np, torch.long)
        x = self.input(self._tensor(np.concatenate(states, axis=0)))
        graph_slots = np.asarray(graph_slots)
        rows, columns, weights = [], [], []
        for slot, entry in enumerate(static):
            ids = self._tensor(np.flatnonzero(graph_slots == slot), torch.long)
            row, col, value = entry['adj_t'].coo()
            rows.append((row[None] + offsets[ids, None]).reshape(-1))
            columns.append((col[None] + offsets[ids, None]).reshape(-1))
            weights.append(value[None].expand(ids.numel(), -1).reshape(-1))
        adjacency = EdgeAdjacency(torch.cat(rows), torch.cat(columns), torch.cat(weights), x.shape[0])
        for conv, norm in zip(self.gnn, self.norms):
            x = norm(x + F.relu(conv(x, adjacency)))
        membership = torch.repeat_interleave(torch.arange(len(states), device=self.device), counts,
                                              output_size=x.shape[0])
        pooled = x.new_zeros((len(states), self.config.width)).index_add(0, membership, x)
        pooled = pooled / counts[:, None]
        pooled = pooled + self.global_input(self._tensor(np.stack(globals_)))
        return x, pooled, offsets_np, counts_np

    @torch.inference_mode(False)
    @torch.no_grad()
    def _position(self, length, width):
        if self._position_cache is not None and self._position_cache.shape[0] >= length:
            return self._position_cache[:length]
        requested = length
        length = max(length, self.config.context_length + 1)
        pos = torch.arange(length, device=self.device)[:, None]
        freq = torch.exp(torch.arange(0, width, 2, device=self.device) * (-math.log(10000.0) / width))
        encoding = torch.zeros(length, width, device=self.device)
        encoding[:, 0::2] = torch.sin(pos * freq)
        encoding[:, 1::2] = torch.cos(pos * freq[:width // 2])
        self._position_cache = encoding
        return encoding[:requested]

    @torch.inference_mode(False)
    @torch.no_grad()
    def _causal(self, length):
        if self._causal_cache is None or self._causal_cache.shape[0] < length:
            capacity = max(length, self.config.context_length + 1)
            self._causal_cache = torch.ones(capacity, capacity, device=self.device, dtype=torch.bool).triu(1)
        return self._causal_cache[:length, :length]

    def _forward_many(self, observations, histories, episodes, history_slots=None):
        batch = len(observations)
        limit = self.config.context_length
        histories = [h[-limit:] if self.config.history and limit > 0 else () for h in histories]
        self._static_cache_limit = max(32, 2 * batch)
        # A 64-task, 384-record rollout needs thousands of states. The old fixed
        # 512-entry LRU thrashed. Bound encoded features by an explicit MiB budget.
        max_n = max(len(obs['pfn_state']) for obs in observations)
        entry_bytes = (max_n + 1) * self.config.width * next(self.parameters()).element_size()
        budget_entries = max(0, int(self.config.inference_cache_mb * 1024**2 / entry_bytes))
        required = max(512, batch * ((limit if self.config.history else 0) + 2))
        self._inference_cache_limit = min(required, budget_entries)
        static, static_ids, slots = [], {}, []
        with self._rollout_range('PFN/CPU_static_catalogs'):
            for obs in observations:
                if not np.any(obs["valid_action_mask"]):
                    raise ValueError("A terminal observation has no action distribution.")
                key, entry = self._static(obs)
                if key not in static_ids:
                    static_ids[key] = len(static)
                    static.append(entry)
                slots.append(static_ids[key])

        # Reuse identical state objects within this forward only. Gradients from
        # all their uses accumulate through the same current-parameter GNN graph.
        states, globals_, graph_slots, state_ids = [], [], [], {}
        # Inference-only packing avoids affecting gradient-enabled forward paths.
        inference_only = not torch.is_grad_enabled()
        def register(state, global_state, slot):
            key = (id(state), id(global_state), slot)
            existing = state_ids.get(key)
            if existing is not None:
                return existing
            index = len(states)
            state_ids[key] = index
            states.append(state)
            globals_.append(global_state)
            graph_slots.append(slot)
            return index

        current, before, after, records, feedback = [], [], [], [], []
        incremental_pack = None
        with self._rollout_range('PFN/CPU_build_histories'):
            if (history_slots is not None and self._incremental_history_cache is not None
                    and not torch.is_grad_enabled()):
                incremental_pack = self._incremental_history_cache.pack(
                    observations, histories, episodes, history_slots, static, slots,
                    max(entry['n'] for entry in static))
            if incremental_pack is not None:
                # All older FactualRecords have been incrementally indexed and
                # packed in bounded per-task rings. Keep the exact original
                # record order, time feedback, and per-forward state dedup.
                states = incremental_pack['states']
                globals_ = incremental_pack['globals']
                graph_slots = incremental_pack['graph_slots']
                current = incremental_pack['current']
                before = incremental_pack['before']
                after = incremental_pack['after']
                feedback = incremental_pack['feedback']
                record_count = len(before)
            else:
                for obs, history, episode, slot in zip(observations, histories, episodes, slots):
                    current.append(register(obs["pfn_state"], obs["global_state"], slot))
                    for record in history:
                        before.append(register(record.state, record.global_state, slot))
                        after.append(register(record.next_state, record.next_global, slot))
                        records.append(record)
                        scalars = [record.reward, record.done, record.time / 20,
                                   (record.episode - episode) / 10]
                        feedback.append(scalars)
                record_count = len(records)
        key_plan = None
        snapshots = getattr(self, '_sealed_rollout_snapshots', None)
        if (incremental_pack is not None and self._inference_active
                and inference_only and snapshots is not None):
            key_plan = self._incremental_history_cache.bank_plan(
                incremental_pack['registry_ids'], self._bank_content_key,
                self._inference_content_ids, snapshots)
        with self._rollout_range('PFN/GNN_encode_unique'):
            x, pooled, offsets, counts = self._encode_unique(
                states, globals_, graph_slots, static, key_plan)
        max_n = max(entry["n"] for entry in static)
        max_m = max(entry["m"] for entry in static)
        # All rollout graphs have the same number of nodes in the common case.
        # The original node_indices() builds an N x max_n int64 tensor via a
        # Python loop, even when every row is simply a consecutive node range.
        # For inference only, reshape the already ordered flat GNN output, then
        # select by *state*; retain the original gather in gradient-enabled code
        # so no changes to backpropagation or its accumulation order are possible.
        # index. This transfers N indices rather than N*max_n indices, without
        # changing any node values or trainable computation.
        uniform_nodes = bool(len(counts) and np.all(counts == max_n))
        # Retain the original graph-connected feature bank before the generic
        # gather below appends its padding sentinel. History pooling can read
        # this bank directly; it is a per-forward view, never a feature cache.
        history_node_bank = (x.reshape(len(states), max_n, self.config.width)
                             if uniform_nodes else None)
        if uniform_nodes and not torch.is_grad_enabled():
            grouped_nodes = history_node_bank

            def gather_nodes(ids):
                return grouped_nodes.index_select(
                    0, self._tensor(np.asarray(ids, dtype=np.int64), torch.long))
        else:
            sentinel = x.shape[0]
            x = torch.cat((x, x.new_zeros((1, self.config.width))), dim=0)

            def gather_nodes(ids):
                indices = np.full((len(ids), max_n), sentinel, dtype=np.int64)
                for row, index in enumerate(ids):
                    indices[row, :counts[index]] = offsets[index] + np.arange(counts[index])
                return x[self._tensor(indices, torch.long)]

        current_ids = self._tensor(current, torch.long)
        with self._rollout_range('PFN/Gather_current_nodes'):
            current_nodes = gather_nodes(current)
        n = self._tensor(counts[current])
        node_mask = torch.arange(max_n, device=self.device)[None] < n[:, None]
        current_mean = current_nodes.sum(1) / n[:, None]
        times = self._tensor([[obs["episode_time"] / 20, 0.0] for obs in observations])
        query = self.query_token(torch.cat((pooled[current_ids], current_mean, times), dim=-1))
        if record_count:
            if incremental_pack is not None:
                allocation = incremental_pack['allocations']
                exposure = incremental_pack['exposures']
            else:
                allocation = None
                if inference_only and uniform_nodes:
                    # For standard fixed-size graphs, convert whole columns in
                    # NumPy instead of doing two Python slice assignments per row.
                    # Check every length so irregular inputs retain baseline rules.
                    allocations = [record.allocation for record in records]
                    exposures = [record.exposure for record in records]
                    if all(len(a) == max_n and len(e) == max_n
                           for a, e in zip(allocations, exposures)):
                        allocation = np.asarray(allocations, dtype=np.float32)
                        exposure = np.asarray(exposures, dtype=np.float32)
                if allocation is None:
                    allocation = np.zeros((record_count, max_n), dtype=np.float32)
                    exposure = np.zeros_like(allocation)
                    for row, record in enumerate(records):
                        size = len(record.allocation)
                        allocation[row, :size] = record.allocation
                        exposure[row, :size] = record.exposure
            before_ids = self._tensor(before, torch.long)
            sizes = self._tensor(counts[before])
            allocation_tensor, exposure_tensor = self._tensor(allocation), self._tensor(exposure)
            with self._rollout_range('PFN/Gather_history_nodes'):
                if history_node_bank is not None:
                    # Frozen rollout reads the state bank directly, avoiding
                    # records*nodes*width gathered historical activations.
                    # The helper uses differentiable BMM when reverse metadata
                    # is absent in an ordinary gradient-enabled forward.
                    history_pools = indexed_history_pools(
                        history_node_bank, before_ids, allocation_tensor,
                        exposure_tensor, sizes)
                else:
                    # Variable-node graphs retain the exact sentinel/padding
                    # gather, followed by one two-channel weighted-pool BMM.
                    past_nodes = gather_nodes(before)
                    history_weights = torch.stack((allocation_tensor, exposure_tensor), dim=1)
                    history_pools = torch.bmm(history_weights, past_nodes) / sizes[:, None, None]
            treated, exposed = history_pools.unbind(dim=1)
            history_tokens = self.record_token(torch.cat((
                pooled[before_ids], treated, exposed,
                pooled[self._tensor(after, torch.long)],
                self._tensor(np.asarray(feedback, dtype=np.float32) if inference_only else feedback)), dim=-1))
        else:
            history_tokens = query.new_zeros((0, self.config.width))

        lengths = (incremental_pack['lengths'] if incremental_pack is not None else
                   np.array([len(history) for history in histories], dtype=np.int64))
        max_length = int(lengths.max()) + 1
        all_tokens = torch.cat((history_tokens, query, query.new_zeros((1, self.config.width))), dim=0)
        token_indices = np.full((batch, max_length), record_count + batch, dtype=np.int64)
        cursor = 0
        for row, length in enumerate(lengths):
            token_indices[row, :length] = np.arange(cursor, cursor + length)
            token_indices[row, length] = record_count + row
            cursor += length
        seq = all_tokens[self._tensor(token_indices, torch.long)]
        seq = seq + self._position(max_length, self.config.width)[None]
        lengths_tensor = self._tensor(lengths, torch.long)
        # Right padding leaves every valid token's original position unchanged.
        padding = torch.arange(max_length, device=self.device)[None] > lengths_tensor[:, None]
        causal = self._causal(max_length)
        with self._rollout_range('PFN/Transformer'):
            global_history = self._history_readout(seq, lengths_tensor, causal, padding)
        nodes = optimized_fuse_nodes(self.fusion, current_nodes, global_history)
        nodes = nodes * node_mask[:, :, None]

        # Pad each distinct catalog once, then select it for all batch members.
        # The common parallel rollout has one observation per task: static
        # catalogs are already stacked in the exact requested batch order.
        # Skip that identity gather; repeated catalogs retain the general path.
        identity_catalogs = len(static) == batch and slots == list(range(batch))
        slot_tensor = None if identity_catalogs else self._tensor(slots, torch.long)
        def catalog_tensor(name):
            bank = _stack_catalog(static, name, max_n, max_m)
            return bank if identity_catalogs else bank[slot_tensor]
        z = catalog_tensor("z")
        e = s = scalars = None
        if self.config.joint_head:
            e, s = catalog_tensor("e"), catalog_tensor("s")
            scalars = catalog_tensor('scalars')
        masks = np.zeros((batch, max_m), dtype=bool)
        for row, obs in enumerate(observations):
            masks[row, :len(obs["valid_action_mask"])] = obs["valid_action_mask"]
        logits = optimized_action_logits(
            self.node_logit, self.action_head, nodes, global_history,
            z, e, s, scalars, node_mask, n, self._tensor(masks, torch.bool),
            joint_head=self.config.joint_head, strategy='auto')
        values = self.value_head(torch.cat((pooled[current_ids], global_history), dim=-1)).squeeze(-1)
        return logits, values

    def prepare_rollout(self, rollout):
        """Pack each unique factual record once; histories use integer indices."""
        observations = rollout.observations[:len(rollout)]
        use_history = self.config.history and self.config.context_length > 0
        lengths = rollout.history_lengths[:len(rollout)] if use_history else np.zeros(len(rollout), dtype=np.int64)
        length = int(lengths.max())
        history_ids = rollout.history_indices[:len(rollout), :length]
        entries, graph_edges, slots, catalog_ids = [], [], [], {}
        def catalog_slot(obs):
            key, entry = self._static(obs)
            if key not in catalog_ids:
                catalog_ids[key] = len(entries)
                entries.append(entry)
                graph_edges.append(obs['edge_index'])
            return catalog_ids[key]
        for obs in observations:
            slots.append(catalog_slot(obs))
        max_n = max(e['n'] for e in entries)
        max_m = max(e['m'] for e in entries)
        states, globals_, state_slots, state_ids = [], [], [], {}
        def register(state, global_state, slot):
            key = (id(state), id(global_state), slot)
            if key not in state_ids:
                state_ids[key] = len(states)
                states.append(state)
                globals_.append(global_state)
                state_slots.append(slot)
            return state_ids[key]
        current = [register(obs['pfn_state'], obs['global_state'], slot)
                   for obs, slot in zip(observations, slots)]
        records = rollout.records[:rollout.record_count] if length else []
        before, after = np.zeros(len(records), dtype=np.int64), np.zeros(len(records), dtype=np.int64)
        allocation = np.zeros((len(records), max_n), dtype=np.float32)
        exposure = np.zeros_like(allocation)
        feedback = np.zeros((len(records), 4), dtype=np.float32)
        record_episodes = np.zeros(len(records), dtype=np.int64)
        for row, record in enumerate(records):
            slot = catalog_slot(rollout.record_observations[row])
            before[row] = register(record.state, record.global_state, slot)
            after[row] = register(record.next_state, record.next_global, slot)
            n = len(record.allocation)
            allocation[row, :n] = record.allocation
            exposure[row, :n] = record.exposure
            feedback[row, :3] = record.reward, record.done, record.time / 20
            record_episodes[row] = record.episode
        raw = np.zeros((len(states), max_n, states[0].shape[1]), dtype=np.float32)
        for row, state in enumerate(states):
            raw[row, :len(state)] = state
        masks = np.zeros((len(observations), max_m), dtype=bool)
        for row, obs in enumerate(observations):
            masks[row, :len(obs['valid_action_mask'])] = obs['valid_action_mask']
        bank = {name: _stack_catalog(entries, name, max_n, max_m)
                for name in ('z', 'e', 's')}
        bank['scalars'] = _stack_catalog(entries, 'scalars', max_n, max_m)
        dense = None
        if self._dense_graphs(max_n):
            dense = self._tensor(np.stack([
                e['dense_cpu'] if e['n'] == max_n
                else np.pad(e['dense_cpu'], ((0, max_n-e['n']), (0, max_n-e['n'])))
                for e in entries]))
        return dict(raw=self._tensor(raw), globals=self._tensor(np.stack(globals_)),
                    state_slots_cpu=np.asarray(state_slots), counts=self._tensor([len(s) for s in states]),
                    current_cpu=np.asarray(current), before_cpu=before, after_cpu=after,
                    history_ids_cpu=history_ids.copy(), lengths_cpu=lengths.copy(),
                    episodes_cpu=rollout.episodes[:len(rollout)].copy(),
                    allocations=self._tensor(allocation), exposures=self._tensor(exposure),
                    feedback=self._tensor(feedback), record_episodes=record_episodes,
                    lengths=self._tensor(lengths, torch.long),
                    times=self._tensor(np.column_stack((rollout.times[:len(rollout)] / 20,
                                                         np.zeros(len(rollout))))),
                    slots=self._tensor(slots, torch.long), masks=self._tensor(masks, torch.bool),
                    graph_edges_cpu=graph_edges, bank=bank, length=length, dense=dense,
                    position=self._position(length + 1, self.config.width),
                    causal=self._causal(length + 1))

    def prepare_minibatches(self, data, orders, minibatch):
        """Retain small CPU orders only; graph plans enter the device one batch at a time."""
        if minibatch <= 0:
            raise ValueError('minibatch must be positive')
        return [MinibatchEpoch(self, data, order, minibatch) for order in orders]

    def _prepare_minibatch(self, data, ids):
        lengths = data['lengths_cpu'][ids]
        length = int(lengths.max())
        batch = len(ids)
        valid = np.arange(length)[None] < lengths[:, None]
        history = data['history_ids_cpu'][ids, :length]
        # A record token depends on the current episode as well as the record.
        # Deduplication therefore uses both fields, never just the record id.
        pairs, token_inverse = unique_pairs_packed(
            history[valid], data['episodes_cpu'][ids], lengths,
            len(data['before_cpu']))
        records = pairs[:, 0]
        references = np.concatenate((data['current_cpu'][ids], data['before_cpu'][records],
                                     data['after_cpu'][records]))
        states, inverse = unique_states_visited(
            references, len(data['state_slots_cpu']))
        number = len(records)
        indices = np.full((batch, length + 1), number + batch, dtype=np.int64)
        indices[:, :length][valid] = token_inverse
        indices[np.arange(batch), lengths] = number + np.arange(batch)
        slots = data['state_slots_cpu'][states]
        arrays = dict(ids=ids, states=states, current=inverse[:batch], before=inverse[batch:batch+number],
                      after=inverse[batch+number:], records=records, token_indices=indices, state_slots=slots)
        integers = self._tensor(np.concatenate([array.reshape(-1) for array in arrays.values()]), torch.long)
        plan, cursor = {}, 0
        for name, array in arrays.items():
            plan[name] = integers[cursor:cursor + array.size].reshape(array.shape)
            cursor += array.size
        plan.update(episode_delta=self._tensor((data['record_episodes'][records] - pairs[:, 1]) / 10),
                    length=length, has_history=number > 0)
        if (number > 0 and not _musa_tf32_enabled(next(self.parameters()))
                and triton_usable(next(self.parameters()))):
            # Build the reverse occurrence lists from the existing CPU inverse.
            # Backward reduces per-state contributions without atomic scatters;
            # no accelerator data is read back to prepare this plan.
            plan['history_reverse'] = prepare_history_reverse(
                arrays['before'], len(states), self.device)
        if data['dense'] is None:
            # Normalization is performed on CPU, avoiding dynamic boolean indexing
            # and its device synchronization while constructing a graph plan.
            max_n = data['raw'].shape[1]
            edges = np.concatenate([data['graph_edges_cpu'][slot] + row * max_n
                                    for row, slot in enumerate(slots)], axis=1)
            adjacency = normalized_adjacency(torch.from_numpy(edges), len(states) * max_n)
            plan['adj_t'] = EdgeAdjacency(adjacency.row.to(self.device), adjacency.col.to(self.device),
                                          adjacency.value.to(self.device), adjacency.count)
        return plan

    def forward_prepared(self, data, plan, return_logits=False):
        """Recompute all GNN/Transformer features under the current parameters."""
        ids, state_ids = plan['ids'], plan['states']
        raw = data['raw'][state_ids]
        count, max_n, _ = raw.shape
        x = self.input(raw)
        if data['dense'] is not None:
            adjacency = data['dense'][plan['state_slots']]
            for conv, norm in zip(self.gnn, self.norms):
                aggregate = torch.bmm(adjacency, conv.lin(x))
                x = fused_bias_relu_residual_layer_norm(
                    aggregate, conv.bias, x, norm.weight, norm.bias, norm.eps)
        else:
            x = x.reshape(-1, self.config.width)
            for conv, norm in zip(self.gnn, self.norms):
                x = norm(x + F.relu(conv(x, plan['adj_t'])))
            x = x.reshape(count, max_n, self.config.width)
        counts = data['counts'][state_ids]
        node_masks = torch.arange(max_n, device=self.device)[None] < counts[:, None]
        x = x * node_masks[:, :, None]
        means = x.sum(1) / counts[:, None]
        pooled = means + self.global_input(data['globals'][state_ids])
        current = plan['current']
        current_nodes, n = x[current], counts[current]
        query = self.query_token(torch.cat((pooled[current], means[current], data['times'][ids]), dim=-1))
        length, batch = plan['length'], ids.shape[0]
        before, after = plan['before'], plan['after']
        if plan['has_history']:
            record_ids = plan['records']
            history_pools = indexed_history_pools(
                x, before, data['allocations'][record_ids],
                data['exposures'][record_ids], counts[before],
                reverse_plan=plan.get('history_reverse'))
            treated, exposed = history_pools.unbind(dim=1)
            feedback = data['feedback'][record_ids].clone()
            feedback[:, 3] = plan['episode_delta']
            tokens = self.record_token(torch.cat((pooled[before], treated, exposed,
                                                 pooled[after], feedback), dim=-1))
        else:
            tokens = query.new_zeros((0, self.config.width))
        lengths = data['lengths'][ids]
        positions = torch.arange(length + 1, device=self.device)[None]
        bank = torch.cat((tokens, query, query.new_zeros((1, self.config.width))))
        seq = bank[plan['token_indices']] + data['position'][None, :length + 1]
        history = self._history_readout(
            seq, lengths, data['causal'][:length + 1, :length + 1],
            positions > lengths[:, None])
        nodes = optimized_fuse_nodes(self.fusion, current_nodes, history)
        current_node_mask = node_masks[current]
        nodes = nodes * current_node_mask[:, :, None]
        slots = data['slots'][ids]
        z = data['bank']['z'][slots]
        e = s = scalars = None
        if self.config.joint_head:
            e, s = data['bank']['e'][slots], data['bank']['s'][slots]
            scalars = data['bank']['scalars'][slots]
        logits = optimized_action_logits(
            self.node_logit, self.action_head, nodes, history, z, e, s,
            scalars, current_node_mask, n, data['masks'][ids],
            joint_head=self.config.joint_head, strategy='auto')
        values = self.value_head(torch.cat((pooled[current], history), dim=-1)).squeeze(-1)
        if return_logits:
            # DDP must see every parameter even on an empty-history rank. These
            # exact-zero edges expose no observations and do not alter the loss.
            unused = ([self.record_token] if not plan['has_history'] else [])
            if not self.config.joint_head:
                unused.append(self.action_head)
            values = self._ddp_anchor(values, unused)
            return logits, values
        return Categorical(logits=logits, validate_args=False), values

    def _ddp_anchor(self, values, unused):
        for module in unused:
            for parameter in module.parameters():
                values = values + parameter.reshape(-1)[0] * 0.0
        return values

    def forward_batch(self, observations, histories, episodes, history_slots=None):
        logits, values = self._forward_many(
            observations, histories, episodes, history_slots=history_slots)
        return Categorical(logits=logits), values

    def forward(self, obs, history=(), episode=0):
        logits, values = self._forward_many([obs], [history], [episode])
        return Categorical(logits=logits[0]), values[0]

    @torch.no_grad()
    def act_batch(self, observations, histories, episodes, generator=None, history_slots=None):
        with self._rollout_range('PFN/forward_batch'):
            dist, values = self.forward_batch(
                observations, histories, episodes, history_slots=history_slots)
        with self._rollout_range('PFN/CPU_choices'):
            return self._cpu_choices(dist, values, generator)

    def _cpu_choices(self, dist, values, generator=None, deterministic=False):
        if generator is not None and generator.device.type != 'cpu':
            raise ValueError('Policy sampling requires a CPU generator')
        m = dist.probs.shape[-1]
        # One device-to-host boundary. Sampling and log-probability lookup stay
        # on the CPU; no sampled action is uploaded just to download it again.
        with self._rollout_range('PFN/GPU_to_CPU_copy_and_sync'):
            packed = torch.cat((dist.probs, dist.logits, values[:, None]), dim=-1).cpu()
        probabilities, log_probs = packed[:, :m], packed[:, m:2*m]
        actions = log_probs.argmax(-1) if deterministic else torch.multinomial(
            probabilities, 1, generator=generator).squeeze(-1)
        selected = log_probs.gather(1, actions[:, None]).squeeze(-1)
        return torch.stack((actions.to(values.dtype), selected, packed[:, -1]), dim=-1).tolist()

    @torch.no_grad()
    def act(self, obs, history=(), episode=0, deterministic=False, generator=None):
        dist, values = self.forward_batch([obs], [history], [episode])
        action_value, log_prob, state_value = self._cpu_choices(dist, values, generator, deterministic)[0]
        return int(action_value), log_prob, state_value

    def save(self, path, extra=None):
        torch.save({"model_config": asdict(self.config), "model": self.state_dict(),
                    "extra": extra or {}}, path)

    @classmethod
    def load(cls, path, device="cpu"):
        device = resolve_device(device)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(ModelConfig(**payload["model_config"])).to(device)
        model.load_state_dict(payload["model"])
        model.eval()
        return model, payload["extra"]

    def _history_readout(self, seq, lengths, causal, padding):
        """Evaluate the current query without unused final-layer token outputs.

        Every preceding layer still processes the complete causal sequence.
        All final-layer K/V positions remain in the autograd graph. Nothing is
        cached across forwards, parameter updates, episodes or window shifts.
        """
        if not len(self.transformer):
            return seq[torch.arange(seq.shape[0], device=seq.device), lengths]
        for layer in self.transformer[:-1]:
            seq = layer(seq, src_mask=causal, src_key_padding_mask=padding)
        last = self.transformer[-1]
        if seq.shape[1] == 1:
            # There is no unused row at L=1: keep the one fused QKV projection.
            return last(seq, src_mask=causal, src_key_padding_mask=padding)[:, 0]
        return last.forward_last(
            seq, lengths, src_mask=causal, src_key_padding_mask=padding)

class MinibatchEpoch:
    """Reiterable lazy plans; device memory is bounded by the current minibatch."""
    def __init__(self, model, data, order, minibatch):
        self.model, self.data = model, data
        self.order, self.minibatch = np.asarray(order), minibatch

    def __len__(self):
        return (len(self.order) + self.minibatch - 1) // self.minibatch

    def __iter__(self):
        for start in range(0, len(self.order), self.minibatch):
            yield self.model._prepare_minibatch(self.data, self.order[start:start + self.minibatch])
