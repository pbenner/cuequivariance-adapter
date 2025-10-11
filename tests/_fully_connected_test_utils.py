"""Shared helpers for fully connected tensor product adapter tests."""

from __future__ import annotations

from typing import Callable

import cuequivariance as cue
import cuequivariance_torch as cuet
import jax.numpy as jnp
import numpy as np
import torch
from e3nn import o3

FullyConnectedApply = Callable[
    [o3.Irreps, o3.Irreps, o3.Irreps, int, bool, bool], Callable[[jnp.ndarray, jnp.ndarray, np.ndarray | None], np.ndarray | jnp.ndarray]
]


def run_fully_connected_comparison(
    build_apply: callable,
    irreps1: str,
    irreps2: str,
    irreps_out: str,
    *,
    shared_weights: bool,
    internal_weights: bool,
    batch: int = 4,
) -> float:
    """Return the maximum deviation across the three implementations."""

    if internal_weights and not shared_weights:
        raise ValueError('internal_weights=True requires shared_weights=True')

    irreps1_o3 = o3.Irreps(irreps1)
    irreps2_o3 = o3.Irreps(irreps2)
    irreps_out_o3 = o3.Irreps(irreps_out)

    tp_e3nn = o3.FullyConnectedTensorProduct(
        irreps1_o3,
        irreps2_o3,
        irreps_out_o3,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )

    tp_cue = cuet.FullyConnectedTensorProduct(
        cue.Irreps(cue.O3, irreps1_o3),
        cue.Irreps(cue.O3, irreps2_o3),
        cue.Irreps(cue.O3, irreps_out_o3),
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )

    adapter_apply = build_apply(
        irreps1_o3,
        irreps2_o3,
        irreps_out_o3,
        tp_e3nn.weight_numel,
        shared_weights,
        internal_weights,
    )

    torch.manual_seed(0)
    x1 = torch.randn(batch, irreps1_o3.dim)
    x2 = torch.randn(batch, irreps2_o3.dim)

    if internal_weights:
        base_weights = tp_e3nn.weight.detach().clone()
        weight_tensor = base_weights.view(1, -1)
        with torch.no_grad():
            if tp_cue.weight is None:
                raise RuntimeError('cue FullyConnectedTensorProduct missing weights')
            tp_cue.weight.copy_(weight_tensor)
        weights_arg_e3nn = None
        weights_arg_cue = None
    elif shared_weights:
        base_weights = torch.randn(1, tp_e3nn.weight_numel)
        weights_arg_e3nn = base_weights.view(-1)
        weights_arg_cue = base_weights
        weight_tensor = base_weights
    else:
        base_weights = torch.randn(batch, tp_e3nn.weight_numel)
        weights_arg_e3nn = base_weights
        weights_arg_cue = base_weights
        weight_tensor = base_weights

    x1_jax = jnp.asarray(x1.detach().cpu().numpy())
    x2_jax = jnp.asarray(x2.detach().cpu().numpy())
    if internal_weights:
        weights_payload = weight_tensor.detach().cpu().numpy()
    else:
        weights_payload = jnp.asarray(weight_tensor.detach().cpu().numpy())

    if weights_arg_e3nn is None:
        out_e3nn = tp_e3nn(x1, x2)
    else:
        out_e3nn = tp_e3nn(x1, x2, weights_arg_e3nn)

    if weights_arg_cue is None:
        out_cue = tp_cue(x1, x2)
    else:
        out_cue = tp_cue(x1, x2, weights_arg_cue)

    out_adapter = adapter_apply(x1_jax, x2_jax, weights_payload)
    out_adapter = torch.from_numpy(np.array(out_adapter, copy=True))

    diff_cue = (out_e3nn - out_cue).abs().max().item()
    diff_adapter = (out_e3nn - out_adapter).abs().max().item()
    diff_cross = (out_cue - out_adapter).abs().max().item()

    return max(diff_cue, diff_adapter, diff_cross)
