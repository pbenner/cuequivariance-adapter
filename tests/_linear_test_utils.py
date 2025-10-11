"""Shared helpers for linear adapter tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import jax.numpy as jnp
import numpy as np
import torch
from e3nn import o3
from mace.modules.wrapper_ops import Linear as LinearWrapper

LinearApply = Callable[
    [o3.Irreps, o3.Irreps, int, bool, bool],
    Callable[[jnp.ndarray, jnp.ndarray | np.ndarray | None], np.ndarray | jnp.ndarray],
]


@dataclass
class LinearComparisonResult:
    """Artifacts produced when comparing an adapter against the reference."""

    max_diff: float
    reference_output: torch.Tensor
    adapter_output: torch.Tensor
    inputs: torch.Tensor
    linear_reference: LinearWrapper
    base_weights: torch.Tensor | None
    shared_weights: bool
    internal_weights: bool


def run_linear_comparison(
    build_apply: LinearApply,
    irreps_in: str,
    irreps_out: str,
    *,
    shared_weights: bool,
    internal_weights: bool,
    batch: int = 4,
) -> LinearComparisonResult:
    """Return comparison data for a single linear adapter configuration."""

    if internal_weights and not shared_weights:
        raise ValueError('internal_weights=True requires shared_weights=True')

    irreps_in_o3 = o3.Irreps(irreps_in)
    irreps_out_o3 = o3.Irreps(irreps_out)

    linear_ref = LinearWrapper(
        irreps_in_o3,
        irreps_out_o3,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        cueq_config=None,
    )

    weight_numel = linear_ref.weight_numel
    adapter_apply = build_apply(
        irreps_in_o3,
        irreps_out_o3,
        weight_numel,
        shared_weights,
        internal_weights,
    )

    torch.manual_seed(0)
    x = torch.randn(batch, irreps_in_o3.dim)

    if internal_weights:
        base_weights = linear_ref.weight.detach().clone()
        weights_arg_ref = None
        weights_for_apply = base_weights.detach().cpu().numpy()
    elif shared_weights:
        base_weights = torch.randn(1, weight_numel)
        weights_arg_ref = base_weights.view(-1)
        weights_for_apply = jnp.asarray(base_weights.detach().cpu().numpy())
    else:
        base_weights = torch.randn(batch, weight_numel)
        weights_arg_ref = base_weights
        weights_for_apply = jnp.asarray(base_weights.detach().cpu().numpy())

    x_jax = jnp.asarray(x.detach().cpu().numpy())

    if weights_arg_ref is None:
        out_ref = linear_ref(x)
    else:
        out_ref = linear_ref(x, weights_arg_ref)

    out_adapter = adapter_apply(
        x_jax,
        weights_for_apply if weights_for_apply is not None else None,
    )
    out_adapter = torch.from_numpy(np.array(out_adapter, copy=True))

    max_diff = float((out_ref - out_adapter).abs().max().item())

    return LinearComparisonResult(
        max_diff=max_diff,
        reference_output=out_ref,
        adapter_output=out_adapter,
        inputs=x,
        linear_reference=linear_ref,
        base_weights=base_weights,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )
