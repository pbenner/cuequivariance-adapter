"""Shared helpers for tensor product adapter tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import cuequivariance as cue
import cuequivariance_torch as cuet
import jax.numpy as jnp
import numpy as np
import torch
from e3nn import o3  # type: ignore
from mace.modules.irreps_tools import tp_out_irreps_with_instructions  # type: ignore
from mace.modules.wrapper_ops import CuEquivarianceConfig  # type: ignore

TensorApply = Callable[
    [
        o3.Irreps,
        o3.Irreps,
        o3.Irreps,
        int,
        bool,
        bool,
        Optional[list[tuple[int, int, int, str, bool, float]]],
    ],
    Callable[
        [jnp.ndarray, jnp.ndarray, Optional[np.ndarray | jnp.ndarray]],
        np.ndarray | jnp.ndarray,
    ],
]


@dataclass
class TensorProductComparisonResult:
    diff_e3nn_cue: float
    diff_e3nn_adapter: float
    diff_cue_adapter: float
    out_e3nn: torch.Tensor
    out_cue: torch.Tensor
    out_adapter: torch.Tensor
    x1: torch.Tensor
    x2: torch.Tensor
    weight_tensor: torch.Tensor
    shared_weights: bool
    internal_weights: bool

    @property
    def max_diff(self) -> float:
        return max(self.diff_e3nn_cue, self.diff_e3nn_adapter, self.diff_cue_adapter)


def run_tensor_product_comparison(
    build_apply: TensorApply,
    irreps1: str,
    irreps2: str,
    irreps_target: str,
    *,
    shared_weights: bool,
    internal_weights: bool,
    batch: int = 8,
) -> float:
    """Return the maximum deviation across e3nn, cuet, and adapter outputs."""

    if internal_weights and not shared_weights:
        raise ValueError('internal_weights=True requires shared_weights=True')

    irreps1_o3 = o3.Irreps(irreps1)
    irreps2_o3 = o3.Irreps(irreps2)
    target_o3, instructions = tp_out_irreps_with_instructions(
        irreps1_o3, irreps2_o3, o3.Irreps(irreps_target)
    )

    tp_e3nn = o3.TensorProduct(
        irreps1_o3,
        irreps2_o3,
        target_o3,
        instructions=instructions,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )

    cueq_config = CuEquivarianceConfig(enabled=True, optimize_channelwise=True)
    tp_cue = cuet.ChannelWiseTensorProduct(
        cue.Irreps(cueq_config.group, irreps1_o3),
        cue.Irreps(cueq_config.group, irreps2_o3),
        cue.Irreps(cueq_config.group, target_o3),
        layout=cueq_config.layout,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )

    adapter_apply = build_apply(
        irreps1_o3,
        irreps2_o3,
        target_o3,
        tp_e3nn.weight_numel,
        shared_weights,
        internal_weights,
        instructions,
    )

    torch.manual_seed(0)
    x1 = torch.randn(batch, irreps1_o3.dim)
    x2 = torch.randn(batch, irreps2_o3.dim)

    if internal_weights:
        base_weights = tp_e3nn.weight.detach().clone()
        weight_tensor = base_weights.view(1, -1)
        with torch.no_grad():
            if tp_cue.weight is None:
                raise RuntimeError('cuet ChannelWiseTensorProduct missing weights')
            tp_cue.weight.copy_(weight_tensor)
        weights_arg_e3nn = None
        weights_arg_cue = None
    elif shared_weights:
        weight_tensor = torch.randn(1, tp_e3nn.weight_numel)
        weights_arg_e3nn = weight_tensor.view(-1)
        weights_arg_cue = weight_tensor
    else:
        weight_tensor = torch.randn(batch, tp_e3nn.weight_numel)
        weights_arg_e3nn = weight_tensor
        weights_arg_cue = weight_tensor

    x1_jax = jnp.asarray(x1.detach().cpu().numpy())
    x2_jax = jnp.asarray(x2.detach().cpu().numpy())
    if internal_weights:
        weights_payload: Optional[np.ndarray | jnp.ndarray] = (
            weight_tensor.detach().cpu().numpy()
        )
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

    return TensorProductComparisonResult(
        diff_e3nn_cue=diff_cue,
        diff_e3nn_adapter=diff_adapter,
        diff_cue_adapter=diff_cross,
        out_e3nn=out_e3nn,
        out_cue=out_cue,
        out_adapter=out_adapter,
        x1=x1,
        x2=x2,
        weight_tensor=weight_tensor,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )
