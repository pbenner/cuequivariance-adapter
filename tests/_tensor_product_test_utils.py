"""Shared helpers for tensor product adapter tests."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Optional

import cuequivariance as cue
import cuequivariance_torch as cuet
import jax.numpy as jnp
import numpy as np
import torch
import e3nn  # type: ignore
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


def _parse_version(version: str) -> tuple[int, int, int]:
    """Return a tuple representation of the semantic version string."""

    parts: list[int] = []
    for chunk in version.split('.'):
        digits = ''
        for char in chunk:
            if char.isdigit():
                digits += char
            else:
                break
        parts.append(int(digits or 0))
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])


_E3NN_NEEDS_SIGN_FIX = _parse_version(e3nn.__version__) < (0, 5, 0)


def _normalise_instruction_tuple(
    instructions: list[tuple[int, int, int, str, bool, float]] | None,
) -> tuple[tuple[int, int, int, str, bool, float], ...]:
    """Convert instructions to a canonical, hashable representation."""

    if not instructions:
        return ()
    normalised: list[tuple[int, int, int, str, bool, float]] = []
    for inst in instructions:
        if len(inst) == 6:
            i1, i2, i_out, mode, train, weight = inst
        elif len(inst) == 5:
            i1, i2, i_out, mode, train = inst  # type: ignore[misc]
            weight = 1.0
        else:
            raise ValueError(f'invalid instruction length {len(inst)} for {inst!r}')
        normalised.append(
            (int(i1), int(i2), int(i_out), str(mode), bool(train), float(weight))
        )
    return tuple(normalised)


def _instructions_from_key(
    key: tuple[tuple[int, int, int, str, bool, float], ...]
) -> list[tuple[int, int, int, str, bool] | tuple[int, int, int, str, bool, float]]:
    """Restore instruction tuples suitable for e3nn/cue constructors."""

    restored: list[
        tuple[int, int, int, str, bool] | tuple[int, int, int, str, bool, float]
    ] = []
    for i1, i2, i_out, mode, train, weight in key:
        if weight == 1.0:
            restored.append((i1, i2, i_out, mode, train))
        else:
            restored.append((i1, i2, i_out, mode, train, weight))
    return restored


@lru_cache(maxsize=None)
def _sign_corrections(
    irreps1: str,
    irreps2: str,
    target: str,
    instructions_key: tuple[tuple[int, int, int, str, bool, float], ...],
    weight_numel: int,
) -> tuple[float, ...]:
    """Return per-weight sign corrections for legacy e3nn tensor products."""

    if not _E3NN_NEEDS_SIGN_FIX:
        return tuple([1.0] * weight_numel)

    irreps1_o3 = o3.Irreps(irreps1)
    irreps2_o3 = o3.Irreps(irreps2)
    target_o3 = o3.Irreps(target)
    instructions = _instructions_from_key(instructions_key)

    tp_e3nn = o3.TensorProduct(
        irreps1_o3,
        irreps2_o3,
        target_o3,
        instructions=instructions if instructions else None,
        shared_weights=True,
        internal_weights=True,
    )

    cueq_config = CuEquivarianceConfig(enabled=True, optimize_channelwise=True)
    tp_cue = cuet.ChannelWiseTensorProduct(
        cue.Irreps(cueq_config.group, irreps1_o3),
        cue.Irreps(cueq_config.group, irreps2_o3),
        cue.Irreps(cueq_config.group, target_o3),
        layout=cueq_config.layout,
        shared_weights=True,
        internal_weights=True,
    )
    if tp_cue.weight is None:
        raise RuntimeError('cue ChannelWiseTensorProduct initialised without weights')

    x1 = torch.randn(2, irreps1_o3.dim, dtype=torch.float64)
    x2 = torch.randn(2, irreps2_o3.dim, dtype=torch.float64)

    signs = torch.ones(weight_numel, dtype=torch.float64)
    for idx in range(weight_numel):
        weight = torch.zeros_like(tp_e3nn.weight, dtype=torch.float64)
        weight.view(-1)[idx] = 1.0

        with torch.no_grad():
            tp_e3nn.weight.copy_(weight)
            tp_cue.weight.copy_(weight.view_as(tp_cue.weight))

        out_e3nn = tp_e3nn(x1, x2)
        out_cue = tp_cue(x1, x2)

        mask = out_cue.abs() > 1e-8
        if mask.any():
            ratio = (out_e3nn[mask] / out_cue[mask]).mean().item()
            signs[idx] = 1.0 if ratio >= 0 else -1.0

    return tuple(signs.tolist())


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

    instructions_key = _normalise_instruction_tuple(
        [tuple(inst) for inst in instructions] if instructions else None
    )
    sign_vector = torch.tensor(
        _sign_corrections(
            str(irreps1_o3),
            str(irreps2_o3),
            str(target_o3),
            instructions_key,
            tp_e3nn.weight_numel,
        ),
        dtype=torch.float32,
    )

    if internal_weights:
        base_weights = tp_e3nn.weight.detach().clone()
        weight_tensor = base_weights.view(1, -1)
        local_signs = sign_vector.to(weight_tensor.device, dtype=weight_tensor.dtype)
        e3nn_weight_tensor = (weight_tensor * local_signs.view(1, -1)).view_as(
            tp_e3nn.weight
        )
        with torch.no_grad():
            tp_e3nn.weight.copy_(e3nn_weight_tensor)
            if tp_cue.weight is None:
                raise RuntimeError('cuet ChannelWiseTensorProduct missing weights')
            tp_cue.weight.copy_(weight_tensor.view_as(tp_cue.weight))
        weights_arg_e3nn = None
        weights_arg_cue = None
    elif shared_weights:
        weight_tensor = torch.randn(1, tp_e3nn.weight_numel)
        local_signs = sign_vector.to(weight_tensor.device, dtype=weight_tensor.dtype)
        weights_arg_e3nn = (weight_tensor * local_signs.view(1, -1)).view(-1)
        weights_arg_cue = weight_tensor
    else:
        weight_tensor = torch.randn(batch, tp_e3nn.weight_numel)
        local_signs = sign_vector.to(weight_tensor.device, dtype=weight_tensor.dtype)
        weights_arg_e3nn = weight_tensor * local_signs.view(1, -1)
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
