"""Shared implementation of the channel-wise tensor product adapter."""

from __future__ import annotations

from dataclasses import dataclass

import cuequivariance as cue
import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore

from .._utility import (
    collapse_ir_mul_segments,
    ir_mul_to_mul_ir,
    mul_ir_to_ir_mul,
)


def _expected_channelwise_instructions(
    irreps_in1: Irreps,
    irreps_in2: Irreps,
    target_irreps: Irreps,
) -> tuple[Irreps, list[tuple[int, int, int, str, bool, float]]]:
    """Return the canonical instructions for channel-wise tensor products."""

    collected: list[tuple[int, Irreps]] = []
    instructions: list[tuple[int, int, int, str, bool, float]] = []
    for i_in1, (mul_in1, ir_in1) in enumerate(irreps_in1):
        for i_in2, (_, ir_in2) in enumerate(irreps_in2):
            for ir_out in ir_in1 * ir_in2:
                if ir_out in target_irreps:
                    idx = len(collected)
                    collected.append((mul_in1, ir_out))
                    instructions.append((i_in1, i_in2, idx, 'uvu', True, 1.0))

    irreps_out = Irreps(collected)
    irreps_out_sorted, perm, _ = irreps_out.sort()
    remapped_instructions = [
        (i_in1, i_in2, perm[i_out], mode, has_weight, path_weight)
        for i_in1, i_in2, i_out, mode, has_weight, path_weight in instructions
    ]
    remapped_instructions.sort(key=lambda item: item[2])
    return irreps_out_sorted, remapped_instructions


def _normalise_instruction(
    inst: tuple[int, int, int, str, bool] | tuple[int, int, int, str, bool, float],
) -> tuple[int, int, int, str, bool, float]:
    """Ensure instructions use the six-field representation."""

    if len(inst) == 5:
        i1, i2, i_out, mode, has_weight = inst
        path_weight = 1.0
    elif len(inst) == 6:
        i1, i2, i_out, mode, has_weight, path_weight = inst
    else:
        raise ValueError(
            'TensorProduct instructions must have length 5 or 6, '
            f'got length {len(inst)}'
        )
    return (
        int(i1),
        int(i2),
        int(i_out),
        str(mode),
        bool(has_weight),
        float(path_weight),
    )


@dataclass(frozen=True)
class TensorProductCore:
    """Common tensor product logic shared by Flax and Haiku wrappers."""

    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps
    shared_weights: bool
    internal_weights: bool
    instructions: list[tuple[int, int, int, str, bool, float]] | None = None

    def __post_init__(self) -> None:
        if self.internal_weights and not self.shared_weights:
            raise ValueError(
                'TensorProduct requires shared_weights=True when internal_weights=True'
            )

        irreps_in1_o3 = Irreps(self.irreps_in1)
        irreps_in2_o3 = Irreps(self.irreps_in2)
        irreps_out_o3 = Irreps(self.irreps_out)
        object.__setattr__(self, 'irreps_in1_o3', irreps_in1_o3)
        object.__setattr__(self, 'irreps_in2_o3', irreps_in2_o3)
        object.__setattr__(self, 'irreps_out_o3', irreps_out_o3)

        object.__setattr__(self, 'irreps_in1_cue', cue.Irreps(cue.O3, irreps_in1_o3))
        object.__setattr__(self, 'irreps_in2_cue', cue.Irreps(cue.O3, irreps_in2_o3))
        object.__setattr__(self, 'irreps_out_cue', cue.Irreps(cue.O3, irreps_out_o3))

        descriptor = cue.descriptors.channelwise_tensor_product(
            self.irreps_in1_cue,
            self.irreps_in2_cue,
            self.irreps_out_cue,
        )
        object.__setattr__(self, 'descriptor', descriptor)
        weight_irreps = descriptor.inputs[0].irreps
        object.__setattr__(self, 'weight_irreps', weight_irreps)
        object.__setattr__(
            self,
            'weight_numel',
            descriptor.polynomial.operands[0].size,
        )

        descriptor_out_irreps = Irreps(str(descriptor.outputs[0].irreps))
        object.__setattr__(self, 'descriptor_out_irreps', descriptor_out_irreps)
        object.__setattr__(
            self,
            'output_segment_shapes',
            tuple(descriptor.polynomial.operands[-1].segments),
        )

        expected_irreps, expected_instructions = _expected_channelwise_instructions(
            irreps_in1_o3,
            irreps_in2_o3,
            irreps_out_o3,
        )
        if expected_irreps != irreps_out_o3:
            raise ValueError(
                'TensorProduct irreps_out is incompatible with channel-wise descriptor'
            )

        if self.instructions is not None:
            normalised = [_normalise_instruction(inst) for inst in self.instructions]
            if normalised != expected_instructions:
                raise ValueError(
                    'TensorProduct only supports channel-wise "uvu" instructions '
                    'matching those returned by e3nn; received '
                    f'{self.instructions!r}'
                )

    @property
    def weight_param_shape(self) -> tuple[int, int]:
        """Shape of the learnable weight parameter."""

        return (1, self.weight_numel)

    def init_weight(self, rng: jax.Array) -> jnp.ndarray:
        """Initialise internal weights with Gaussian noise."""

        return jax.random.normal(rng, self.weight_param_shape)

    def _resolve_weight_tensor(
        self,
        *,
        dtype: jnp.dtype,
        batch_size: int,
        internal_weight: jnp.ndarray | None,
        external_weights: jnp.ndarray | None,
    ) -> jnp.ndarray:
        """Return a validated weight tensor of shape (batch, weight_numel)."""

        if self.internal_weights:
            if external_weights is not None:
                raise ValueError(
                    'TensorProduct uses internal weights; weights argument must be None'
                )
            if internal_weight is None:
                raise ValueError(
                    'Internal weights must be provided when internal_weights=True in TensorProduct'
                )
            tensor = jnp.asarray(internal_weight, dtype=dtype)
        else:
            if external_weights is None:
                raise ValueError(
                    'TensorProduct requires explicit weights when internal_weights=False'
                )
            tensor = jnp.asarray(external_weights, dtype=dtype)

        if tensor.ndim == 1:
            tensor = tensor[jnp.newaxis, :]
        elif tensor.ndim != 2:
            raise ValueError(f'Weights must have rank 1 or 2, got rank {tensor.ndim}')

        if tensor.shape[-1] != self.weight_numel:
            raise ValueError(
                f'Expected weights last dimension {self.weight_numel}, got {tensor.shape[-1]}'
            )

        leading = tensor.shape[0]
        if self.shared_weights:
            if leading not in (1, batch_size):
                raise ValueError(
                    'Shared weights require leading dimension 1 or equal to the batch size'
                )
            if leading == 1 and batch_size != 1:
                tensor = jnp.broadcast_to(tensor, (batch_size, self.weight_numel))
        else:
            if leading != batch_size:
                raise ValueError(
                    'Unshared weights require leading dimension equal to the batch size'
                )

        return tensor

    def apply(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        *,
        weights: jnp.ndarray | None,
        internal_weight: jnp.ndarray | None,
        math_dtype: jnp.dtype,
    ) -> jnp.ndarray:
        """Evaluate the tensor product and return mul_ir output."""

        batch_size = x1.shape[0]
        dtype = x1.dtype

        x1_ir_mul = mul_ir_to_ir_mul(x1, self.irreps_in1_o3)
        x2_ir_mul = mul_ir_to_ir_mul(x2, self.irreps_in2_o3)

        weight_tensor = self._resolve_weight_tensor(
            dtype=dtype,
            batch_size=batch_size,
            internal_weight=internal_weight,
            external_weights=weights,
        )

        x1_rep = cuex.RepArray(
            self.irreps_in1_cue,
            jnp.asarray(x1_ir_mul),
            cue.ir_mul,
        )
        x2_rep = cuex.RepArray(
            self.irreps_in2_cue,
            jnp.asarray(x2_ir_mul),
            cue.ir_mul,
        )
        weight_rep = cuex.RepArray(
            self.weight_irreps,
            weight_tensor,
            cue.ir_mul,
        )

        [out_ir_mul] = cuex.segmented_polynomial(
            self.descriptor.polynomial,
            [weight_rep.array, x1_rep.array, x2_rep.array],
            [
                jax.ShapeDtypeStruct(
                    (*x1.shape[:-1], self.descriptor_out_irreps.dim),
                    dtype,
                )
            ],
            method='naive',
            math_dtype=math_dtype,
        )
        out_ir_mul = collapse_ir_mul_segments(
            out_ir_mul,
            self.descriptor_out_irreps,
            self.irreps_out_o3,
            self.output_segment_shapes,
        )
        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, self.irreps_out_o3)
        return out_mul_ir
