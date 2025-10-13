"""Framework-agnostic implementation of the cue-equivariant linear layer."""

from __future__ import annotations

from dataclasses import dataclass

import cuequivariance as cue
import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps, IrrepsArray  # type: ignore

from .._utility import ir_mul_to_mul_ir, mul_ir_to_ir_mul


@dataclass(frozen=True)
class LinearCore:
    """Shared logic for Flax and Haiku linear adapters."""

    irreps_in: Irreps
    irreps_out: Irreps
    shared_weights: bool
    internal_weights: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, 'irreps_in_o3', Irreps(self.irreps_in))
        object.__setattr__(self, 'irreps_out_o3', Irreps(self.irreps_out))
        object.__setattr__(
            self,
            'irreps_in_cue',
            cue.Irreps(cue.O3, self.irreps_in_o3),
        )
        object.__setattr__(
            self,
            'irreps_out_cue',
            cue.Irreps(cue.O3, self.irreps_out_o3),
        )

        descriptor = cue.descriptors.linear(
            self.irreps_in_cue,
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

    @property
    def weight_param_shape(self) -> tuple[int, int]:
        """Shape of the internal weight parameter."""

        return (1, self.weight_numel)

    def init_weight(self, rng: jax.Array) -> jnp.ndarray:
        """Initialise internal weights with Gaussian noise."""

        return jax.random.normal(rng, self.weight_param_shape)

    def _extract_input(
        self,
        x: jnp.ndarray | IrrepsArray,
    ) -> tuple[jnp.ndarray, cuex.RepArray, bool]:
        """Return the raw array, RepArray view, and metadata flag."""

        had_irreps = False
        if isinstance(x, IrrepsArray):
            if x.irreps != self.irreps_in_o3:
                raise ValueError(
                    f'Linear expects input irreps {self.irreps_in_o3}, got {x.irreps}'
                )
            array = jnp.asarray(x.array)
            had_irreps = True
        else:
            array = jnp.asarray(x)

        ir_mul = mul_ir_to_ir_mul(array, self.irreps_in_o3)
        rep = cuex.RepArray(
            self.irreps_in_cue,
            jnp.asarray(ir_mul),
            cue.ir_mul,
        )
        return array, rep, had_irreps

    def _normalise_internal_weight(
        self,
        weight: jnp.ndarray,
        *,
        dtype: jnp.dtype,
    ) -> jnp.ndarray:
        """Cast and reshape the internal weight parameter."""

        array = jnp.asarray(weight, dtype=dtype)
        if array.ndim == 1:
            array = array[jnp.newaxis, :]
        elif array.ndim != 2:
            raise ValueError(
                f'Internal weights must have rank 1 or 2, got rank {array.ndim}'
            )
        if array.shape[-1] != self.weight_numel:
            raise ValueError(
                f'Expected weights last dimension {self.weight_numel}, got {array.shape[-1]}'
            )
        return array

    def _normalise_external_weights(
        self,
        weights: jnp.ndarray,
        *,
        dtype: jnp.dtype,
        batch_size: int,
    ) -> jnp.ndarray:
        """Validate and reshape external weights into a 2D tensor."""

        array = jnp.asarray(weights, dtype=dtype)
        if array.ndim == 1:
            if array.shape[0] != self.weight_numel:
                raise ValueError(
                    f'Expected weights last dimension {self.weight_numel}, got {array.shape[-1]}'
                )
            array = array[jnp.newaxis, :]
        elif array.ndim == 2:
            if array.shape[-1] != self.weight_numel:
                raise ValueError(
                    f'Expected weights last dimension {self.weight_numel}, got {array.shape[-1]}'
                )
        else:
            raise ValueError('Weights must have rank 1 or 2 for Linear')

        leading = array.shape[0]
        if self.shared_weights:
            if leading not in (1, batch_size):
                raise ValueError(
                    'Shared weights require leading dimension 1 or equal to the batch size'
                )
            if leading == 1 and batch_size > 1:
                array = jnp.broadcast_to(array, (batch_size, self.weight_numel))
        else:
            if leading != batch_size:
                raise ValueError(
                    'Unshared weights require leading dimension equal to the batch size'
                )
        return array

    def _resolve_weight_operand(
        self,
        *,
        dtype: jnp.dtype,
        batch_size: int,
        internal_weight: jnp.ndarray | None,
        external_weights: jnp.ndarray | None,
    ) -> cuex.RepArray | jnp.ndarray:
        """Return the operand to use in the segmented polynomial call."""

        if self.internal_weights:
            if external_weights is not None:
                raise ValueError(
                    'Weights must be None when internal_weights=True in Linear'
                )
            if internal_weight is None:
                raise ValueError(
                    'Internal weights must be provided when internal_weights=True in Linear'
                )
            array = self._normalise_internal_weight(internal_weight, dtype=dtype)
            return cuex.RepArray(self.weight_irreps, array, cue.ir_mul)

        if external_weights is None:
            raise ValueError(
                'Weights must be provided when internal_weights=False in Linear'
            )
        array = self._normalise_external_weights(
            external_weights,
            dtype=dtype,
            batch_size=batch_size,
        )
        if self.shared_weights:
            return cuex.RepArray(self.weight_irreps, array, cue.ir_mul)
        return array

    def apply(
        self,
        x: jnp.ndarray | IrrepsArray,
        *,
        weights: jnp.ndarray | None,
        internal_weight: jnp.ndarray | None,
        math_dtype: jnp.dtype = jnp.float64,
    ) -> tuple[jnp.ndarray, bool]:
        """Evaluate the linear map and return ``mul_ir`` output."""

        array, x_rep, had_irreps = self._extract_input(x)
        dtype = array.dtype
        batch_size = array.shape[0]

        weight_operand = self._resolve_weight_operand(
            dtype=dtype,
            batch_size=batch_size,
            internal_weight=internal_weight,
            external_weights=weights,
        )

        if isinstance(weight_operand, cuex.RepArray):
            inputs = [weight_operand, x_rep]
            output_rep = cuex.equivariant_polynomial(
                self.descriptor,
                inputs,
                math_dtype=math_dtype,
                method='naive',
            )
            out_ir_mul = output_rep.array
        else:
            inputs = [jnp.asarray(weight_operand, dtype=dtype), x_rep.array]
            [out_ir_mul] = cuex.segmented_polynomial(
                self.descriptor.polynomial,
                inputs,
                [
                    jax.ShapeDtypeStruct(
                        (*array.shape[:-1], self.irreps_out_o3.dim),
                        dtype,
                    )
                ],
                method='naive',
                math_dtype=math_dtype,
            )

        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, self.irreps_out_o3)
        return out_mul_ir, had_irreps
