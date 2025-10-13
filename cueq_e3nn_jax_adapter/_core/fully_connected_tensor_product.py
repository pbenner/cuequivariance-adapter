"""Shared implementation of the fully connected tensor product adapter."""

from __future__ import annotations

from dataclasses import dataclass

import cuequivariance as cue
import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore

from .._utility import ir_mul_to_mul_ir, mul_ir_to_ir_mul


@dataclass(frozen=True)
class FullyConnectedTensorProductCore:
    """Backend-agnostic tensor product evaluator."""

    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps
    shared_weights: bool
    internal_weights: bool

    def __post_init__(self) -> None:
        if self.internal_weights and not self.shared_weights:
            raise ValueError(
                'FullyConnectedTensorProduct requires shared_weights=True when internal_weights=True'
            )

        object.__setattr__(self, 'irreps_in1_o3', Irreps(self.irreps_in1))
        object.__setattr__(self, 'irreps_in2_o3', Irreps(self.irreps_in2))
        object.__setattr__(self, 'irreps_out_o3', Irreps(self.irreps_out))

        object.__setattr__(
            self,
            'irreps_in1_cue',
            cue.Irreps(cue.O3, self.irreps_in1_o3),
        )
        object.__setattr__(
            self,
            'irreps_in2_cue',
            cue.Irreps(cue.O3, self.irreps_in2_o3),
        )
        object.__setattr__(
            self,
            'irreps_out_cue',
            cue.Irreps(cue.O3, self.irreps_out_o3),
        )

        descriptor = cue.descriptors.fully_connected_tensor_product(
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
        object.__setattr__(self, 'descriptor_out_dim', descriptor_out_irreps.dim)

    @property
    def weight_param_shape(self) -> tuple[int, int]:
        """Shape of the learnable weight parameter."""

        return (1, self.weight_numel)

    def init_weight(self, rng: jax.Array) -> jnp.ndarray:
        """Initialise internal weights with Gaussian noise."""

        return jax.random.normal(rng, self.weight_param_shape)

    def _input_rep(
        self,
        array: jnp.ndarray,
        irreps_o3: Irreps,
        irreps_cue: cue.Irreps,
    ) -> cuex.RepArray:
        """Convert mul_ir array to cue RepArray."""

        data = mul_ir_to_ir_mul(array, irreps_o3)
        return cuex.RepArray(
            irreps_cue,
            jnp.asarray(data),
            cue.ir_mul,
        )

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
        """Validate and reshape external weights."""

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
            raise ValueError(
                'Weights must have rank 1 or 2 when internal weights are not used'
            )

        leading = array.shape[0]
        if self.shared_weights:
            if leading not in (1, batch_size):
                raise ValueError(
                    'Shared weights require leading dimension 1 or equal to the batch size'
                )
            if leading == 1 and batch_size != 1:
                array = jnp.broadcast_to(array, (batch_size, self.weight_numel))
        else:
            if leading != batch_size:
                raise ValueError(
                    'Unshared weights require leading dimension equal to the batch size'
                )

        return array

    def _resolve_weight_rep(
        self,
        *,
        dtype: jnp.dtype,
        batch_size: int,
        internal_weight: jnp.ndarray | None,
        external_weights: jnp.ndarray | None,
    ) -> cuex.RepArray:
        """Return the weight RepArray according to the sharing policy."""

        if self.internal_weights:
            if external_weights is not None:
                raise ValueError(
                    'Weights must be None when internal weights are used in FullyConnectedTensorProduct'
                )
            if internal_weight is None:
                raise ValueError(
                    'Internal weights must be provided when internal_weights=True in FullyConnectedTensorProduct'
                )
            array = self._normalise_internal_weight(internal_weight, dtype=dtype)
        else:
            if external_weights is None:
                raise ValueError(
                    'Weights must be provided when internal weights are not used'
                )
            array = self._normalise_external_weights(
                external_weights,
                dtype=dtype,
                batch_size=batch_size,
            )
        return cuex.RepArray(self.weight_irreps, array, cue.ir_mul)

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

        irreps_in1 = Irreps(self.irreps_in1_o3)
        irreps_in2 = Irreps(self.irreps_in2_o3)
        irreps_out = Irreps(self.irreps_out_o3)

        x1_rep = self._input_rep(x1, irreps_in1, self.irreps_in1_cue)
        x2_rep = self._input_rep(x2, irreps_in2, self.irreps_in2_cue)
        weight_rep = self._resolve_weight_rep(
            dtype=dtype,
            batch_size=batch_size,
            internal_weight=internal_weight,
            external_weights=weights,
        )

        [out_ir_mul] = cuex.segmented_polynomial(
            self.descriptor.polynomial,
            [weight_rep.array, x1_rep.array, x2_rep.array],
            [
                jax.ShapeDtypeStruct(
                    (*x1.shape[:-1], self.descriptor_out_dim),
                    dtype,
                )
            ],
            method='naive',
            math_dtype=math_dtype,
        )

        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, irreps_out)
        return out_mul_ir
