"""Cue-equivariant tensor product implemented via segmented polynomials."""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_jax as cuex
import haiku as hk
import jax
import jax.numpy as jnp
from e3nn_jax import Irreps

from .utility import collapse_ir_mul_segments, ir_mul_to_mul_ir, mul_ir_to_ir_mul


class TensorProduct(hk.Module):
    """Channel-wise tensor product evaluated with cuequivariance-jax."""

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = False,
        internal_weights: bool = False,
        instructions=None,
        cueq_config=None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        if internal_weights and not shared_weights:
            raise ValueError(
                'TensorProduct requires shared_weights=True when internal_weights=True'
            )
        self.shared_weights = shared_weights
        self.internal_weights = internal_weights

        self.irreps_in1_o3 = Irreps(irreps_in1)
        self.irreps_in2_o3 = Irreps(irreps_in2)
        self.irreps_out_o3 = Irreps(irreps_out)

        self.irreps_in1_cue = cue.Irreps(cue.O3, irreps_in1)
        self.irreps_in2_cue = cue.Irreps(cue.O3, irreps_in2)
        self.irreps_out_cue = cue.Irreps(cue.O3, irreps_out)

        descriptor = cue.descriptors.channelwise_tensor_product(
            self.irreps_in1_cue, self.irreps_in2_cue, self.irreps_out_cue
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = descriptor.polynomial.operands[0].size
        self.descriptor_out_irreps_o3 = Irreps(str(descriptor.outputs[0].irreps))
        self.output_segment_shapes = tuple(descriptor.polynomial.operands[-1].segments)

    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        batch_size = x1.shape[0]

        if self.internal_weights:
            if weights is not None:
                raise ValueError(
                    'TensorProduct uses internal weights; weights argument must be None'
                )
            parameter = hk.get_parameter(
                'weight',
                shape=(1, self.weight_numel),
                dtype=x1.dtype,
                init=hk.initializers.RandomNormal(),
            )
            weight_tensor = parameter
        else:
            if weights is None:
                raise ValueError(
                    'TensorProduct requires explicit weights when internal_weights=False'
                )
            weight_tensor = jnp.asarray(weights, dtype=x1.dtype)

        if weight_tensor.ndim == 1:
            weight_tensor = weight_tensor[jnp.newaxis, :]
        elif weight_tensor.ndim != 2:
            raise ValueError(
                f'Weights must have rank 1 or 2, got rank {weight_tensor.ndim}'
            )

        if weight_tensor.shape[-1] != self.weight_numel:
            raise ValueError(
                f'Expected weights last dimension {self.weight_numel}, got {weight_tensor.shape[-1]}'
            )

        if self.shared_weights:
            if weight_tensor.shape[0] not in (1, batch_size):
                raise ValueError(
                    'Shared weights require leading dimension 1 or equal to the batch size'
                )
            if weight_tensor.shape[0] == 1 and batch_size != 1:
                weight_tensor = jnp.broadcast_to(
                    weight_tensor, (batch_size, self.weight_numel)
                )
        else:
            if weight_tensor.shape[0] != batch_size:
                raise ValueError(
                    'Unshared weights require leading dimension equal to the batch size'
                )

        x1_ir_mul = mul_ir_to_ir_mul(x1, self.irreps_in1_o3)
        x2_ir_mul = mul_ir_to_ir_mul(x2, self.irreps_in2_o3)

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
                    (*x1.shape[:-1], self.descriptor_out_irreps_o3.dim), x1.dtype
                )
            ],
            method='naive',
            math_dtype=x1.dtype,
        )
        out_ir_mul = collapse_ir_mul_segments(
            out_ir_mul,
            self.descriptor_out_irreps_o3,
            self.irreps_out_o3,
            self.output_segment_shapes,
        )
        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, self.irreps_out_o3)
        return out_mul_ir
