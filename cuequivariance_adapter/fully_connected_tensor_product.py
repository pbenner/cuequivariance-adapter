"""Cue-equivariant tensor product implemented via segmented polynomials."""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_jax as cuex
import haiku as hk
import jax.numpy as jnp
from e3nn_jax import Irreps

from .utility import ir_mul_to_mul_ir, mul_ir_to_ir_mul


class FullyConnectedTensorProduct(hk.Module):
    """FullyConnectedTensorProduct evaluated with cuequivariance-jax."""

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = True,
        internal_weights: bool = True,
        initial_weight: jnp.ndarray | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        if internal_weights and not shared_weights:
            raise ValueError(
                'FullyConnectedTensorProduct requires shared_weights=True when internal_weights=True'
            )

        self.internal_weights = internal_weights
        self.shared_weights = shared_weights

        self.irreps_in1_o3 = Irreps(irreps_in1)
        self.irreps_in2_o3 = Irreps(irreps_in2)
        self.irreps_out_o3 = Irreps(irreps_out)

        self.irreps_in1_cue = cue.Irreps(cue.O3, irreps_in1)
        self.irreps_in2_cue = cue.Irreps(cue.O3, irreps_in2)
        self.irreps_out_cue = cue.Irreps(cue.O3, irreps_out)

        descriptor = cue.descriptors.fully_connected_tensor_product(
            self.irreps_in1_cue,
            self.irreps_in2_cue,
            self.irreps_out_cue,
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = descriptor.polynomial.operands[0].size

        self.internal_weight_rep = None
        if self.internal_weights:
            weights = hk.get_parameter(
                'weight', (1, self.weight_numel), init=hk.initializers.RandomNormal()
            )
            self.internal_weight_rep = cuex.RepArray(
                self.weight_irreps,
                weights,
                cue.ir_mul,
            )

    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        x1_ir_mul = mul_ir_to_ir_mul(x1, self.irreps_in1_o3)
        x2_ir_mul = mul_ir_to_ir_mul(x2, self.irreps_in2_o3)

        x1_rep = cuex.RepArray(
            self.irreps_in1_cue,
            x1_ir_mul,
            cue.ir_mul,
        )
        x2_rep = cuex.RepArray(
            self.irreps_in2_cue,
            x2_ir_mul,
            cue.ir_mul,
        )

        if self.internal_weights:
            if weights is not None:
                raise ValueError(
                    'Weights must be None when internal weights are used in FullyConnectedTensorProduct'
                )
            weight_rep = self.internal_weight_rep
        else:
            if weights is None:
                raise ValueError(
                    'Weights must be provided when internal weights are not used'
                )
            if weights.shape[-1] != self.weight_numel:
                raise ValueError(
                    f'Expected weights last dimension {self.weight_numel}, got {weights.shape[-1]}'
                )
            weight_rep = cuex.RepArray(
                self.weight_irreps,
                weights,
                cue.ir_mul,
            )

        output_rep = cuex.equivariant_polynomial(
            self.descriptor,
            [weight_rep, x1_rep, x2_rep],
            method='naive',
        )

        out_ir_mul = output_rep.array
        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, self.irreps_out_o3)
        return out_mul_ir
