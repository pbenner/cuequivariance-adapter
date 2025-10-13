"""Flax implementation of the symmetric contraction adapter."""

from __future__ import annotations

import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore
from flax import linen as nn

from .._core import SymmetricContractionCore


class SymmetricContraction(nn.Module):
    """Flax wrapper around the shared symmetric contraction core."""

    irreps_in: Irreps
    irreps_out: Irreps
    correlation: int
    num_elements: int
    use_reduced_cg: bool = True

    def setup(self) -> None:
        self.core = SymmetricContractionCore(
            self.irreps_in,
            self.irreps_out,
            correlation=self.correlation,
            num_elements=self.num_elements,
            use_reduced_cg=self.use_reduced_cg,
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        input_layout: str = 'mul_ir',
    ) -> jnp.ndarray:
        features = self.core.ensure_mul_ir_layout(x, layout=input_layout)
        basis_weights = self.param(
            'weight',
            self.core.init_weight,
        )
        return self.core.apply(
            features,
            indices,
            basis_weights=basis_weights,
            math_dtype=x.dtype,
        )

    @property
    def weight_param_shape(self) -> tuple[int, int, int]:
        return self.core.weight_param_shape
