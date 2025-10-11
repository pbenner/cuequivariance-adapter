"""Cue-equivariant symmetric contraction implemented with segmented polynomials."""

from __future__ import annotations

import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore

import haiku as hk

from .._core import SymmetricContractionCore


class SymmetricContraction(hk.Module):
    """Haiku wrapper around the shared symmetric contraction core."""

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        correlation: int,
        num_elements: int,
        use_reduced_cg: bool = True,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.core = SymmetricContractionCore(
            irreps_in,
            irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        indices: jnp.ndarray,
    ) -> jnp.ndarray:
        basis_weights = hk.get_parameter(
            'weight',
            shape=self.core.weight_param_shape,
            dtype=x.dtype,
            init=hk.initializers.RandomNormal(),
        )
        return self.core.apply(
            x,
            indices,
            basis_weights=basis_weights,
            math_dtype=x.dtype,
        )

    @property
    def weight_param_shape(self) -> tuple[int, int, int]:
        return self.core.weight_param_shape
