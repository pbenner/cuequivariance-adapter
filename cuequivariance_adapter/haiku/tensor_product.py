"""Cue-equivariant tensor product implemented via segmented polynomials."""

from __future__ import annotations

import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore
import haiku as hk

from .._core import TensorProductCore


class TensorProduct(hk.Module):
    """Haiku wrapper around the shared tensor product core."""

    def __init__(
        self,
        irreps_in1: Irreps,
        irreps_in2: Irreps,
        irreps_out: Irreps,
        shared_weights: bool = False,
        internal_weights: bool = False,
        instructions=None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.core = TensorProductCore(
            irreps_in1,
            irreps_in2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
            instructions=instructions,
        )

    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        internal_weight = None
        if self.core.internal_weights:
            internal_weight = hk.get_parameter(
                'weight',
                shape=self.core.weight_param_shape,
                dtype=x1.dtype,
                init=hk.initializers.RandomNormal(),
            )

        return self.core.apply(
            x1,
            x2,
            weights=weights,
            internal_weight=internal_weight,
            math_dtype=x1.dtype,
        )

    @property
    def weight_numel(self) -> int:
        return self.core.weight_numel
