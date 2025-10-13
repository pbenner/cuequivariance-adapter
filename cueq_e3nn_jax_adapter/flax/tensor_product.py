"""Flax implementation of the channel-wise tensor product."""

from __future__ import annotations

import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore
from flax import linen as nn

from .._core import TensorProductCore


class TensorProduct(nn.Module):
    """Flax wrapper around the shared tensor product core."""

    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps
    shared_weights: bool = False
    internal_weights: bool = False
    instructions: list[tuple[int, int, int, str, bool, float]] | None = None

    def setup(self) -> None:
        self.core = TensorProductCore(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            shared_weights=self.shared_weights,
            internal_weights=self.internal_weights,
            instructions=self.instructions,
        )

    @nn.compact
    def __call__(
        self,
        x1: jnp.ndarray,
        x2: jnp.ndarray,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        internal_weight: jnp.ndarray | None = None
        if self.core.internal_weights:
            internal_weight = self.param(
                'weight',
                self.core.init_weight,
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
