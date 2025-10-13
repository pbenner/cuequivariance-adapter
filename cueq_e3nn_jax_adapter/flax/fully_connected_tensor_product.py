"""Flax implementation of the fully connected tensor product."""

from __future__ import annotations

import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore
from flax import linen as nn

from .._core import FullyConnectedTensorProductCore


class FullyConnectedTensorProduct(nn.Module):
    """Flax wrapper around the shared fully connected tensor product core."""

    irreps_in1: Irreps
    irreps_in2: Irreps
    irreps_out: Irreps
    shared_weights: bool = True
    internal_weights: bool = True

    def setup(self) -> None:
        self.core = FullyConnectedTensorProductCore(
            self.irreps_in1,
            self.irreps_in2,
            self.irreps_out,
            shared_weights=self.shared_weights,
            internal_weights=self.internal_weights,
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
