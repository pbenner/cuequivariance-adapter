"""Flax implementation of the cue-equivariant linear layer."""

from __future__ import annotations

import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore
from flax import linen as nn

from .._core import LinearCore


class Linear(nn.Module):
    """Flax wrapper around the shared linear core."""

    irreps_in: Irreps
    irreps_out: Irreps
    shared_weights: bool | None = None
    internal_weights: bool | None = None

    def setup(self) -> None:
        shared = self.shared_weights
        internal = self.internal_weights

        if shared is False and internal is None:
            internal = False

        if shared is None:
            shared = True

        if internal is None:
            internal = True

        self.core = LinearCore(
            self.irreps_in,
            self.irreps_out,
            shared_weights=shared,
            internal_weights=internal,
        )

    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        internal_weight: jnp.ndarray | None = None
        if self.core.internal_weights:
            internal_weight = self.param(
                'weight',
                self.core.init_weight,
            )

        out_mul_ir, _ = self.core.apply(
            x,
            weights=weights,
            internal_weight=internal_weight,
            math_dtype=jnp.float64,
        )
        return out_mul_ir

    @property
    def weight_numel(self) -> int:
        return self.core.weight_numel
