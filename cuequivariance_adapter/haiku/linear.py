"""Cue-equivariant linear layer."""

from __future__ import annotations

import jax.numpy as jnp
from e3nn_jax import Irreps  # type: ignore
import haiku as hk

from .._core import LinearCore


class Linear(hk.Module):
    """Haiku wrapper around the shared linear core."""

    def __init__(
        self,
        irreps_in: Irreps,
        irreps_out: Irreps,
        *,
        shared_weights: bool | None = None,
        internal_weights: bool | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)

        if shared_weights is False and internal_weights is None:
            internal_weights = False

        if shared_weights is None:
            shared_weights = True

        if internal_weights is None:
            internal_weights = True

        self.core = LinearCore(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )

    def __call__(
        self,
        x: jnp.ndarray,
        weights: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        internal_weight = None
        if self.core.internal_weights:
            internal_weight = hk.get_parameter(
                'weight',
                shape=(self.core.weight_numel,),
                dtype=x.dtype,
                init=hk.initializers.RandomNormal(),
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
