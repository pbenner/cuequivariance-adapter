"""Helpers for constructing Flax module apply functions in tests."""

from __future__ import annotations

from typing import Tuple

import jax.numpy as jnp
from flax.core import freeze, unfreeze


def resolve_flax_weights(
    params,
    weights,
    *,
    batch_size: int,
    internal_weights: bool,
    shared_weights: bool,
    weight_numel: int,
) -> Tuple[dict, jnp.ndarray | None]:
    """Return updated parameter variables and the weight argument for Flax modules."""

    variables = params
    if internal_weights:
        if weights is not None:
            weight_array = jnp.asarray(weights)
            if weight_array.ndim == 1:
                weight_array = weight_array[jnp.newaxis, :]
            elif weight_array.ndim == 2:
                if weight_array.shape[0] != 1:
                    raise ValueError(
                        'Internal weights expect a single shared weight vector'
                    )
                weight_array = weight_array[:1]
            else:
                raise ValueError('Internal weights must have rank 1 or 2')

            if weight_array.shape[-1] != weight_numel:
                raise ValueError(
                    f'Expected weights last dimension {weight_numel}, got {weight_array.shape[-1]}'
                )
            mutable = unfreeze(params)
            mutable['params']['weight'] = weight_array
            variables = freeze(mutable)
        return variables, None

    if weights is None:
        raise ValueError('External weights must be provided')

    weight_array = jnp.asarray(weights)
    if weight_array.ndim == 1:
        weight_array = weight_array[jnp.newaxis, :]
    elif weight_array.ndim != 2:
        raise ValueError(f'Weights must have rank 1 or 2, got rank {weight_array.ndim}')

    if weight_array.shape[-1] != weight_numel:
        raise ValueError(
            f'Expected weights last dimension {weight_numel}, got {weight_array.shape[-1]}'
        )

    leading = weight_array.shape[0]
    if shared_weights:
        if leading not in (1, batch_size):
            raise ValueError(
                'Shared weights require leading dimension 1 or equal to the batch size'
            )
        if leading == 1 and batch_size != 1:
            weight_array = jnp.broadcast_to(weight_array, (batch_size, weight_numel))
    else:
        if leading != batch_size:
            raise ValueError(
                'Unshared weights require leading dimension equal to the batch size'
            )

    return variables, weight_array
