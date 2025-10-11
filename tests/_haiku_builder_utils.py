"""Helpers for Haiku test modules to manage weight parameters."""

from __future__ import annotations

from typing import Tuple

import haiku as hk
import jax.numpy as jnp


def find_weight_parameter(params: hk.Params) -> tuple[str, str]:
    """Return the module scope and parameter name for the first 'weight' entry."""

    for module_name, module_params in params.items():
        if 'weight' in module_params:
            return module_name, 'weight'
    raise RuntimeError('Weight parameter not found in Haiku params')


def resolve_haiku_weights(
    params: hk.Params,
    weights,
    *,
    batch_size: int,
    internal_weights: bool,
    shared_weights: bool,
    weight_numel: int,
    weight_location: tuple[str, str] | None,
    flatten_internal: bool = False,
) -> Tuple[hk.Params, jnp.ndarray | None]:
    """Update parameters for internal weights or validate external weights."""

    if internal_weights:
        if weights is None:
            return params, None

        weight_array = jnp.asarray(weights)
        if weight_array.ndim == 1:
            weight_array = weight_array[jnp.newaxis, :]
        elif weight_array.ndim == 2:
            if weight_array.shape[0] != 1:
                raise ValueError('Internal weights expect a single shared weight vector')
            weight_array = weight_array[:1]
        else:
            raise ValueError('Internal weights must have rank 1 or 2')

        if weight_array.shape[-1] != weight_numel:
            raise ValueError(
                f'Expected weights last dimension {weight_numel}, got {weight_array.shape[-1]}'
            )
        if flatten_internal:
            weight_array = weight_array.reshape(-1)
        if weight_location is None:
            weight_location = find_weight_parameter(params)
        module_name, param_name = weight_location
        mutable = hk.data_structures.to_mutable_dict(params)
        mutable[module_name][param_name] = weight_array
        return hk.data_structures.to_immutable_dict(mutable), None

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

    return params, weight_array
