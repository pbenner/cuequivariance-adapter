"""Cue-equivariant tensor product implemented via segmented polynomials."""

from __future__ import annotations

import jax.numpy as jnp
from e3nn_jax import Irreps


def mul_ir_to_ir_mul(array: jnp.ndarray, irreps: Irreps) -> jnp.ndarray:
    """Reorder the last axis of ``array`` from mul_ir to ir_mul layout."""

    if irreps.dim == 0:
        return array

    leading_shape = array.shape[:-1]
    array = array.reshape(*leading_shape, irreps.dim)
    segments: list[jnp.ndarray] = []
    offset = 0
    for mul, ir in irreps:
        block = array[..., offset : offset + mul * ir.dim]
        offset += mul * ir.dim
        block = block.reshape(*leading_shape, mul, ir.dim)
        block = jnp.swapaxes(block, -1, -2)  # -> (..., ir_dim, mul)
        block = block.reshape(*leading_shape, ir.dim * mul)
        segments.append(block)
    return jnp.concatenate(segments, axis=-1) if segments else array


def ir_mul_to_mul_ir(array: jnp.ndarray, irreps: Irreps) -> jnp.ndarray:
    """Reorder the last axis of ``array`` from ir_mul back to mul_ir layout."""

    if irreps.dim == 0:
        return array

    leading_shape = array.shape[:-1]
    array = array.reshape(*leading_shape, irreps.dim)
    segments: list[jnp.ndarray] = []
    offset = 0
    for mul, ir in irreps:
        block = array[..., offset : offset + mul * ir.dim]
        offset += mul * ir.dim
        block = block.reshape(*leading_shape, ir.dim, mul)
        block = jnp.swapaxes(block, -1, -2)  # -> (..., mul, ir_dim)
        block = block.reshape(*leading_shape, mul * ir.dim)
        segments.append(block)
    return jnp.concatenate(segments, axis=-1) if segments else array
