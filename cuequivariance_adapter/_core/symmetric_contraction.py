"""Shared implementation of the symmetric contraction adapter."""

from __future__ import annotations

from dataclasses import dataclass

import cuequivariance as cue
import cuequivariance_jax as cuex
import jax
import jax.numpy as jnp
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction as cue_mace_symmetric_contraction,
)  # type: ignore
from e3nn_jax import Irreps  # type: ignore

from .._utility import ir_mul_to_mul_ir


def _validate_features(x: jnp.ndarray, mul: int, feature_dim: int) -> None:
    """Ensure input tensors follow the expected mul_ir layout."""

    if x.ndim != 3 or x.shape[1] != mul or x.shape[2] != feature_dim:
        raise ValueError(
            'SymmetricContraction expects input with shape '
            f'(batch, {mul}, {feature_dim}); got {tuple(x.shape)}'
        )


def _select_weights(
    weight_flat: jnp.ndarray,
    selector: jnp.ndarray,
    *,
    dtype: jnp.dtype,
    num_elements: int,
) -> jnp.ndarray:
    """Select element weights by index or mixing matrix."""

    selector = jnp.asarray(selector)
    if selector.ndim == 1:
        idx = selector.astype(jnp.int32)
        if jnp.any(idx < 0) or jnp.any(idx >= num_elements):
            raise ValueError('indices out of range for the available elements')
        return weight_flat[idx]

    if selector.ndim == 2:
        if selector.shape[1] != num_elements:
            raise ValueError('Mixing matrix must have second dimension num_elements')
        mix = jnp.asarray(selector, dtype=dtype)
        return mix @ weight_flat

    raise ValueError('indices must be rank-1 (element ids) or rank-2 (mixing matrix)')


@dataclass(frozen=True)
class SymmetricContractionCore:
    """Core logic shared by Flax and Haiku symmetric contraction wrappers."""

    irreps_in: Irreps
    irreps_out: Irreps
    correlation: int
    num_elements: int
    use_reduced_cg: bool = True

    def __post_init__(self) -> None:
        if self.correlation <= 0:
            raise ValueError('correlation must be a positive integer')
        if self.num_elements <= 0:
            raise ValueError('num_elements must be positive')

        irreps_in_o3 = Irreps(self.irreps_in)
        irreps_out_o3 = Irreps(self.irreps_out)
        object.__setattr__(self, 'irreps_in_o3', irreps_in_o3)
        object.__setattr__(self, 'irreps_out_o3', irreps_out_o3)

        muls_in = {mul for mul, _ in irreps_in_o3}
        muls_out = {mul for mul, _ in irreps_out_o3}
        if len(muls_in) != 1 or len(muls_out) != 1 or muls_in != muls_out:
            raise ValueError(
                'SymmetricContraction requires all input/output irreps to share the same multiplicity'
            )
        mul = next(iter(muls_in))
        object.__setattr__(self, 'mul', mul)

        irreps_in_cue = cue.Irreps(cue.O3, irreps_in_o3)
        irreps_out_cue = cue.Irreps(cue.O3, irreps_out_o3)
        object.__setattr__(self, 'irreps_in_cue', irreps_in_cue)
        object.__setattr__(self, 'irreps_out_cue', irreps_out_cue)
        object.__setattr__(self, 'feature_dim', sum(ir.dim for _, ir in irreps_in_o3))
        object.__setattr__(self, 'irreps_in_cue_base', irreps_in_cue.set_mul(1))

        degrees = tuple(range(1, self.correlation + 1))
        descriptor, projection = cue_mace_symmetric_contraction(
            irreps_in_cue,
            irreps_out_cue,
            degrees,
        )
        object.__setattr__(self, 'descriptor', descriptor)
        weight_irreps = descriptor.inputs[0].irreps
        object.__setattr__(self, 'weight_irreps', weight_irreps)
        object.__setattr__(self, 'weight_numel', weight_irreps.dim)

        if self.use_reduced_cg:
            object.__setattr__(self, 'projection', None)
            basis_dim = weight_irreps.dim // mul
        else:
            proj = jnp.asarray(projection)
            object.__setattr__(self, 'projection', proj)
            basis_dim = proj.shape[0]
        object.__setattr__(self, 'weight_basis_dim', basis_dim)

    @property
    def weight_param_shape(self) -> tuple[int, int, int]:
        """Shape of the learnable basis weights."""

        return (self.num_elements, self.weight_basis_dim, self.mul)

    def init_weight(self, rng: jax.Array) -> jnp.ndarray:
        """Initialise basis weights with Gaussian noise."""

        return jax.random.normal(rng, self.weight_param_shape)

    def ensure_mul_ir_layout(self, x: jnp.ndarray, *, layout: str) -> jnp.ndarray:
        """Convert inputs to mul_ir layout if ``layout`` is ``ir_mul``."""

        if layout == 'mul_ir':
            return x
        if layout != 'ir_mul':
            raise ValueError(
                "input_layout must be either 'mul_ir' or 'ir_mul'; "
                f'got {layout!r}'
            )
        return self._convert_ir_mul_to_mul_ir(x)

    def _convert_ir_mul_to_mul_ir(self, x: jnp.ndarray) -> jnp.ndarray:
        """Reorder ir_mul input layout to mul_ir."""

        segments: list[jnp.ndarray] = []
        offset = 0
        for mul_ir in self.irreps_in_cue_base:
            width = mul_ir.ir.dim
            seg = x[:, offset : offset + width, :]
            if seg.shape[1] != width:
                raise ValueError('Input feature dimension mismatch with irreps.')
            segments.append(jnp.swapaxes(seg, -1, -2))
            offset += width

        if offset != x.shape[1]:
            raise ValueError('Input feature dimension mismatch with irreps.')

        if not segments:
            return jnp.swapaxes(x, -1, -2)

        return jnp.concatenate(segments, axis=-1)

    def _features_to_rep(self, x: jnp.ndarray, dtype: jnp.dtype) -> cuex.RepArray:
        """Pack mul_ir features into cue RepArray segments."""

        segments: list[jnp.ndarray] = []
        offset = 0
        for mul_ir in self.irreps_in_cue_base:
            width = mul_ir.ir.dim
            seg = x[:, :, offset : offset + width]
            if seg.shape[-1] != width:
                raise ValueError('Input feature dimension mismatch with irreps.')
            segments.append(jnp.swapaxes(seg, -2, -1))
            offset += width

        return cuex.from_segments(
            self.irreps_in_cue,
            segments,
            (x.shape[0], self.mul),
            cue.ir_mul,
            dtype=dtype,
        )

    def _project_basis_weights(
        self,
        basis_weights: jnp.ndarray,
        *,
        dtype: jnp.dtype,
    ) -> jnp.ndarray:
        """Project basis weights when using the full CG basis."""

        if self.projection is None:
            return jnp.asarray(basis_weights, dtype=dtype)
        projection = jnp.asarray(self.projection, dtype=dtype)
        return jnp.einsum('zau,ab->zbu', basis_weights.astype(dtype), projection)

    def _weight_rep_from_indices(
        self,
        basis_weights: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        dtype: jnp.dtype,
    ) -> cuex.RepArray:
        """Select element weights and wrap them in a cue RepArray."""

        projected = self._project_basis_weights(basis_weights, dtype=dtype)
        weight_flat = projected.reshape(self.num_elements, self.weight_numel)
        selected = _select_weights(
            weight_flat,
            indices,
            dtype=dtype,
            num_elements=self.num_elements,
        )
        return cuex.RepArray(self.weight_irreps, selected, cue.ir_mul)

    def apply(
        self,
        x: jnp.ndarray,
        indices: jnp.ndarray,
        *,
        basis_weights: jnp.ndarray,
        math_dtype: jnp.dtype,
    ) -> jnp.ndarray:
        """Evaluate the symmetric contraction and return mul_ir output."""

        array = jnp.asarray(x)
        dtype = array.dtype

        _validate_features(array, self.mul, self.feature_dim)

        weight_rep = self._weight_rep_from_indices(
            basis_weights,
            indices,
            dtype=dtype,
        )
        x_rep = self._features_to_rep(array, dtype)

        [out_ir_mul] = cuex.segmented_polynomial(
            self.descriptor.polynomial,
            [weight_rep.array, x_rep.array],
            [
                jax.ShapeDtypeStruct(
                    (array.shape[0], self.irreps_out_o3.dim),
                    dtype,
                )
            ],
            method='naive',
            math_dtype=math_dtype,
        )

        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, self.irreps_out_o3)
        return out_mul_ir
