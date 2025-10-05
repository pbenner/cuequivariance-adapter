"""Cue-equivariant symmetric contraction implemented with segmented polynomials."""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_jax as cuex
import haiku as hk
import jax
import jax.numpy as jnp
from cuequivariance.group_theory.experimental.mace.symmetric_contractions import (
    symmetric_contraction as cue_mace_symmetric_contraction,
)
from e3nn_jax import Irreps

from .utility import ir_mul_to_mul_ir, mul_ir_to_ir_mul


class SymmetricContraction(hk.Module):
    r"""Symmetric contraction evaluated with cuequivariance-jax.

    Given an input feature vector ``x`` whose irreps are described by
    :attr:`irreps_in`, an integer ``index`` selecting one of ``num_elements``
    learned weight vectors, and a contraction degree ``correlation``, this
    module computes

    .. math::

        z_{w,k} = \sum_{d=1}^{C} \sum_{u_1,\ldots,u_d}
            w^{(d)}_{w,u_1,\ldots,u_d}
            \left\langle x_{u_1} \otimes \cdots \otimes x_{u_d},
            \mathrm{CG}_{u_1,\ldots,u_d \to k} \right\rangle,

    where ``C`` is ``correlation`` and the Clebsch–Gordan (CG) coefficients are
    supplied by the cue descriptor.  The element index ``w`` selects which row
    of the weight tensor is used for each batch item.  Inputs and outputs are in
    the familiar e3nn ``mul_ir`` layout; cue handles the segmented-polynomial
    evaluation in ``ir_mul`` order internally.  Depending on ``use_reduced_cg``
    the weights are either parameterised in the reduced CG basis or projected to
    the original MACE basis using the matrices returned by
    :mod:`cuequivariance`.
    """

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

        if correlation <= 0:
            raise ValueError('correlation must be a positive integer')
        if num_elements <= 0:
            raise ValueError('num_elements must be positive')

        self.correlation = correlation
        self.num_elements = num_elements
        self.use_reduced_cg = use_reduced_cg

        self.irreps_in_o3 = Irreps(irreps_in)
        self.irreps_out_o3 = Irreps(irreps_out)

        muls_in = {mul for mul, _ in self.irreps_in_o3}
        muls_out = {mul for mul, _ in self.irreps_out_o3}
        if len(muls_in) != 1 or len(muls_out) != 1 or muls_in != muls_out:
            raise ValueError(
                'SymmetricContraction requires all input/output irreps to share the same multiplicity'
            )
        self.mul = next(iter(muls_in))

        self.irreps_in_cue = cue.Irreps(cue.O3, irreps_in)
        self.irreps_out_cue = cue.Irreps(cue.O3, irreps_out)

        degrees = tuple(range(1, correlation + 1))
        descriptor, projection = cue_mace_symmetric_contraction(
            self.irreps_in_cue,
            self.irreps_out_cue,
            degrees,
        )
        self.descriptor = descriptor
        self.weight_irreps = descriptor.inputs[0].irreps
        self.weight_numel = self.weight_irreps.dim

        if use_reduced_cg:
            self.projection = None
            self.weight_basis_dim = self.weight_numel // self.mul
        else:
            self.projection = jnp.asarray(projection)
            self.weight_basis_dim = self.projection.shape[0]

        self.weight_param_shape = (self.num_elements, self.weight_basis_dim, self.mul)

    def __call__(
        self,
        x: jnp.ndarray,
        indices: jnp.ndarray,
    ) -> jnp.ndarray:
        dtype = x.dtype

        basis_weights = hk.get_parameter(
            'weight',
            shape=self.weight_param_shape,
            dtype=dtype,
            init=hk.initializers.RandomNormal(),
        )

        if self.projection is not None:
            projection = jnp.asarray(self.projection, dtype=dtype)
            weight_flat = jnp.einsum('zau,ab->zbu', basis_weights, projection)
        else:
            weight_flat = basis_weights

        weight_flat = weight_flat.reshape(self.num_elements, self.weight_numel)

        indices = jnp.asarray(indices)
        if indices.ndim == 2:
            indices = jnp.argmax(indices, axis=1)
        if indices.ndim != 1:
            raise ValueError(
                'indices must be a rank-1 array or a batch of one-hot vectors'
            )
        indices = indices.astype(jnp.int32)
        if jnp.any(indices < 0) or jnp.any(indices >= self.num_elements):
            raise ValueError('indices out of range for the available elements')

        selected_weights = weight_flat[indices]

        weight_rep = cuex.RepArray(
            self.weight_irreps,
            selected_weights,
            cue.ir_mul,
        )

        x_ir_mul = mul_ir_to_ir_mul(x, self.irreps_in_o3)
        x_rep = cuex.RepArray(
            self.irreps_in_cue,
            jnp.asarray(x_ir_mul, dtype=dtype),
            cue.ir_mul,
        )

        [out_ir_mul] = cuex.segmented_polynomial(
            self.descriptor.polynomial,
            [weight_rep.array, x_rep.array],
            [
                jax.ShapeDtypeStruct(
                    (*x.shape[:-1], self.irreps_out_o3.dim),
                    dtype,
                )
            ],
            method='naive',
            math_dtype=dtype,
        )

        out_mul_ir = ir_mul_to_mul_ir(out_ir_mul, self.irreps_out_o3)
        return out_mul_ir
