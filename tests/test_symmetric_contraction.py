import cuequivariance as cue
import cuequivariance_torch as cuet
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3  # type: ignore
from e3nn_jax import Irreps  # type: ignore

from cuequivariance_adapter.symmetric_contraction import SymmetricContraction

jax.config.update('jax_enable_x64', True)


def _build_cuex_apply(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    correlation: int,
    num_elements: int,
    use_reduced_cg: bool,
):
    def forward(x, indices):
        module = SymmetricContraction(
            Irreps(irreps_in),
            Irreps(irreps_out),
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
            name='symmetric_contraction',
        )
        return module(x, indices)

    transformed = hk.without_apply_rng(hk.transform(forward))
    zeros_x = jnp.zeros((1, irreps_in.dim))
    zeros_idx = jnp.arange(1)
    params = transformed.init(jax.random.PRNGKey(0), zeros_x, zeros_idx)
    return transformed, params


def _compare_once(
    irreps_in: str,
    irreps_out: str,
    *,
    correlation: int,
    num_elements: int,
    use_reduced_cg: bool,
    batch: int = 5,
) -> float:
    irreps_in_o3 = o3.Irreps(irreps_in)
    irreps_out_o3 = o3.Irreps(irreps_out)

    cues_in = cue.Irreps(cue.O3, irreps_in)
    cues_out = cue.Irreps(cue.O3, irreps_out)

    cuet_module = cuet.SymmetricContraction(
        cues_in,
        cues_out,
        contraction_degree=correlation,
        num_elements=num_elements,
        layout_in=cue.mul_ir,
        layout_out=cue.mul_ir,
        original_mace=(not use_reduced_cg),
    )

    transformed, params = _build_cuex_apply(
        irreps_in_o3,
        irreps_out_o3,
        correlation,
        num_elements,
        use_reduced_cg,
    )

    weight_basis = cuet_module.weight.detach().cpu().numpy()
    mutable = hk.data_structures.to_mutable_dict(params)
    mutable['symmetric_contraction']['weight'] = jnp.asarray(weight_basis)
    params = hk.data_structures.to_immutable_dict(mutable)

    torch.manual_seed(0)
    x_torch = torch.randn(batch, irreps_in_o3.dim)
    indices_torch = torch.randint(0, num_elements, (batch,))

    out_cuet = cuet_module(x_torch, indices_torch)

    x_jax = jnp.asarray(x_torch.detach().cpu().numpy())
    indices_jax = jnp.asarray(indices_torch.detach().cpu().numpy())

    out_cuex = transformed.apply(params, x_jax, indices_jax)

    out_cuex = np.array(out_cuex, copy=False)
    out_cuet = out_cuet.detach().cpu().numpy()

    return float(np.max(np.abs(out_cuet - out_cuex)))


SYMMETRIC_CASES = [
    ('2x0e + 2x1o', '2x0e + 2x1o', 2, 3),
    ('1x0e + 1x1o', '1x0e + 1x1o', 3, 4),
    ('2x0e + 2x1o', '2x0e', 2, 5),
]


class TestSymmetricContraction:
    tol = 1e-11

    @pytest.mark.parametrize(
        'use_reduced_cg',
        [pytest.param(True, id='reduced'), pytest.param(False, id='original')],
    )
    @pytest.mark.parametrize(
        'irreps_in, irreps_out, correlation, num_elements', SYMMETRIC_CASES
    )
    def test_symmetric_contraction_agreement(
        self,
        irreps_in,
        irreps_out,
        correlation,
        num_elements,
        use_reduced_cg,
    ):
        diff = _compare_once(
            irreps_in,
            irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
        )
        assert diff <= self.tol, (
            f'max deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for correlation={correlation}, use_reduced_cg={use_reduced_cg}'
        )
