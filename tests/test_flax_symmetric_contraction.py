import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3  # type: ignore
from e3nn_jax import Irreps  # type: ignore
from flax.core import freeze, unfreeze
from mace.modules.wrapper_ops import (  # type: ignore
    CuEquivarianceConfig,
    SymmetricContractionWrapper,
)

from cuequivariance_adapter.flax.symmetric_contraction import SymmetricContraction

jax.config.update('jax_enable_x64', True)


def _init_flax_module(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    *,
    correlation: int,
    num_elements: int,
    use_reduced_cg: bool,
):
    info: dict[str, object] = {}

    mul = irreps_in[0].mul
    feature_width = sum(ir.dim for _, ir in irreps_in)

    module = SymmetricContraction(
        Irreps(irreps_in),
        Irreps(irreps_out),
        correlation=correlation,
        num_elements=num_elements,
        use_reduced_cg=use_reduced_cg,
    )

    zeros_x = jnp.zeros((1, mul, feature_width), dtype=jnp.float64)
    zeros_idx = jnp.zeros((1,), dtype=jnp.int32)
    params = module.init(jax.random.PRNGKey(0), zeros_x, zeros_idx)
    info['weight_shape'] = module.apply(params, method=lambda m: m.weight_param_shape)
    return module, params, info


def _flatten_mul_ir(x: torch.Tensor) -> torch.Tensor:
    return x.transpose(1, 2).reshape(x.shape[0], -1)


def _compare_to_cuet(
    irreps_in: str,
    irreps_out: str,
    *,
    correlation: int,
    num_elements: int,
    use_reduced_cg: bool,
    batch: int = 5,
    seed: int = 0,
    use_per_node_mix: bool = False,
) -> float:
    irreps_in_o3 = o3.Irreps(irreps_in)
    irreps_out_o3 = o3.Irreps(irreps_out)

    module, params, info = _init_flax_module(
        irreps_in_o3,
        irreps_out_o3,
        correlation=correlation,
        num_elements=num_elements,
        use_reduced_cg=use_reduced_cg,
    )

    cueq_config = CuEquivarianceConfig(enabled=True, optimize_symmetric=True)
    cue_torch = SymmetricContractionWrapper(
        irreps_in_o3,
        irreps_out_o3,
        correlation=correlation,
        num_elements=num_elements,
        cueq_config=cueq_config,
        use_reduced_cg=use_reduced_cg,
    ).double()

    cue_weights = cue_torch.weight.detach().cpu().numpy().astype(np.float64, copy=False)
    weight_shape = info['weight_shape']
    if cue_weights.shape != weight_shape:
        raise AssertionError(
            f'cue-equivariant weight shape {cue_weights.shape} does not match adapter shape {weight_shape}'
        )

    mutable = unfreeze(params)
    weight_param = mutable['params']['weight']
    dtype = weight_param.dtype
    mutable['params']['weight'] = jnp.asarray(cue_weights, dtype=dtype)
    params = freeze(mutable)

    rng = np.random.default_rng(seed)
    mul = irreps_in_o3[0].mul
    feature_width = sum(ir.dim for _, ir in irreps_in_o3)
    x_features = rng.standard_normal((batch, mul, feature_width)).astype(np.float64)
    if use_per_node_mix:
        mix_matrix = rng.standard_normal((batch, num_elements)).astype(np.float64)
        selector = jnp.asarray(mix_matrix, dtype=dtype)
    else:
        indices = rng.integers(0, num_elements, size=(batch,), dtype=np.int32)
        selector = jnp.asarray(indices)
        mix_matrix = np.eye(num_elements, dtype=np.float64)[indices]

    out_flax = module.apply(
        params,
        jnp.asarray(x_features, dtype=dtype),
        selector,
    )

    x_torch = torch.from_numpy(x_features).to(dtype=torch.double)
    x_flat = _flatten_mul_ir(x_torch)
    mix_torch = torch.from_numpy(mix_matrix).to(dtype=torch.double)

    out_cuet_flat = torch.zeros(
        (batch, irreps_out_o3.dim), dtype=torch.double, device=x_flat.device
    )
    for element_idx in range(num_elements):
        elem_indices = torch.full(
            (batch,), element_idx, dtype=torch.int32, device=x_flat.device
        )
        partial = cue_torch(x_flat, elem_indices)
        out_cuet_flat = (
            out_cuet_flat + mix_torch[:, element_idx : element_idx + 1] * partial
        )

    out_flax_np = np.asarray(out_flax)
    out_cuet_np = out_cuet_flat.detach().cpu().numpy()

    return float(np.max(np.abs(out_flax_np - out_cuet_np)))


SYMMETRIC_CASES = [
    ('2x0e + 2x1o', '2x0e + 2x1o', 2, 3),
    ('1x0e + 1x1o', '1x0e + 1x1o', 3, 4),
    ('2x0e + 2x1o', '2x0e', 2, 5),
    ('4x0e + 4x1o', '4x0e + 4x1o', 3, 6),
    ('4x0e + 4x1o', '4x0e', 3, 6),
    ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 2, 10),
    ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 3, 10),
]


class TestFlaxSymmetricContraction:
    tol = 1e-11

    @pytest.mark.parametrize('use_reduced_cg', [True, False])
    @pytest.mark.parametrize(
        'irreps_in, irreps_out, correlation, num_elements', SYMMETRIC_CASES
    )
    def test_matches_cuet_symmetric_contraction(
        self,
        irreps_in,
        irreps_out,
        correlation,
        num_elements,
        use_reduced_cg,
    ):
        diff = _compare_to_cuet(
            irreps_in,
            irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
        )
        assert diff <= self.tol, (
            f'cue-equivariant comparison deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for correlation={correlation} use_reduced_cg={use_reduced_cg}'
        )
