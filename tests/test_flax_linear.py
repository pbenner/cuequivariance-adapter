import jax
import jax.numpy as jnp
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.modules.wrapper_ops import Linear as LinearWrapper

from cuequivariance_adapter.flax.linear import Linear as LinearFlax
from tests._flax_builder_utils import resolve_flax_weights
from tests._linear_test_utils import run_linear_comparison

jax.config.update('jax_enable_x64', True)


def _build_flax_apply(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    weight_numel: int,
    shared_weights: bool,
    internal_weights: bool,
):
    module = LinearFlax(
        Irreps(irreps_in),
        Irreps(irreps_out),
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )
    zeros = jnp.zeros((1, irreps_in.dim))
    init_weights = None
    if not internal_weights:
        init_weights = jnp.zeros((1, weight_numel))
    params = module.init(jax.random.PRNGKey(0), zeros, init_weights)

    flax_numel = module.apply(params, method=lambda m: m.weight_numel)
    if flax_numel != weight_numel:
        raise ValueError(
            f'flax weight_numel {flax_numel} does not match e3nn weight_numel {weight_numel}'
        )

    def apply_fn(x, weights):
        x_array = jnp.asarray(x)
        variables, call_weights = resolve_flax_weights(
            params,
            weights,
            batch_size=x_array.shape[0],
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            weight_numel=weight_numel,
        )
        return module.apply(variables, x_array, call_weights)

    return apply_fn


LINEAR_CASES = [
    ('2x0e + 2x1o', '3x0e + 3x1o'),
    ('3x0e + 3x1e', '3x0e + 2x1e + 1x2e'),
    ('4x0e + 4x1o', '4x0e + 4x1o'),
    ('128x0e+128x1o+128x2e', '128x0e+128x1o'),
]

WEIGHT_CONFIGS = [
    pytest.param(False, False, id='external_unshared'),
    pytest.param(True, False, id='external_shared'),
    pytest.param(True, True, id='internal_shared'),
]


class TestFlaxLinear:
    tol = 1e-12

    @pytest.mark.parametrize('shared_weights, internal_weights', WEIGHT_CONFIGS)
    @pytest.mark.parametrize('irreps_in, irreps_out', LINEAR_CASES)
    def test_linear_agreement(
        self,
        irreps_in,
        irreps_out,
        shared_weights,
        internal_weights,
    ):
        result = run_linear_comparison(
            _build_flax_apply,
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        diffs = [result.max_diff]
        if internal_weights:
            cue_config = CuEquivarianceConfig(enabled=True, optimize_linear=True)
            linear_cue = LinearWrapper(
                o3.Irreps(irreps_in),
                o3.Irreps(irreps_out),
                shared_weights=shared_weights,
                internal_weights=True,
                cueq_config=cue_config,
            )
            with torch.no_grad():
                if linear_cue.weight is None:
                    raise RuntimeError('cue Linear missing internal weight tensor')
                linear_cue.weight.copy_(result.base_weights.view(1, -1))
            out_cue = linear_cue(result.inputs)
            diffs.append((out_cue - result.adapter_output).abs().max().item())
        diff = max(diffs)
        assert diff <= self.tol, (
            f'max deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for Linear {irreps_in} → {irreps_out}'
        )
