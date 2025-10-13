import jax
import jax.numpy as jnp
import pytest
from e3nn import o3
from e3nn_jax import Irreps

from cueq_e3nn_jax_adapter.flax.fully_connected_tensor_product import (
    FullyConnectedTensorProduct as FullyConnectedTensorProductFlax,
)
from tests._flax_builder_utils import resolve_flax_weights
from tests._fully_connected_test_utils import run_fully_connected_comparison

jax.config.update('jax_enable_x64', True)


def _build_flax_apply(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    weight_numel: int,
    shared_weights: bool,
    internal_weights: bool,
):
    module = FullyConnectedTensorProductFlax(
        Irreps(irreps_in1),
        Irreps(irreps_in2),
        Irreps(irreps_out),
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )
    zeros1 = jnp.zeros((1, irreps_in1.dim))
    zeros2 = jnp.zeros((1, irreps_in2.dim))
    init_weights = None
    if not internal_weights:
        init_weights = jnp.zeros((1, weight_numel))
    params = module.init(jax.random.PRNGKey(0), zeros1, zeros2, init_weights)

    flax_numel = module.apply(params, method=lambda m: m.weight_numel)
    if flax_numel != weight_numel:
        raise ValueError(
            f'flax weight_numel {flax_numel} does not match e3nn weight_numel {weight_numel}'
        )

    def apply_fn(x1, x2, weights):
        x1_array = jnp.asarray(x1)
        x2_array = jnp.asarray(x2)
        variables, call_weights = resolve_flax_weights(
            params,
            weights,
            batch_size=x1_array.shape[0],
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            weight_numel=weight_numel,
        )
        return module.apply(variables, x1_array, x2_array, call_weights)

    return apply_fn


FULLY_CONNECTED_CASES = [
    ('1x0e + 1x1o', '1x0e + 1x1o', '2x0e + 2x1o'),
    ('1x2e', '1x1e + 1x2e', '1x1o + 1x2o + 1x3e'),
    (
        '64x0e + 64x1o + 64x2e + 64x3o',
        '64x0e + 64x1o + 64x2e + 64x3o',
        '64x0e + 64x1o + 64x2e',
    ),
]

WEIGHT_CONFIGS = [
    pytest.param(False, False, id='external_unshared'),
    pytest.param(True, False, id='external_shared'),
    pytest.param(True, True, id='internal_shared'),
]


class TestFlaxFullyConnectedTensorProduct:
    tol = 1e-12

    @pytest.mark.parametrize('shared_weights, internal_weights', WEIGHT_CONFIGS)
    @pytest.mark.parametrize('irreps1, irreps2, irreps_out', FULLY_CONNECTED_CASES)
    def test_fully_connected_agreement(
        self,
        irreps1,
        irreps2,
        irreps_out,
        shared_weights,
        internal_weights,
    ):
        result = run_fully_connected_comparison(
            _build_flax_apply,
            irreps1,
            irreps2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        diff = result.max_diff
        assert diff <= self.tol, (
            f'max deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for {irreps1} ⊗ {irreps2} → {irreps_out}'
        )
