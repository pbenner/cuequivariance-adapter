import haiku as hk
import jax
import jax.numpy as jnp
import pytest
from e3nn import o3
from e3nn_jax import Irreps

from cuequivariance_adapter.haiku.fully_connected_tensor_product import (
    FullyConnectedTensorProduct as FullyConnectedTensorProductCuex,
)
from tests._fully_connected_test_utils import run_fully_connected_comparison
from tests._haiku_builder_utils import find_weight_parameter, resolve_haiku_weights

jax.config.update('jax_enable_x64', True)


def _build_cuex_apply(
    irreps_in1: o3.Irreps,
    irreps_in2: o3.Irreps,
    irreps_out: o3.Irreps,
    weight_numel: int,
    shared_weights: bool,
    internal_weights: bool,
):
    recorded: dict[str, int] = {}

    def forward(x1, x2, weights):
        module = FullyConnectedTensorProductCuex(
            Irreps(irreps_in1),
            Irreps(irreps_in2),
            Irreps(irreps_out),
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        recorded['numel'] = module.weight_numel
        return module(x1, x2, weights)

    transformed = hk.without_apply_rng(hk.transform(forward))
    zeros1 = jnp.zeros((1, irreps_in1.dim))
    zeros2 = jnp.zeros((1, irreps_in2.dim))
    if internal_weights:
        params = transformed.init(jax.random.PRNGKey(0), zeros1, zeros2, None)
    else:
        zerosw = jnp.zeros((1, weight_numel))
        params = transformed.init(jax.random.PRNGKey(0), zeros1, zeros2, zerosw)

    cuex_numel = recorded.get('numel')
    if cuex_numel != weight_numel:
        raise ValueError(
            f'cuex weight_numel {cuex_numel} does not match e3nn weight_numel {weight_numel}'
        )

    weight_location: tuple[str, str] | None = None
    if internal_weights:
        weight_location = find_weight_parameter(params)

    def apply_fn(x1, x2, weights):
        nonlocal params
        next_params, call_weights = resolve_haiku_weights(
            params,
            weights,
            batch_size=x1.shape[0],
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            weight_numel=weight_numel,
            weight_location=weight_location,
        )
        params = next_params
        return transformed.apply(next_params, x1, x2, call_weights)

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


class TestFullyConnectedTensorProduct:
    tol = 1e-12

    @pytest.mark.parametrize('shared_weights, internal_weights', WEIGHT_CONFIGS)
    @pytest.mark.parametrize('irreps1, irreps2, irreps_out', FULLY_CONNECTED_CASES)
    def test_fully_connected_agreement(
        self, irreps1, irreps2, irreps_out, shared_weights, internal_weights
    ):
        result = run_fully_connected_comparison(
            _build_cuex_apply,
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
