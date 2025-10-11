import haiku as hk
import jax
import jax.numpy as jnp
import pytest
from e3nn import o3  # type: ignore

from cuequivariance_adapter.haiku.tensor_product import (
    TensorProduct as TensorProductCuex,
)
from tests._tensor_product_test_utils import run_tensor_product_comparison

jax.config.update('jax_enable_x64', True)


def _build_cuex_apply(
    irreps1_o3: o3.Irreps,
    irreps2_o3: o3.Irreps,
    target_o3: o3.Irreps,
    weight_numel: int,
    shared_weights: bool,
    internal_weights: bool,
    instructions=None,
):
    """Create a pure function that evaluates TensorProductCuex with frozen params."""

    recorded: dict[str, int] = {}

    def forward(x1, x2, weights):
        module = TensorProductCuex(
            irreps1_o3,
            irreps2_o3,
            target_o3,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        recorded['numel'] = module.weight_numel
        return module(x1, x2, weights)

    transformed = hk.without_apply_rng(hk.transform(forward))
    zeros1 = jnp.zeros((1, irreps1_o3.dim))
    zeros2 = jnp.zeros((1, irreps2_o3.dim))
    if internal_weights:
        params = transformed.init(jax.random.PRNGKey(0), zeros1, zeros2, None)
    else:
        zerosw = jnp.zeros((1, weight_numel))
        params = transformed.init(jax.random.PRNGKey(0), zeros1, zeros2, zerosw)

    cuex_numel = recorded.get('numel')
    if cuex_numel is None:
        raise RuntimeError(
            'TensorProductCuex failed to report weight_numel during init'
        )
    if cuex_numel != weight_numel:
        raise ValueError(
            f'cuex weight_numel {cuex_numel} does not match e3nn weight_numel {weight_numel}'
        )

    weight_location: tuple[str, str] | None = None
    if internal_weights:
        for module_name, module_params in params.items():
            if 'weight' in module_params:
                weight_location = (module_name, 'weight')
                break
        if weight_location is None:
            raise RuntimeError('TensorProductCuex internal weight parameter not found')

    def apply_fn(x1, x2, weights):
        nonlocal params
        call_weights = weights
        next_params = params
        if internal_weights:
            if weights is not None:
                mutable = hk.data_structures.to_mutable_dict(params)
                weight_value = jnp.asarray(weights)
                if weight_value.ndim == 1:
                    weight_value = weight_value[jnp.newaxis, :]
                elif weight_value.ndim == 2:
                    if weight_value.shape[0] != 1:
                        raise ValueError(
                            'TensorProductCuex internal weights expect a single shared weight vector'
                        )
                    weight_value = weight_value[:1]
                else:
                    raise ValueError(
                        'TensorProductCuex internal weights must be rank 1 or 2'
                    )
                module_name, param_name = weight_location
                mutable[module_name][param_name] = weight_value
                next_params = hk.data_structures.to_immutable_dict(mutable)
                params = next_params
            call_weights = None
        return transformed.apply(next_params, x1, x2, call_weights)

    return apply_fn


class TensorProductCuet:
    """Wrapper around o3.TensorProduct/cuet.ChannelwiseTensorProduct/oeq.TensorProduct followed by a scatter sum"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        shared_weights: bool = False,
        internal_weights: bool = False,
        instructions=None,
        cueq_config=None,
        oeq_config=None,
    ):
        raise RuntimeError('TensorProductCuet should not be instantiated in tests')


TENSOR_PRODUCT_CASES = [
    ('2x0e + 1x1o', '1x0e + 1x1o', '3x0e + 3x1o + 1x2e'),
    ('3x1e', '1x0e + 1x1e + 1x2e', '3x0e + 6x1e + 3x2e'),
    ('1x2o + 2x1e', '1x0e + 1x1o', '3x1e + 3x2o + 1x3e'),
    ('2x0e', '2x0e', '2x0e'),
    (
        '32x0e + 32x1o',
        '32x0e + 32x1o',
        '64x0e + 64x1o + 32x2e',
    ),
]

WEIGHT_CONFIGS = [
    pytest.param(False, False, id='external_unshared'),
    pytest.param(True, False, id='external_shared'),
    pytest.param(True, True, id='internal_shared'),
]


class TestTensorProduct:
    tol = 1e-12

    @pytest.mark.parametrize('shared_weights, internal_weights', WEIGHT_CONFIGS)
    @pytest.mark.parametrize('irreps1, irreps2, irreps_target', TENSOR_PRODUCT_CASES)
    def test_tensor_product_agreement(
        self, irreps1, irreps2, irreps_target, shared_weights, internal_weights
    ):
        diff = run_tensor_product_comparison(
            _build_cuex_apply,
            irreps1,
            irreps2,
            irreps_target,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        assert diff <= self.tol, (
            f'max deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for {irreps1} ⊗ {irreps2} → {irreps_target}'
        )
