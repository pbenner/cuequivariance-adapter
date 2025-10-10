import cuequivariance as cue
import cuequivariance_torch as cuet
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps

from cuequivariance_adapter.haiku.fully_connected_tensor_product import (
    FullyConnectedTensorProduct as FullyConnectedTensorProductCuex,
)

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
        for module_name, module_params in params.items():
            if 'weight' in module_params:
                weight_location = (module_name, 'weight')
                break
        if weight_location is None:
            raise RuntimeError(
                'FullyConnectedTensorProduct cuex internal weight not found'
            )

    def apply_fn(x1, x2, weights):
        nonlocal params
        next_params = params
        call_weights = weights
        if internal_weights:
            if weights is not None:
                mutable = hk.data_structures.to_mutable_dict(params)
                weight_value = jnp.asarray(weights)
                if weight_value.ndim == 1:
                    weight_value = weight_value[jnp.newaxis, :]
                elif weight_value.ndim == 2:
                    if weight_value.shape[0] != 1:
                        raise ValueError('Internal weights expect a single vector')
                    weight_value = weight_value[:1]
                else:
                    raise ValueError('Internal weights must be rank 1 or 2')
                module_name, param_name = weight_location
                mutable[module_name][param_name] = weight_value
                next_params = hk.data_structures.to_immutable_dict(mutable)
                params = next_params
            call_weights = None
        return transformed.apply(next_params, x1, x2, call_weights)

    return apply_fn


def compare_once(
    irreps1: str,
    irreps2: str,
    irreps_out: str,
    *,
    shared_weights: bool,
    internal_weights: bool,
    batch: int = 4,
) -> float:
    if internal_weights and not shared_weights:
        raise ValueError('internal_weights=True requires shared_weights=True')

    irreps1_o3 = o3.Irreps(irreps1)
    irreps2_o3 = o3.Irreps(irreps2)
    irreps_out_o3 = o3.Irreps(irreps_out)

    tp_e3nn = o3.FullyConnectedTensorProduct(
        irreps1_o3,
        irreps2_o3,
        irreps_out_o3,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )

    tp_cue = cuet.FullyConnectedTensorProduct(
        cue.Irreps(cue.O3, irreps1_o3),
        cue.Irreps(cue.O3, irreps2_o3),
        cue.Irreps(cue.O3, irreps_out_o3),
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )

    cuex_apply = _build_cuex_apply(
        irreps1_o3,
        irreps2_o3,
        irreps_out_o3,
        tp_e3nn.weight_numel,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )

    torch.manual_seed(0)
    x1 = torch.randn(batch, irreps1_o3.dim)
    x2 = torch.randn(batch, irreps2_o3.dim)

    if internal_weights:
        base_weights = tp_e3nn.weight.detach().clone()
        weight_tensor = base_weights.view(1, -1)
        with torch.no_grad():
            if tp_cue.weight is None:
                raise RuntimeError('cue FullyConnectedTensorProduct missing weights')
            tp_cue.weight.copy_(weight_tensor)
        weights_arg_e3nn = None
        weights_arg_cue = None
    elif shared_weights:
        base_weights = torch.randn(1, tp_e3nn.weight_numel)
        weights_arg_e3nn = base_weights.view(-1)
        weights_arg_cue = base_weights
        weight_tensor = base_weights
    else:
        base_weights = torch.randn(batch, tp_e3nn.weight_numel)
        weights_arg_e3nn = base_weights
        weights_arg_cue = base_weights
        weight_tensor = base_weights

    x1_jax = jnp.asarray(x1.detach().cpu().numpy())
    x2_jax = jnp.asarray(x2.detach().cpu().numpy())
    weights_jax = jnp.asarray(weight_tensor.detach().cpu().numpy())

    if weights_arg_e3nn is None:
        out_e3nn = tp_e3nn(x1, x2)
    else:
        out_e3nn = tp_e3nn(x1, x2, weights_arg_e3nn)

    if weights_arg_cue is None:
        out_cue = tp_cue(x1, x2)
    else:
        out_cue = tp_cue(x1, x2, weights_arg_cue)

    out_cuex = cuex_apply(x1_jax, x2_jax, weights_jax)
    out_cuex = torch.from_numpy(np.array(out_cuex, copy=True))

    diff_cuet = (out_e3nn - out_cue).abs().max().item()
    diff_cuex = (out_e3nn - out_cuex).abs().max().item()
    diff_cross = (out_cue - out_cuex).abs().max().item()

    return max(diff_cuet, diff_cuex, diff_cross)


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
        diff = compare_once(
            irreps1,
            irreps2,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        assert diff <= self.tol, (
            f'max deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for {irreps1} ⊗ {irreps2} → {irreps_out}'
        )
