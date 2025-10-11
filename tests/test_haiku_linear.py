import haiku as hk
import jax
import jax.numpy as jnp
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.modules.wrapper_ops import Linear as LinearWrapper

from cuequivariance_adapter.haiku.linear import Linear as LinearCuex
from tests._haiku_builder_utils import find_weight_parameter, resolve_haiku_weights
from tests._linear_test_utils import run_linear_comparison

jax.config.update('jax_enable_x64', True)


def _build_cuex_apply(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    weight_numel: int,
    shared_weights: bool,
    internal_weights: bool,
):
    recorded: dict[str, int] = {}

    def forward(x, weights):
        module = LinearCuex(
            Irreps(irreps_in),
            Irreps(irreps_out),
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        recorded['numel'] = module.weight_numel
        return module(x, weights)

    transformed = hk.without_apply_rng(hk.transform(forward))
    zeros = jnp.zeros((1, irreps_in.dim))
    if internal_weights:
        params = transformed.init(jax.random.PRNGKey(0), zeros, None)
    else:
        zerosw = jnp.zeros((1, weight_numel))
        params = transformed.init(jax.random.PRNGKey(0), zeros, zerosw)

    cuex_numel = recorded.get('numel')
    if cuex_numel != weight_numel:
        raise ValueError(
            f'cuex weight_numel {cuex_numel} does not match e3nn weight_numel {weight_numel}'
        )

    weight_location: tuple[str, str] | None = None
    if internal_weights:
        weight_location = find_weight_parameter(params)

    def apply_fn(x, weights):
        nonlocal params
        x_array = jnp.asarray(x)
        next_params, call_weights = resolve_haiku_weights(
            params,
            weights,
            batch_size=x_array.shape[0],
            internal_weights=internal_weights,
            shared_weights=shared_weights,
            weight_numel=weight_numel,
            weight_location=weight_location,
            flatten_internal=True,
        )
        params = next_params
        return transformed.apply(next_params, x_array, call_weights)

    return apply_fn


LINEAR_CASES = [
    ('1x0e + 1x1o', '2x0e + 1x1o'),
    ('2x1e', '1x0e + 2x1e'),
    ('64x0e + 64x1o + 64x2e + 64x3o', '64x0e + 64x1o + 64x2e'),
]


class _BaseLinearTest:
    tol = 1e-12
    shared_weights: bool
    internal_weights: bool

    @pytest.mark.parametrize('irreps_in, irreps_out', LINEAR_CASES)
    def test_linear_agreement(self, irreps_in, irreps_out):
        result = run_linear_comparison(
            _build_cuex_apply,
            irreps_in,
            irreps_out,
            shared_weights=self.shared_weights,
            internal_weights=self.internal_weights,
        )
        diffs = [result.max_diff]
        if self.internal_weights:
            cue_config = CuEquivarianceConfig(enabled=True, optimize_linear=True)
            linear_cue = LinearWrapper(
                o3.Irreps(irreps_in),
                o3.Irreps(irreps_out),
                shared_weights=self.shared_weights,
                internal_weights=True,
                cueq_config=cue_config,
            )
            with torch.no_grad():
                if linear_cue.weight is None:
                    raise RuntimeError('cue Linear missing internal weight tensor')
                linear_cue.weight.copy_(result.base_weights.view(1, -1))
            out_cue = linear_cue(result.inputs)
            diffs.append((out_cue - result.adapter_output).abs().max().item())
            diffs.append((result.reference_output - out_cue).abs().max().item())
        diff = max(diffs)
        assert diff <= self.tol, (
            f'max deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for {irreps_in} → {irreps_out}'
        )


class TestLinearInternalShared(_BaseLinearTest):
    shared_weights = True
    internal_weights = True


class TestLinearExternalShared(_BaseLinearTest):
    shared_weights = True
    internal_weights = False
