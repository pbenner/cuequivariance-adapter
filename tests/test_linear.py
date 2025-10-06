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
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.modules.wrapper_ops import Linear as LinearWrapper

from cuequivariance_adapter.linear import Linear as LinearCuex

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
        for module_name, module_params in params.items():
            if 'weight' in module_params:
                weight_location = (module_name, 'weight')
                break
        if weight_location is None:
            raise RuntimeError('Linear cuex internal weight parameter not found')

    def apply_fn(x, weights):
        nonlocal params
        next_params = params
        call_weights = weights
        if internal_weights:
            if weights is not None:
                weight_value = jnp.asarray(weights)
                if weight_value.ndim == 2:
                    if weight_value.shape[0] != 1:
                        raise ValueError('Internal weights expect a single vector')
                    weight_value = weight_value.reshape(-1)
                elif weight_value.ndim != 1:
                    raise ValueError('Internal weights must be rank 1 or 2')
                mutable = hk.data_structures.to_mutable_dict(params)
                module_name, param_name = weight_location
                mutable[module_name][param_name] = weight_value
                next_params = hk.data_structures.to_immutable_dict(mutable)
                params = next_params
            call_weights = None
        else:
            call_weights = jnp.asarray(weights)
        x_array = jnp.asarray(x)
        return transformed.apply(next_params, x_array, call_weights)

    return apply_fn


def _compare_linear(
    irreps_in: str,
    irreps_out: str,
    *,
    shared_weights: bool,
    internal_weights: bool,
    batch: int = 4,
) -> float:
    if internal_weights and not shared_weights:
        raise ValueError('internal_weights=True requires shared_weights=True')

    irreps_in_o3 = o3.Irreps(irreps_in)
    irreps_out_o3 = o3.Irreps(irreps_out)

    linear_ref = LinearWrapper(
        irreps_in_o3,
        irreps_out_o3,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        cueq_config=None,
    )

    weight_numel = linear_ref.weight_numel
    cuex_apply = _build_cuex_apply(
        irreps_in_o3,
        irreps_out_o3,
        weight_numel,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )

    torch.manual_seed(0)
    x = torch.randn(batch, irreps_in_o3.dim)

    if internal_weights:
        base_weights = linear_ref.weight.detach().clone()
        weights_arg_ref = None
        weights_array = base_weights.detach().cpu().numpy()
    else:
        base_weights = torch.randn(1, weight_numel)
        weights_arg_ref = base_weights.view(-1)
        weights_array = base_weights.detach().cpu().numpy()

    x_jax = jnp.asarray(x.detach().cpu().numpy())
    weights_jax = None if internal_weights else jnp.asarray(weights_array)

    out_ref = (
        linear_ref(x) if weights_arg_ref is None else linear_ref(x, weights_arg_ref)
    )

    out_cuex = cuex_apply(
        x_jax,
        weights_array if internal_weights else weights_jax,
    )
    out_cuex = torch.from_numpy(np.array(out_cuex, copy=True))

    diffs = [(out_ref - out_cuex).abs().max().item()]

    if internal_weights:
        cue_config = CuEquivarianceConfig(enabled=True, optimize_linear=True)
        linear_cue = LinearWrapper(
            irreps_in_o3,
            irreps_out_o3,
            shared_weights=shared_weights,
            internal_weights=True,
            cueq_config=cue_config,
        )
        with torch.no_grad():
            if linear_cue.weight is None:
                raise RuntimeError('cue Linear missing internal weight tensor')
            linear_cue.weight.copy_(base_weights.view(1, -1))
        out_cue = linear_cue(x)
        diffs.append((out_cue - out_cuex).abs().max().item())
        diffs.append((out_ref - out_cue).abs().max().item())

    return max(diffs)


LINEAR_CASES = [
    ('1x0e + 1x1o', '2x0e + 1x1o'),
    ('2x1e', '1x0e + 2x1e'),
]


class _BaseLinearTest:
    tol = 1e-12
    shared_weights: bool
    internal_weights: bool

    @pytest.mark.parametrize('irreps_in, irreps_out', LINEAR_CASES)
    def test_linear_agreement(self, irreps_in, irreps_out):
        diff = _compare_linear(
            irreps_in,
            irreps_out,
            shared_weights=self.shared_weights,
            internal_weights=self.internal_weights,
        )
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
