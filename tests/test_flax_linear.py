import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from e3nn_jax import Irreps
from flax.core import freeze, unfreeze
from mace.modules.wrapper_ops import CuEquivarianceConfig
from mace.modules.wrapper_ops import Linear as LinearWrapper

from cuequivariance_adapter.flax.linear import Linear as LinearFlax

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
        variables = params
        call_weights = weights
        if internal_weights:
            if weights is not None:
                weight_array = jnp.asarray(weights)
                if weight_array.ndim == 2:
                    if weight_array.shape[0] != 1:
                        raise ValueError(
                            'Internal weights expect a single shared vector'
                        )
                    weight_array = weight_array[:1]
                elif weight_array.ndim == 1:
                    weight_array = weight_array[jnp.newaxis, :]
                else:
                    raise ValueError('Internal weights must have rank 1 or 2')
                mutable = unfreeze(params)
                mutable['params']['weight'] = weight_array
                variables = freeze(mutable)
            call_weights = None
        else:
            call_weights = jnp.asarray(weights)
        x_array = jnp.asarray(x)
        return module.apply(variables, x_array, call_weights)

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
    flax_apply = _build_flax_apply(
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
        weights_jax = None
    elif shared_weights:
        base_weights = torch.randn(1, weight_numel)
        weights_arg_ref = base_weights.view(-1)
        weights_array = base_weights.detach().cpu().numpy()
        weights_jax = jnp.asarray(weights_array)
    else:
        base_weights = torch.randn(batch, weight_numel)
        weights_arg_ref = base_weights
        weights_array = base_weights.detach().cpu().numpy()
        weights_jax = jnp.asarray(weights_array)

    x_jax = jnp.asarray(x.detach().cpu().numpy())

    out_ref = (
        linear_ref(x) if weights_arg_ref is None else linear_ref(x, weights_arg_ref)
    )

    out_flax = flax_apply(
        x_jax,
        weights_array if internal_weights else weights_jax,
    )
    out_flax = torch.from_numpy(np.array(out_flax, copy=True))

    diffs = [(out_ref - out_flax).abs().max().item()]

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
        diffs.append((out_cue - out_flax).abs().max().item())

    return max(diffs)


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
        diff = _compare_linear(
            irreps_in,
            irreps_out,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )
        assert diff <= self.tol, (
            f'max deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for Linear {irreps_in} → {irreps_out}'
        )
