from typing import List, Optional

import cuequivariance as cue
import cuequivariance_torch as cuet
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3  # type: ignore
from flax.core import freeze, unfreeze
from mace.modules.irreps_tools import tp_out_irreps_with_instructions  # type: ignore
from mace.modules.wrapper_ops import (  # type: ignore
    CuEquivarianceConfig,
    OEQConfig,
)

from cuequivariance_adapter.flax.tensor_product import (
    TensorProduct as TensorProductFlax,
)

jax.config.update('jax_enable_x64', True)


def _build_flax_apply(
    irreps1_o3: o3.Irreps,
    irreps2_o3: o3.Irreps,
    target_o3: o3.Irreps,
    weight_numel: int,
    shared_weights: bool,
    internal_weights: bool,
    instructions: Optional[List[tuple[int, int, int, str, bool, float]]],
):
    module = TensorProductFlax(
        irreps1_o3,
        irreps2_o3,
        target_o3,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        instructions=instructions,
    )
    zeros1 = jnp.zeros((1, irreps1_o3.dim))
    zeros2 = jnp.zeros((1, irreps2_o3.dim))
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
        variables = params
        call_weights = weights
        if internal_weights:
            if weights is not None:
                weight_array = jnp.asarray(weights)
                if weight_array.ndim == 1:
                    weight_array = weight_array[jnp.newaxis, :]
                elif weight_array.ndim == 2:
                    if weight_array.shape[0] != 1:
                        raise ValueError(
                            'Internal weights expect a single shared weight vector'
                        )
                    weight_array = weight_array[:1]
                else:
                    raise ValueError('Internal weights must be rank 1 or 2')
                mutable = unfreeze(params)
                mutable['params']['weight'] = weight_array
                variables = freeze(mutable)
            call_weights = None
        else:
            if weights is None:
                raise ValueError('External weights must be provided')
            call_weights = jnp.asarray(weights)

        x1_array = jnp.asarray(x1)
        x2_array = jnp.asarray(x2)
        return module.apply(variables, x1_array, x2_array, call_weights)

    return apply_fn


class TensorProductCuet:
    """Wrapper around o3.TensorProduct/cuet.ChannelwiseTensorProduct/oeq.TensorProduct followed by a scatter sum"""

    def __new__(
        cls,
        irreps_in1: o3.Irreps,
        irreps_in2: o3.Irreps,
        irreps_out: o3.Irreps,
        instructions: Optional[List] = None,
        shared_weights: bool = False,
        internal_weights: bool = False,
        cueq_config: Optional[CuEquivarianceConfig] = None,
        oeq_config: Optional[OEQConfig] = None,
    ):
        cueq_config = CuEquivarianceConfig(enabled=True, optimize_channelwise=True)

        return cuet.ChannelWiseTensorProduct(
            cue.Irreps(cueq_config.group, irreps_in1),
            cue.Irreps(cueq_config.group, irreps_in2),
            cue.Irreps(cueq_config.group, irreps_out),
            layout=cueq_config.layout,
            shared_weights=shared_weights,
            internal_weights=internal_weights,
        )


def compare_once(
    irreps1: str,
    irreps2: str,
    irreps_target: str,
    *,
    shared_weights: bool,
    internal_weights: bool,
    batch: int = 8,
) -> float:
    if internal_weights and not shared_weights:
        raise ValueError('internal_weights=True requires shared_weights=True')
    irreps1_o3 = o3.Irreps(irreps1)
    irreps2_o3 = o3.Irreps(irreps2)
    target_o3, instructions = tp_out_irreps_with_instructions(
        irreps1_o3, irreps2_o3, o3.Irreps(irreps_target)
    )

    tp_e3nn = o3.TensorProduct(
        irreps1_o3,
        irreps2_o3,
        target_o3,
        instructions=instructions,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )
    tp_cue = TensorProductCuet(
        irreps1_o3,
        irreps2_o3,
        target_o3,
        instructions=instructions,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
    )

    assert isinstance(tp_cue, cuet.ChannelWiseTensorProduct), 'cuet path not selected'

    flax_apply = _build_flax_apply(
        irreps1_o3,
        irreps2_o3,
        target_o3,
        tp_e3nn.weight_numel,
        shared_weights=shared_weights,
        internal_weights=internal_weights,
        instructions=instructions,
    )

    torch.manual_seed(0)
    x1 = torch.randn(batch, irreps1_o3.dim)
    x2 = torch.randn(batch, irreps2_o3.dim)

    if internal_weights:
        base_weights = tp_e3nn.weight.detach().clone()
        weight_tensor = base_weights.view(1, -1)
        with torch.no_grad():
            if tp_cue.weight is None:
                raise RuntimeError('cuet module missing internal weight tensor')
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
    weights_jax = None
    if not internal_weights:
        weights_jax = jnp.asarray(weight_tensor.detach().cpu().numpy())

    if weights_arg_e3nn is None:
        out_e3nn = tp_e3nn(x1, x2)
    else:
        out_e3nn = tp_e3nn(x1, x2, weights_arg_e3nn)

    if weights_arg_cue is None:
        out_cue = tp_cue(x1, x2)
    else:
        out_cue = tp_cue(x1, x2, weights_arg_cue)

    out_flax = flax_apply(
        x1_jax,
        x2_jax,
        weight_tensor.detach().cpu().numpy() if internal_weights else weights_jax,
    )
    out_flax = torch.from_numpy(np.array(out_flax, copy=True))

    diff_cuet = (out_e3nn - out_cue).abs().max().item()
    diff_flax = (out_e3nn - out_flax).abs().max().item()
    diff_cross = (out_cue - out_flax).abs().max().item()

    return max(diff_cuet, diff_flax, diff_cross)


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


class TestFlaxTensorProduct:
    tol = 1e-12

    @pytest.mark.parametrize('shared_weights, internal_weights', WEIGHT_CONFIGS)
    @pytest.mark.parametrize('irreps1, irreps2, irreps_target', TENSOR_PRODUCT_CASES)
    def test_tensor_product_agreement(
        self, irreps1, irreps2, irreps_target, shared_weights, internal_weights
    ):
        diff = compare_once(
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
