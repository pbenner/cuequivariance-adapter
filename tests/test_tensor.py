from typing import List, Optional

import cuequivariance as cue
import cuequivariance_torch as cuet
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from mace.modules.wrapper_ops import (
    CuEquivarianceConfig,
    OEQConfig,
)

from cuequivariance_adapter.tensor_product import TensorProduct as TensorProductCuex

jax.config.update('jax_enable_x64', True)


def _build_cuex_apply(
    irreps1_o3: o3.Irreps,
    irreps2_o3: o3.Irreps,
    target_o3: o3.Irreps,
    weight_numel: int,
):
    """Create a pure function that evaluates TensorProductCuex with frozen params."""

    recorded: dict[str, int] = {}

    def forward(x1, x2, weights):
        module = TensorProductCuex(irreps1_o3, irreps2_o3, target_o3)
        recorded['numel'] = module.weight_numel
        return module(x1, x2, weights)

    transformed = hk.without_apply_rng(hk.transform(forward))
    zeros1 = jnp.zeros((1, irreps1_o3.dim), dtype=jnp.float64)
    zeros2 = jnp.zeros((1, irreps2_o3.dim), dtype=jnp.float64)
    zerosw = jnp.zeros((1, weight_numel), dtype=jnp.float64)
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

    def apply_fn(x1, x2, weights):
        return transformed.apply(params, x1, x2, weights)

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
            dtype=torch.get_default_dtype(),
            math_dtype=torch.get_default_dtype(),
        )


def compare_once(
    irreps1: str, irreps2: str, irreps_target: str, batch: int = 8
) -> float:
    """Return the max |difference| between e3nn and cuet tensor products."""
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
        shared_weights=False,
        internal_weights=False,
    )
    tp_cue = TensorProductCuet(
        irreps1_o3,
        irreps2_o3,
        target_o3,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
    )

    assert isinstance(tp_cue, cuet.ChannelWiseTensorProduct), 'cuet path not selected'

    cuex_apply = _build_cuex_apply(
        irreps1_o3, irreps2_o3, target_o3, tp_e3nn.weight_numel
    )

    torch.manual_seed(0)
    x1 = torch.randn(batch, irreps1_o3.dim)
    x2 = torch.randn(batch, irreps2_o3.dim)
    weights = torch.randn(batch, tp_e3nn.weight_numel)

    x1_jax = jnp.asarray(x1.detach().cpu().numpy())
    x2_jax = jnp.asarray(x2.detach().cpu().numpy())
    weights_jax = jnp.asarray(weights.detach().cpu().numpy())

    out_e3nn = tp_e3nn(x1, x2, weights)
    out_cue = tp_cue(x1, x2, weights)
    out_cuex = cuex_apply(x1_jax, x2_jax, weights_jax)
    out_cuex = torch.from_numpy(np.array(out_cuex, copy=True))

    diff_cuet = (out_e3nn - out_cue).abs().max().item()
    diff_cuex = (out_e3nn - out_cuex).abs().max().item()
    diff_cross = (out_cue - out_cuex).abs().max().item()

    return max(diff_cuet, diff_cuex, diff_cross)


TENSOR_PRODUCT_CASES = [
    ('2x0e + 1x1o', '1x0e + 1x1o', '3x0e + 3x1o + 1x2e'),
    ('3x1e', '1x0e + 1x1e + 1x2e', '3x0e + 6x1e + 3x2e'),
    ('1x2o + 2x1e', '1x0e + 1x1o', '3x1e + 3x2o + 1x3e'),
]


class TestTensorProduct:
    tol = 1e-12

    @pytest.fixture(autouse=True)
    def _set_default_dtype(self):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        yield
        torch.set_default_dtype(previous_dtype)

    @pytest.mark.parametrize(
        'irreps1, irreps2, irreps_target',
        TENSOR_PRODUCT_CASES,
    )
    def test_tensor_product_agreement(self, irreps1, irreps2, irreps_target):
        diff = compare_once(irreps1, irreps2, irreps_target)
        assert diff <= self.tol, (
            f'max deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for {irreps1} ⊗ {irreps2} → {irreps_target}'
        )
