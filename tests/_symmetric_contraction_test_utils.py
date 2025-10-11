"""Shared helpers for symmetric contraction adapter tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import torch
from e3nn import o3  # type: ignore
from e3nn_jax import Irreps  # type: ignore
from mace.modules.wrapper_ops import (  # type: ignore
    CuEquivarianceConfig,
    SymmetricContractionWrapper,
)


@dataclass
class SymmetricAdapterHandle:
    weight_shape: tuple[int, int, int]
    set_weights: Callable[[np.ndarray], None]
    apply: Callable[[np.ndarray, np.ndarray], np.ndarray]


AdapterBuilder = Callable[
    [o3.Irreps, o3.Irreps, int, int, bool],
    SymmetricAdapterHandle,
]


@dataclass
class SymmetricComparisonResult:
    diff: float
    adapter_output: np.ndarray
    cue_output: np.ndarray
    features: np.ndarray
    selector: np.ndarray
    use_per_node_mix: bool

    @property
    def max_diff(self) -> float:
        return self.diff


def run_symmetric_contraction_comparison(
    build_adapter: AdapterBuilder,
    irreps_in: str,
    irreps_out: str,
    *,
    correlation: int,
    num_elements: int,
    use_reduced_cg: bool,
    batch: int = 5,
    seed: int = 0,
    use_per_node_mix: bool = False,
) -> SymmetricComparisonResult:
    """Return comparison data between the adapter and cuet reference."""

    irreps_in_o3 = o3.Irreps(irreps_in)
    irreps_out_o3 = o3.Irreps(irreps_out)

    adapter = build_adapter(
        irreps_in_o3,
        irreps_out_o3,
        correlation,
        num_elements,
        use_reduced_cg,
    )

    cueq_config = CuEquivarianceConfig(enabled=True, optimize_symmetric=True)
    cue_torch = SymmetricContractionWrapper(
        irreps_in_o3,
        irreps_out_o3,
        correlation=correlation,
        num_elements=num_elements,
        cueq_config=cueq_config,
        use_reduced_cg=use_reduced_cg,
    ).double()

    cue_weights = cue_torch.weight.detach().cpu().numpy().astype(np.float64, copy=False)
    if cue_weights.shape != adapter.weight_shape:
        raise AssertionError(
            f'cue weight shape {cue_weights.shape} does not match adapter shape {adapter.weight_shape}'
        )
    adapter.set_weights(cue_weights)

    rng = np.random.default_rng(seed)
    mul = irreps_in_o3[0].mul
    feature_width = sum(ir.dim for _, ir in irreps_in_o3)
    x_features = rng.standard_normal((batch, mul, feature_width)).astype(np.float64)

    if use_per_node_mix:
        selector = rng.standard_normal((batch, num_elements)).astype(np.float64)
        adapter_selector = selector
    else:
        indices = rng.integers(0, num_elements, size=(batch,), dtype=np.int32)
        selector = np.eye(num_elements, dtype=np.float64)[indices]
        adapter_selector = indices

    out_adapter = adapter.apply(x_features, adapter_selector)

    x_torch = torch.from_numpy(x_features).to(dtype=torch.double)
    x_flat = x_torch.transpose(1, 2).reshape(x_torch.shape[0], -1)
    mix_torch = torch.from_numpy(selector).to(dtype=torch.double)

    out_cuet_flat = torch.zeros(
        (batch, irreps_out_o3.dim), dtype=torch.double, device=x_flat.device
    )
    for element_idx in range(num_elements):
        elem_indices = torch.full(
            (batch,), element_idx, dtype=torch.int32, device=x_flat.device
        )
        partial = cue_torch(x_flat, elem_indices)
        out_cuet_flat = (
            out_cuet_flat + mix_torch[:, element_idx : element_idx + 1] * partial
        )

    out_cue = out_cuet_flat.detach().cpu().numpy()
    diff = float(np.max(np.abs(out_adapter - out_cue)))
    return SymmetricComparisonResult(
        diff=diff,
        adapter_output=out_adapter,
        cue_output=out_cue,
        features=x_features,
        selector=selector,
        use_per_node_mix=use_per_node_mix,
    )


def build_flax_symmetric_adapter(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    correlation: int,
    num_elements: int,
    use_reduced_cg: bool,
) -> SymmetricAdapterHandle:
    from flax.core import freeze, unfreeze  # type: ignore

    from cuequivariance_adapter.flax.symmetric_contraction import SymmetricContraction

    module = SymmetricContraction(
        Irreps(irreps_in),
        Irreps(irreps_out),
        correlation=correlation,
        num_elements=num_elements,
        use_reduced_cg=use_reduced_cg,
    )

    mul = irreps_in[0].mul
    feature_width = sum(ir.dim for _, ir in irreps_in)
    zeros_x = jnp.zeros((1, mul, feature_width), dtype=jnp.float64)
    zeros_idx = jnp.zeros((1,), dtype=jnp.int32)
    params = module.init(jax.random.PRNGKey(0), zeros_x, zeros_idx)
    weight_shape = module.apply(params, method=lambda m: m.weight_param_shape)

    def set_weights(weights: np.ndarray) -> None:
        nonlocal params
        mutable = unfreeze(params)
        mutable['params']['weight'] = jnp.asarray(weights, dtype=mutable['params']['weight'].dtype)
        params = freeze(mutable)

    def apply(x_features: np.ndarray, selector: np.ndarray) -> np.ndarray:
        x_array = jnp.asarray(x_features, dtype=jnp.float64)
        if selector.dtype == np.int32 or selector.dtype == np.int64:
            selector_array = jnp.asarray(selector, dtype=jnp.int32)
        else:
            selector_array = jnp.asarray(selector, dtype=jnp.float64)
        return np.asarray(module.apply(params, x_array, selector_array))

    return SymmetricAdapterHandle(
        weight_shape=weight_shape,
        set_weights=set_weights,
        apply=apply,
    )


def build_haiku_symmetric_adapter(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    correlation: int,
    num_elements: int,
    use_reduced_cg: bool,
) -> SymmetricAdapterHandle:
    from cuequivariance_adapter.haiku.symmetric_contraction import SymmetricContraction

    info: dict[str, object] = {}

    def forward(x, indices):
        module = SymmetricContraction(
            Irreps(irreps_in),
            Irreps(irreps_out),
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
            name='symmetric_contraction',
        )
        info['weight_shape'] = module.weight_param_shape
        info['scope'] = module.module_name
        return module(x, indices)

    transformed = hk.without_apply_rng(hk.transform(forward))
    mul = irreps_in[0].mul
    feature_width = sum(ir.dim for _, ir in irreps_in)
    zeros_x = jnp.zeros((1, mul, feature_width), dtype=jnp.float64)
    zeros_idx = jnp.zeros((1,), dtype=jnp.int32)
    params = transformed.init(jax.random.PRNGKey(0), zeros_x, zeros_idx)

    weight_shape = info['weight_shape']
    scope = info['scope']

    def set_weights(weights: np.ndarray) -> None:
        nonlocal params
        mutable = hk.data_structures.to_mutable_dict(params)
        mutable[scope]['weight'] = jnp.asarray(
            weights, dtype=mutable[scope]['weight'].dtype
        )
        params = hk.data_structures.to_immutable_dict(mutable)

    def apply(x_features: np.ndarray, selector: np.ndarray) -> np.ndarray:
        x_array = jnp.asarray(x_features, dtype=jnp.float64)
        if selector.dtype == np.int32 or selector.dtype == np.int64:
            selector_array = jnp.asarray(selector, dtype=jnp.int32)
        else:
            selector_array = jnp.asarray(selector, dtype=jnp.float64)
        return np.asarray(transformed.apply(params, x_array, selector_array))

    return SymmetricAdapterHandle(
        weight_shape=weight_shape,
        set_weights=set_weights,
        apply=apply,
    )
