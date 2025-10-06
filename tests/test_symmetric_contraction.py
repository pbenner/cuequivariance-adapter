import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
from e3nn import o3  # type: ignore
from e3nn_jax import Irreps  # type: ignore
from mace.modules.symmetric_contraction import (  # type: ignore
    SymmetricContraction as MaceSymmetricContraction,
)

from cuequivariance_adapter.symmetric_contraction import SymmetricContraction

jax.config.update('jax_enable_x64', True)


def _init_adapter_module(
    irreps_in: o3.Irreps,
    irreps_out: o3.Irreps,
    *,
    correlation: int,
    num_elements: int,
) -> tuple[hk.Transformed, hk.Params, dict[str, object]]:
    info: dict[str, object] = {}

    mul = irreps_in[0].mul
    feature_width = sum(ir.dim for _, ir in irreps_in)

    def forward(x, indices):
        module = SymmetricContraction(
            Irreps(irreps_in),
            Irreps(irreps_out),
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=False,
            name='symmetric_contraction',
        )
        info['weight_shape'] = module.weight_param_shape
        info['scope'] = module.module_name
        return module(x, indices)

    transformed = hk.without_apply_rng(hk.transform(forward))
    zeros_x = jnp.zeros((1, mul, feature_width), dtype=jnp.float64)
    zeros_idx = jnp.zeros((1,), dtype=jnp.int32)
    params = transformed.init(jax.random.PRNGKey(0), zeros_x, zeros_idx)
    return transformed, params, info


def _collect_mace_torch_basis(
    mace_module: MaceSymmetricContraction,
) -> np.ndarray:
    """Flatten per-block MACE weights into adapter layout.

    MACE splits the learnable parameters across one ``Contraction`` object per
    output irrep. Each contraction stores ``weights_max`` (highest-order term)
    and additional tensors in ``contraction.weights`` for the lower degrees.
    For parity we reshape each of those tensors to
    ``(num_elements, params_per_degree, num_features)`` and concatenate along
    the parameter axis so the combined array matches the single Haiku parameter
    ``(num_elements, total_params, num_features)`` used by the adapter.
    """
    parts: list[np.ndarray] = []
    for contraction in mace_module.contractions:
        tensors = [contraction.weights_max, *contraction.weights]
        for tensor in tensors:
            if tensor.numel() == 0:
                continue
            parts.append(tensor.detach().cpu().numpy().astype(np.float64, copy=False))

    if not parts:
        num_elements = mace_module.contractions[0].weights_max.shape[0]
        num_features = mace_module.contractions[0].weights_max.shape[-1]
        return np.zeros((num_elements, 0, num_features), dtype=np.float64)

    return np.concatenate(parts, axis=1)


def _compare_to_mace(
    irreps_in: str,
    irreps_out: str,
    *,
    correlation: int,
    num_elements: int,
    batch: int = 5,
    seed: int = 0,
) -> float:
    irreps_in_o3 = o3.Irreps(irreps_in)
    irreps_out_o3 = o3.Irreps(irreps_out)

    adapter_tf, adapter_params, adapter_info = _init_adapter_module(
        irreps_in_o3,
        irreps_out_o3,
        correlation=correlation,
        num_elements=num_elements,
    )

    mace_torch = MaceSymmetricContraction(
        irreps_in_o3,
        irreps_out_o3,
        correlation=correlation,
        num_elements=num_elements,
        use_reduced_cg=False,
    ).double()

    mace_basis = _collect_mace_torch_basis(mace_torch)
    weight_shape = adapter_info['weight_shape']
    if mace_basis.shape != weight_shape:
        raise AssertionError(
            f'MACE weight shape {mace_basis.shape} does not match adapter shape {weight_shape}'
        )

    adapter_scope = adapter_info['scope']
    dtype = adapter_params[adapter_scope]['weight'].dtype
    adapter_mutable = hk.data_structures.to_mutable_dict(adapter_params)
    adapter_mutable[adapter_scope]['weight'] = jnp.asarray(mace_basis, dtype=dtype)
    adapter_params = hk.data_structures.to_immutable_dict(adapter_mutable)

    rng = np.random.default_rng(seed)
    mul = irreps_in_o3[0].mul
    feature_width = sum(ir.dim for _, ir in irreps_in_o3)
    x_features = rng.standard_normal((batch, mul, feature_width)).astype(np.float64)
    indices = rng.integers(0, num_elements, size=(batch,), dtype=np.int32)
    y_one_hot = np.eye(num_elements, dtype=np.float64)[indices]

    out_adapter = adapter_tf.apply(
        adapter_params,
        jnp.asarray(x_features, dtype=dtype),
        jnp.asarray(indices),
    )

    x_torch = torch.from_numpy(x_features).to(dtype=torch.double)
    y_torch = torch.from_numpy(y_one_hot).to(dtype=torch.double)
    out_mace = mace_torch(x_torch, y_torch).detach().cpu().numpy()

    return float(np.max(np.abs(np.asarray(out_adapter) - out_mace)))


SYMMETRIC_CASES = [
    ('2x0e + 2x1o', '2x0e + 2x1o', 2, 3),
    ('1x0e + 1x1o', '1x0e + 1x1o', 3, 4),
    ('2x0e + 2x1o', '2x0e', 2, 5),
    ('4x0e + 4x1o', '4x0e + 4x1o', 3, 6),
    ('4x0e + 4x1o', '4x0e', 3, 6),
]


class TestSymmetricContraction:
    tol = 1e-11

    @pytest.mark.parametrize(
        'irreps_in, irreps_out, correlation, num_elements', SYMMETRIC_CASES
    )
    def test_matches_mace_symmetric_contraction(
        self,
        irreps_in,
        irreps_out,
        correlation,
        num_elements,
    ):
        diff = _compare_to_mace(
            irreps_in,
            irreps_out,
            correlation=correlation,
            num_elements=num_elements,
        )
        assert diff <= self.tol, (
            f'MACE comparison deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for correlation={correlation}'
        )
