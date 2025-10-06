import haiku as hk
import jax
import jax.numpy as jnp
import pytest
from e3nn_jax import Irreps

from cuequivariance_adapter.symmetric_contraction import SymmetricContraction


def _forward_module(irreps_in: str, irreps_out: str, correlation: int, num_elements: int):
    def forward(x, idx):
        module = SymmetricContraction(
            Irreps(irreps_in),
            Irreps(irreps_out),
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=True,
        )
        return module(x, idx)

    return hk.without_apply_rng(hk.transform(forward))


class TestSymmetricContraction:
    @pytest.mark.parametrize(
        'irreps_in, irreps_out, correlation, num_elements',
        [
            ('2x0e + 2x1o', '2x0e + 2x1o', 2, 3),
            ('2x0e + 2x1o', '2x0e', 2, 5),
            ('4x0e + 4x1o', '4x0e + 4x1o', 2, 6),
            ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 2, 10),
        ],
    )
    def test_forward_shape_matches_irreps(
        self,
        irreps_in: str,
        irreps_out: str,
        correlation: int,
        num_elements: int,
    ) -> None:
        transformed = _forward_module(irreps_in, irreps_out, correlation, num_elements)

        irreps_in_obj = Irreps(irreps_in)
        mul = irreps_in_obj[0].mul
        feature_dim = sum(ir.dim for _, ir in irreps_in_obj)
        batch = 2

        params = transformed.init(
            jax.random.PRNGKey(0),
            jnp.zeros((batch, mul, feature_dim)),
            jnp.zeros((batch,), dtype=jnp.int32),
        )

        out = transformed.apply(
            params,
            jnp.ones((batch, mul, feature_dim)),
            jnp.zeros((batch,), dtype=jnp.int32),
        )

        expected_dim = Irreps(irreps_out).dim
        assert out.shape == (batch, expected_dim)

    def test_invalid_correlation_raises_value_error(self) -> None:
        with pytest.raises(ValueError):
            SymmetricContraction(
                Irreps('1x0e'),
                Irreps('1x0e'),
                correlation=0,
                num_elements=1,
            )

    def test_use_reduced_cg_false_not_supported(self) -> None:
        irreps = Irreps('1x0e + 1x1o')
        mul = irreps[0].mul
        feature_dim = sum(ir.dim for _, ir in irreps)

        def forward(x, idx):
            module = SymmetricContraction(
                Irreps('1x0e + 1x1o'),
                Irreps('1x0e + 1x1o'),
                correlation=2,
                num_elements=1,
                use_reduced_cg=False,
            )
            return module(x, idx)

        transformed = hk.without_apply_rng(hk.transform(forward))

        with pytest.raises(NotImplementedError):
            transformed.init(
                jax.random.PRNGKey(0),
                jnp.zeros((1, mul, feature_dim)),
                jnp.zeros((1,), dtype=jnp.int32),
            )
