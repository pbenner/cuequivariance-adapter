import jax
import pytest

from tests._symmetric_contraction_test_utils import (
    build_flax_symmetric_adapter,
    run_symmetric_contraction_comparison,
)

jax.config.update('jax_enable_x64', True)


SYMMETRIC_CASES = [
    ('2x0e + 2x1o', '2x0e + 2x1o', 2, 3),
    ('1x0e + 1x1o', '1x0e + 1x1o', 3, 4),
    ('2x0e + 2x1o', '2x0e', 2, 5),
    ('4x0e + 4x1o', '4x0e + 4x1o', 3, 6),
    ('4x0e + 4x1o', '4x0e', 3, 6),
    ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 2, 10),
    ('128x0e+128x1o+128x2e+128x3o', '128x0e+128x1o+128x2e', 3, 10),
]


class TestFlaxSymmetricContraction:
    tol = 1e-11

    @pytest.mark.parametrize('use_reduced_cg', [True, False])
    @pytest.mark.parametrize(
        'irreps_in, irreps_out, correlation, num_elements', SYMMETRIC_CASES
    )
    def test_matches_cuet_symmetric_contraction(
        self,
        irreps_in,
        irreps_out,
        correlation,
        num_elements,
        use_reduced_cg,
    ):
        result = run_symmetric_contraction_comparison(
            build_flax_symmetric_adapter,
            irreps_in,
            irreps_out,
            correlation=correlation,
            num_elements=num_elements,
            use_reduced_cg=use_reduced_cg,
        )
        diff = result.max_diff
        assert diff <= self.tol, (
            f'cue comparison deviation {diff:.3e} exceeds tolerance {self.tol} '
            f'for correlation={correlation} use_reduced_cg={use_reduced_cg}'
        )
