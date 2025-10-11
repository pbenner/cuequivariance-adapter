"""Test utilities package for cuequivariance adapter."""

import pytest

pytestmark = [
    pytest.mark.filterwarnings('ignore::DeprecationWarning:haiku._src.transform'),
    pytest.mark.filterwarnings('ignore::DeprecationWarning:jax.lib.xla_extension'),
]
