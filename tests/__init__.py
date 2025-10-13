"""Test utilities package for cueq-e3nn-jax adapter."""

import pytest

pytestmark = [
    pytest.mark.filterwarnings('ignore::DeprecationWarning:haiku._src.transform'),
    pytest.mark.filterwarnings('ignore::DeprecationWarning:jax.lib.xla_extension'),
]
