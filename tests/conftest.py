"""Pytest configuration to prepare torch bindings."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the tests directory is importable whether pytest runs from repo root or tests/
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

import warnings

_WARNING_FILTERS: list[dict[str, object]] = [
    {'category': DeprecationWarning, 'module': 'haiku'},
    {'category': DeprecationWarning, 'module': r'haiku\._src\.transform'},
    {'category': DeprecationWarning, 'message': r'.*PjitFunction is deprecated.*'},
    {'category': DeprecationWarning, 'message': r'.*PmapFunction is deprecated.*'},
    {'category': DeprecationWarning, 'module': r'jax\.lib\.xla_extension'},
    {
        'category': UserWarning,
        'module': r'cuequivariance\.group_theory\.irreps_array\.misc_ui',
        'message': r'.*layout is not specified.*',
    },
    {
        'category': UserWarning,
        'module': r'cuequivariance_torch\.primitives\.segmented_polynomial',
        'message': r'.*cuequivariance_ops_torch is not available.*',
    },
    {
        'category': UserWarning,
        'module': r'cuequivariance_torch\.primitives\.segmented_polynomial',
        'message': r'.*Fused TP is not supported on CPU.*',
    },
    {
        'category': UserWarning,
        'module': r'cuequivariance_torch\.operations\.tp_channel_wise',
        'message': r'.*Segments are not the same shape.*',
    },
    {
        'category': UserWarning,
        'module': r'cuequivariance_torch\..*',
        'message': r'.*Falling back to naive implementation.*',
    },
    {
        'category': UserWarning,
        'module': r'torch.cuda',
        'message': r'.*CUDA initialization: Unexpected error from cudaGetDeviceCount.*',
    },
]


def _apply_warning_filters() -> None:
    for kwargs in _WARNING_FILTERS:
        warnings.filterwarnings('ignore', **kwargs)


_apply_warning_filters()

# Importing applies the ChannelWiseTensorProduct patch for cuequivariance_torch.
import _torch_patch  # type: ignore  # noqa: F401
import torch
from jax import config as jax_config

# Register safe globals for torch before imports in test files
torch.serialization.add_safe_globals([slice])

# Set default dtype for JAX and Torch
jax_config.update('jax_enable_x64', True)
torch.set_default_dtype(torch.float64)


def pytest_configure(config) -> None:  # type: ignore[override]
    _apply_warning_filters()
