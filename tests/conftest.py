"""Pytest configuration to prepare torch bindings."""

from __future__ import annotations

import importlib
import sys
import types

import warnings


def _ensure_mosaic_profiler_stub() -> None:
    """Provide a stub mosaic profiler when running on systems without GPU support."""

    try:
        profiler_mod = importlib.import_module('jax.experimental.mosaic.gpu.profiler')
    except ImportError:
        profiler_mod = None
    else:
        if not hasattr(profiler_mod, '_event_record'):
            def _event_record(state, copy_before=True):
                return None, state

            profiler_mod._event_record = _event_record  # type: ignore[attr-defined]
        if not hasattr(profiler_mod, '_event_elapsed'):
            def _event_elapsed(start_event, end_event):
                return 0.0

            profiler_mod._event_elapsed = _event_elapsed  # type: ignore[attr-defined]
        return

    profiler_mod = types.ModuleType('jax.experimental.mosaic.gpu.profiler')

    def _event_record(state, copy_before=True):
        return None, state

    def _event_elapsed(start_event, end_event):
        return 0.0

    profiler_mod._event_record = _event_record  # type: ignore[attr-defined]
    profiler_mod._event_elapsed = _event_elapsed  # type: ignore[attr-defined]

    gpu_mod = types.ModuleType('jax.experimental.mosaic.gpu')
    gpu_mod.profiler = profiler_mod  # type: ignore[attr-defined]

    experimental_mod = sys.modules.get('jax.experimental')
    if experimental_mod is None:
        experimental_mod = types.ModuleType('jax.experimental')
        sys.modules['jax.experimental'] = experimental_mod

    mosaic_mod = getattr(experimental_mod, 'mosaic', None)
    if mosaic_mod is None:
        mosaic_mod = types.ModuleType('jax.experimental.mosaic')
        experimental_mod.mosaic = mosaic_mod  # type: ignore[attr-defined]

    mosaic_mod.gpu = gpu_mod  # type: ignore[attr-defined]

    sys.modules['jax.experimental.mosaic'] = mosaic_mod
    sys.modules['jax.experimental.mosaic.gpu'] = gpu_mod
    sys.modules['jax.experimental.mosaic.gpu.profiler'] = profiler_mod


_ensure_mosaic_profiler_stub()

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
from . import _torch_patch  # type: ignore  # noqa: F401
import torch
from jax import config as jax_config

# Register safe globals for torch before imports in test files
torch.serialization.add_safe_globals([slice])

# Set default dtype for JAX and Torch
jax_config.update('jax_enable_x64', True)
torch.set_default_dtype(torch.float64)


def pytest_configure(config) -> None:  # type: ignore[override]
    _apply_warning_filters()
