"""Pytest configuration to prepare torch bindings."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the tests directory is importable whether pytest runs from repo root or tests/
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Importing applies the ChannelWiseTensorProduct patch for cuequivariance_torch.
import warnings

import _torch_patch  # type: ignore  # noqa: F401
import torch
from jax import config as jax_config

# Apply warning filters globally before any tests run
warnings.filterwarnings('ignore', category=DeprecationWarning, module='haiku')
warnings.filterwarnings(
    'ignore',
    message='cuequivariance_ops_torch is not available',
    module='cuequivariance_torch'
)
warnings.filterwarnings(
    'ignore',
    message='Fused TP is not supported on CPU',
    module='cuequivariance_torch'
)
warnings.filterwarnings(
    'ignore',
    message='layout is not specified, defaulting to cue.mul_ir',
    module='cuequivariance'
)
warnings.filterwarnings(
    'ignore',
    message='CUDA initialization: Unexpected error from cudaGetDeviceCount',
    module='torch.cuda'
)

# Register safe globals for torch before imports in test files
torch.serialization.add_safe_globals([slice])

# Set default dtype for JAX and Torch
jax_config.update('jax_enable_x64', True)
torch.set_default_dtype(torch.float64)
