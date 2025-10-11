"""Framework-agnostic cores for Flax and Haiku adapters."""

from .fully_connected_tensor_product import FullyConnectedTensorProductCore
from .linear import LinearCore
from .symmetric_contraction import SymmetricContractionCore
from .tensor_product import TensorProductCore

__all__ = [
    'FullyConnectedTensorProductCore',
    'LinearCore',
    'SymmetricContractionCore',
    'TensorProductCore',
]
