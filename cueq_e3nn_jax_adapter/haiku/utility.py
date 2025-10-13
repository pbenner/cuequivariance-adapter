"""Haiku view of the shared tensor layout utilities."""

from __future__ import annotations

from .._utility import (
    _ensure_tuple_ints,
    collapse_ir_mul_segments,
    ir_mul_to_mul_ir,
    mul_ir_to_ir_mul,
)

__all__ = [
    'collapse_ir_mul_segments',
    'mul_ir_to_ir_mul',
    'ir_mul_to_mul_ir',
    '_ensure_tuple_ints',
]
