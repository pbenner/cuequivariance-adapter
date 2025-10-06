"""Test-only torch helpers for layout conversions matching e3nn conventions."""

from __future__ import annotations

from typing import Iterable, Tuple

import cuequivariance as cue
import torch

IrrepsLike = cue.Irreps


def torch_mul_ir_to_ir_mul(array: torch.Tensor, irreps: IrrepsLike) -> torch.Tensor:
    """Reorder the last axis of ``array`` from mul_ir to ir_mul layout."""

    if irreps.dim == 0:
        return array

    leading_shape = array.shape[:-1]
    array = array.reshape(*leading_shape, irreps.dim)
    segments: list[torch.Tensor] = []
    offset = 0
    for mul_ir in irreps:
        mul = mul_ir.mul
        ir_dim = mul_ir.ir.dim
        block = array[..., offset : offset + mul * ir_dim]
        offset += mul * ir_dim
        block = block.reshape(*leading_shape, mul, ir_dim)
        block = block.transpose(-1, -2)  # -> (..., ir_dim, mul)
        block = block.reshape(*leading_shape, ir_dim * mul)
        segments.append(block)
    return torch.cat(segments, dim=-1) if segments else array


def torch_ir_mul_to_mul_ir(array: torch.Tensor, irreps: IrrepsLike) -> torch.Tensor:
    """Reorder the last axis of ``array`` from ir_mul to mul_ir layout."""

    if irreps.dim == 0:
        return array

    leading_shape = array.shape[:-1]
    array = array.reshape(*leading_shape, irreps.dim)
    segments: list[torch.Tensor] = []
    offset = 0
    for mul_ir in irreps:
        mul = mul_ir.mul
        ir_dim = mul_ir.ir.dim
        block = array[..., offset : offset + mul * ir_dim]
        offset += mul * ir_dim
        block = block.reshape(*leading_shape, ir_dim, mul)
        block = block.transpose(-1, -2)  # -> (..., mul, ir_dim)
        block = block.reshape(*leading_shape, mul * ir_dim)
        segments.append(block)
    return torch.cat(segments, dim=-1) if segments else array


def torch_collapse_ir_mul_segments(
    array: torch.Tensor,
    descriptor_irreps: IrrepsLike,
    target_irreps: IrrepsLike,
    segment_shapes: Iterable[Tuple[int, ...]],
) -> torch.Tensor:
    """Collapse redundant multiplicities in ``array`` to match ``target_irreps``."""

    if descriptor_irreps == target_irreps:
        return array

    leading_shape = array.shape[:-1]
    flat = array.reshape(*leading_shape, descriptor_irreps.dim)

    descriptor_list = list(descriptor_irreps)
    target_list = list(target_irreps)
    shapes_list = list(segment_shapes)

    if len(descriptor_list) != len(shapes_list):
        raise ValueError(
            'Descriptor irreps and segment shapes length mismatch: '
            f'{len(descriptor_list)} vs {len(shapes_list)}'
        )
    if len(descriptor_list) != len(target_list):
        raise ValueError(
            'Descriptor and target irreps length mismatch: '
            f'{len(descriptor_list)} vs {len(target_list)}'
        )

    blocks: list[torch.Tensor] = []
    offset = 0

    for idx, (desc_entry, target_entry, seg_shape) in enumerate(
        zip(descriptor_list, target_list, shapes_list)
    ):
        desc_mul, desc_ir = desc_entry
        target_mul, target_ir = target_entry

        if desc_ir != target_ir:
            raise ValueError(
                f'Descriptor/target irreps mismatch at index {idx}: '
                f'{desc_ir} vs {target_ir}'
            )

        ir_dim = desc_ir.dim
        block_size = ir_dim * desc_mul
        block = flat[..., offset : offset + block_size]
        offset += block_size

        if len(seg_shape) != 3:
            raise ValueError(
                f'Expected segment shape (ir, mul1, mul2) at index {idx}, got {seg_shape}'
            )
        ir_axis, mul1, mul2 = seg_shape
        if ir_axis != ir_dim:
            raise ValueError(
                f'Irrep dimension mismatch at index {idx}: segment {ir_axis}, expected {ir_dim}'
            )

        block = block.reshape(*leading_shape, ir_dim, mul1, mul2)

        if desc_mul == target_mul:
            if mul1 * mul2 != target_mul:
                raise ValueError(
                    f'Unexpected multiplicity {mul1 * mul2} at index {idx}, expected {target_mul}'
                )
            block_ir_mul = block.reshape(*leading_shape, ir_dim, target_mul)
        elif desc_mul == target_mul * mul2:
            block_ir_mul = block.sum(dim=-1) / (float(mul2) ** 0.5)
            if block_ir_mul.shape[-1] != target_mul:
                raise ValueError(
                    f'Failed collapsing v-axis at index {idx}: '
                    f'got {block_ir_mul.shape[-1]}, expected {target_mul}'
                )
        elif desc_mul == target_mul * mul1:
            block_ir_mul = block.sum(dim=-2) / (float(mul1) ** 0.5)
            if block_ir_mul.shape[-1] != target_mul:
                raise ValueError(
                    f'Failed collapsing u-axis at index {idx}: '
                    f'got {block_ir_mul.shape[-1]}, expected {target_mul}'
                )
        else:
            raise ValueError(
                'Cannot map descriptor multiplicity to target multiplicity '
                f'at index {idx}: desc_mul={desc_mul}, target_mul={target_mul}, '
                f'segment_shape={seg_shape}'
            )

        blocks.append(block_ir_mul.reshape(*leading_shape, ir_dim * target_mul))

    if offset != flat.shape[-1]:
        raise ValueError(
            'Processed elements do not cover descriptor output '
            f'(covered {offset}, total {flat.shape[-1]})'
        )

    return torch.cat(blocks, dim=-1) if blocks else flat[..., :0]
