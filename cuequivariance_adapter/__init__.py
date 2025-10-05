"""Cue-equivariant adapters with compatible torch bindings."""

from __future__ import annotations

from .fully_connected_tensor_product import FullyConnectedTensorProduct
from .linear import Linear
from .tensor_product import TensorProduct

__all__ = ['TensorProduct', 'FullyConnectedTensorProduct', 'Linear']


def _apply_channelwise_patch() -> None:
    """Ensure torch `ChannelWiseTensorProduct` matches expected layouts."""

    try:
        import cuequivariance as cue
        import cuequivariance_torch as cuet
    except ImportError:  # pragma: no cover - optional torch backend
        return

    from .torch_utils import (
        torch_collapse_ir_mul_segments,
        torch_ir_mul_to_mul_ir,
        torch_mul_ir_to_ir_mul,
    )

    if getattr(cuet.ChannelWiseTensorProduct, '_cue_adapter_patched', False):
        return

    original_init = cuet.ChannelWiseTensorProduct.__init__
    original_forward = cuet.ChannelWiseTensorProduct.forward

    def patched_init(self, irreps_in1, irreps_in2, filter_irreps_out=None, **kwargs):
        descriptor = cue.descriptors.channelwise_tensor_product(
            irreps_in1, irreps_in2, filter_irreps_out
        )
        descriptor_irreps = descriptor.outputs[0].irreps
        segment_shapes = tuple(descriptor.polynomial.operands[-1].segments)

        if filter_irreps_out is None:
            target_irreps = descriptor_irreps
        elif isinstance(filter_irreps_out, cue.Irreps):
            target_irreps = filter_irreps_out
        else:
            target_irreps = cue.Irreps(irreps_in1.irrep_class, filter_irreps_out)

        original_init(
            self,
            irreps_in1,
            irreps_in2,
            filter_irreps_out=filter_irreps_out,
            **kwargs,
        )

        self._cue_descriptor_irreps = descriptor_irreps
        self._cue_target_irreps = target_irreps
        self._cue_segment_shapes = segment_shapes

    def patched_forward(self, *args, **kwargs):
        output = original_forward(self, *args, **kwargs)

        descriptor_irreps = getattr(self, '_cue_descriptor_irreps', None)
        target_irreps = getattr(self, '_cue_target_irreps', None)
        segment_shapes = getattr(self, '_cue_segment_shapes', None)

        if (
            descriptor_irreps is None
            or target_irreps is None
            or segment_shapes is None
            or descriptor_irreps.dim == target_irreps.dim
        ):
            return output

        ir_mul = torch_mul_ir_to_ir_mul(output, descriptor_irreps)
        ir_mul = torch_collapse_ir_mul_segments(
            ir_mul, descriptor_irreps, target_irreps, segment_shapes
        )
        mul_ir = torch_ir_mul_to_mul_ir(ir_mul, target_irreps)
        return mul_ir

    cuet.ChannelWiseTensorProduct.__init__ = patched_init
    cuet.ChannelWiseTensorProduct.forward = patched_forward
    cuet.ChannelWiseTensorProduct._cue_adapter_patched = True


_apply_channelwise_patch()
