"""Diagnostic script comparing e3nn and cue tensor-product shapes."""

from __future__ import annotations

import torch
from e3nn import o3

import cuequivariance as cue
import cuequivariance_torch as cuet

from mace.modules.irreps_tools import tp_out_irreps_with_instructions


def main() -> None:
    batch = 8
    irreps_spec = '2x0e'

    irreps_in1_o3 = o3.Irreps(irreps_spec)
    irreps_in2_o3 = o3.Irreps(irreps_spec)
    requested_out_o3 = o3.Irreps(irreps_spec)

    target_o3, instructions = tp_out_irreps_with_instructions(
        irreps_in1_o3,
        irreps_in2_o3,
        requested_out_o3,
    )

    tp_e3nn = o3.TensorProduct(
        irreps_in1_o3,
        irreps_in2_o3,
        target_o3,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
    )

    irreps_in1_cue = cue.Irreps(cue.O3, irreps_spec)
    irreps_in2_cue = cue.Irreps(cue.O3, irreps_spec)
    irreps_out_cue = cue.Irreps(cue.O3, str(requested_out_o3))

    descriptor = cue.descriptors.channelwise_tensor_product(
        irreps_in1_cue,
        irreps_in2_cue,
        irreps_out_cue,
    )
    tp_cue = cuet.ChannelWiseTensorProduct(
        irreps_in1_cue,
        irreps_in2_cue,
        irreps_out_cue,
        shared_weights=False,
        internal_weights=False,
    )

    torch.manual_seed(0)
    x1 = torch.randn(batch, irreps_in1_o3.dim)
    x2 = torch.randn(batch, irreps_in2_o3.dim)
    weights = torch.randn(batch, tp_e3nn.weight_numel)

    out_e3nn = tp_e3nn(x1, x2, weights)
    out_cue = tp_cue(x1, x2, weights)

    print('Input irreps:', irreps_spec)
    print('Requested output irreps:', requested_out_o3)
    print('Target irreps from e3nn:', target_o3)
    print('ChannelWise descriptor output irreps:', descriptor.outputs[0].irreps)
    print('Instructions (i1, i2, i_out, mode, train):', instructions)
    print('Batch size:', batch)
    print('e3nn output shape:', tuple(out_e3nn.shape))
    print('cue output shape:', tuple(out_cue.shape))

    if out_e3nn.shape != out_cue.shape:
        print('\n⚠️  Shape mismatch detected! cue channels:', out_cue.shape[-1],
              'vs e3nn channels:', out_e3nn.shape[-1])
    else:
        print('\n✅  Shapes match.')


if __name__ == '__main__':
    main()
