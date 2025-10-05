"""Diagnostic script comparing e3nn and cue tensor-product shapes."""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
from e3nn import o3
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from mace.modules.wrapper_ops import CuEquivarianceConfig, TensorProduct


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

    tp_e3nn = TensorProduct(
        irreps_in1_o3,
        irreps_in2_o3,
        target_o3,
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
        cueq_config=None,
    )

    tp_cue = TensorProduct(
        o3.Irreps(irreps_spec),
        o3.Irreps(irreps_spec),
        o3.Irreps(str(requested_out_o3)),
        instructions=instructions,
        shared_weights=False,
        internal_weights=False,
        cueq_config=CuEquivarianceConfig(enabled=True, optimize_channelwise=True),
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
    print('Instructions (i1, i2, i_out, mode, train):', instructions)
    print('e3nn implementation:', type(tp_e3nn).__name__)
    print('cue implementation:', type(tp_cue).__name__)
    print('Batch size:', batch)
    print('e3nn output shape:', tuple(out_e3nn.shape))
    print('cue output shape:', tuple(out_cue.shape))

    if out_e3nn.shape != out_cue.shape:
        print(
            '\n⚠️  Shape mismatch detected! cue channels:',
            out_cue.shape[-1],
            'vs e3nn channels:',
            out_e3nn.shape[-1],
        )
    else:
        print('\n✅  Shapes match.')


if __name__ == '__main__':
    main()
