"""Diagnostic script comparing e3nn and cue tensor-product shapes."""

from __future__ import annotations

import cuequivariance as cue
import cuequivariance_torch as cuet
import torch
from e3nn import o3
from mace.modules.irreps_tools import tp_out_irreps_with_instructions
from mace.modules.wrapper_ops import CuEquivarianceConfig, TensorProduct
from mace.tools.scatter import scatter_sum


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

    # ------------------------------------------------------------------
    # Mimic the typical graph usage pattern: TensorProduct followed by scatter_sum.
    num_nodes = 6
    num_edges = batch
    node_feats = torch.randn(num_nodes, irreps_in1_o3.dim)
    edge_index = torch.randint(0, num_nodes, (2, num_edges))
    edge_attrs = torch.randn(num_edges, irreps_in2_o3.dim)
    edge_weights = torch.randn(num_edges, tp_e3nn.weight_numel)

    sender, receiver = edge_index

    mji_e3nn = tp_e3nn(node_feats[sender], edge_attrs, edge_weights)
    mji_cue = tp_cue(node_feats[sender], edge_attrs, edge_weights)

    message_e3nn = scatter_sum(mji_e3nn, receiver, dim=0, dim_size=num_nodes)
    message_cue = scatter_sum(mji_cue, receiver, dim=0, dim_size=num_nodes)

    print('\n--- Graph-style usage ---')
    print('edge_index (sender -> receiver):')
    print(edge_index)
    print('Aggregated e3nn output shape:', tuple(message_e3nn.shape))
    print('Aggregated cue output shape:', tuple(message_cue.shape))
    if message_e3nn.shape != message_cue.shape:
        print(
            '⚠️  Aggregated shapes still differ: cue channels',
            message_cue.shape[-1],
            'vs e3nn channels',
            message_e3nn.shape[-1],
        )
    else:
        diff = (message_e3nn - message_cue).abs().max().item()
        print('Max abs difference between aggregated outputs:', diff)


if __name__ == '__main__':
    main()
