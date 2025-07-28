import e3nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet
import torch
from torch.nn import Linear, init
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data
import math
from typing import Optional
from .common import MCST, Gate, MyScatter
from .EdgeSelfInteraction import EdgeSelfInteraction


class E3Convolution(torch.nn.Module):
    def __init__(self,
                 irreps_node: o3.Irreps,
                 irreps_out: o3.Irreps,
                 num_type: int,
                 basis_size=32,
                 sh_channel=None,
                 hidden_neurons=64,
                 split_stru=0,
                 para=None,
                 **kwargs):
        super().__init__()
        self.irreps_node = irreps_node
        self.irreps_out = irreps_out
        self.basis_size = basis_size
        self.num_type = num_type
        self.split_stru = split_stru
        self.para = para
        self.scatter = MyScatter(para.fix_average, para.N_average)
        self.lmax = self.irreps_node.lmax

        self.sc_node = o3.FullyConnectedTensorProduct(self.irreps_node, o3.Irreps(f"{num_type}x0e"), self.irreps_out)
        self.sc_edge = o3.FullyConnectedTensorProduct(self.irreps_node, o3.Irreps(f"{2*num_type+basis_size}x0e"), self.irreps_out)

        self.lin1_node = o3.Linear(irreps_in=self.irreps_node, irreps_out=self.irreps_node)
        self.lin1_edge = o3.Linear(irreps_in=self.irreps_node, irreps_out=self.irreps_node)

        self.e3tp = o3.FullyConnectedTensorProduct((self.irreps_node+self.irreps_node+self.irreps_node).sort()[0].simplify(),
                                                   o3.Irreps.spherical_harmonics(self.lmax),
                                                   self.irreps_out,
                                                   shared_weights=False)
        self.weight_e3tp = e3nn.nn.FullyConnectedNet([self.basis_size,
                                                     hidden_neurons,
                                                     self.e3tp.weight_numel],
                                                     torch.nn.functional.silu)

        self.gate = Gate(irreps_in=self.irreps_out)
        self.lin2_node = o3.Linear(irreps_in=self.irreps_out, irreps_out=self.irreps_out)
        self.lin2_edge = o3.Linear(irreps_in=self.irreps_out, irreps_out=self.irreps_out)

    def forward(self, f_node, f_edge, sh, node_emb, length_emb, data: Data):
        edge_src, edge_dst = data.edge_index_hop

        # self-connection
        sc_node = self.sc_node(f_node, node_emb)
        sc_edge = self.sc_edge(f_edge, torch.cat([node_emb[edge_src], node_emb[edge_dst], length_emb], dim=-1))

        f_node = self.lin1_node(f_node)
        f_edge = self.lin1_edge(f_edge)

        f_cat = torch.cat([torch.cat([f_node[:, s][edge_src], f_node[:, s][edge_dst], f_edge[:, s]], dim=-1) for s in self.irreps_node.slices()], dim=-1)
        weight = self.weight_e3tp(length_emb)

        if self.split_stru > 1:
            sub_batch_size = math.ceil(data.num_edges/self.split_stru)
            f_edge = [[] for _ in range(self.split_stru)]
            for index_sub_batch in range(self.split_stru):
                if index_sub_batch == self.split_stru-1:
                    batch_slice = slice(sub_batch_size*index_sub_batch, data.num_edges)
                else:
                    batch_slice = slice(sub_batch_size*index_sub_batch, sub_batch_size*(index_sub_batch+1))
                sub_f_edge = self.e3tp(f_cat[batch_slice], sh[batch_slice], weight[batch_slice])
                f_edge[index_sub_batch] = sub_f_edge
            f_edge = torch.cat(f_edge, dim=0)
        else:
            f_edge = self.e3tp(f_cat, sh, weight)

        f_edge = self.gate(f_edge)
        f_node = self.scatter(f_edge, edge_dst, dim=0, dim_size=data.num_nodes)

        f_node = self.lin2_node(f_node)
        f_edge = self.lin2_edge(f_edge)

        # add self-connection
        f_node = f_node+sc_node
        f_edge = f_edge+sc_edge

        return f_node, f_edge
