import torch
from torch_geometric.data import Data
from typing import Union
import e3nn
from e3nn import o3
from .parameters import Parameters
from ..modules.embedding import NodeEmbedding, EdgeEmbedding
from ..modules.so2_convolution import SO2OutputLayer
from ..modules.general_local_convolution import GeneralLocalConv
from ..modules.common import MCST, E3LayerNormal, MyScatter
from ..modules.output_linear import OutputLinear
from ..modules.magnetic_module import MagMessagePassing
from ..modules.te3nn.to3 import TO3_Irrep, TO3_Irreps
from .basemodel import BaseModel


class Model(BaseModel):
    def __init__(self, para: Union[dict, Parameters], **kwargs):
        super().__init__(para, **kwargs)

        self.lmax = para.L_max
        self.basis_size = self.para.basis_size
        self.Ce = para.Ce
        self.Co = para.Co
        assert len(self.Ce) == len(self.Co)

        self.layer_irreps = [MCST(self.lmax, 1, self.Ce[i]) + MCST(self.lmax, -1, self.Co[i]) for i in range(len(self.Ce))]
        self.layer_irreps = [ir.remove_zero_multiplicities() for ir in self.layer_irreps]
        # restrict the last layer irreps
        if para.restrict_last_out:
            self.layer_irreps[-1] = o3.Irreps([irs for irs in self.layer_irreps[-1] if irs[1].l <= self.edge_irreps_output.lmax])

        # Define scatter function
        self.scatter = MyScatter(para.N_average)

        # Embedding
        self.node_emb = NodeEmbedding(self.num_atomtype)
        self.edge_emb = EdgeEmbedding(self.basis_size, self.para.rc, self.layer_irreps[0].num_irreps, basis=self.para.basis)
        self.edge_sh_emb = o3.SphericalHarmonics(irreps_out=o3.Irreps.spherical_harmonics(self.lmax), normalize=True)

        self.f_edge_lin = o3.Linear(self.edge_sh_emb.irreps_out, self.layer_irreps[0], shared_weights=False)
        self.weight_f = e3nn.nn.FullyConnectedNet(
            [2 * self.node_emb.out_dim + self.edge_emb.out_dim, 64, self.f_edge_lin.weight_numel], torch.nn.functional.silu
        )

        # Gaunt Conv
        self.convblock_num = len(self.Ce) - 2
        self.using_layernorm = para.using_layernorm

        if self.using_layernorm:
            LayerNorm_node, LayerNorm_edge = torch.nn.ModuleList([]), torch.nn.ModuleList([])
        GauntConv_layer = torch.nn.ModuleList([])

        if para.model_type == 0:
            from ..modules.gaunt_convolution_0 import GauntConvolution as GeneralConvolution
        elif para.model_type == 1:
            from ..modules.gaunt_convolution_1 import GauntConvolution as GeneralConvolution
        for layer_index in range(self.convblock_num):
            layer = GeneralConvolution(
                irreps_node=self.layer_irreps[layer_index],
                irreps_out=self.layer_irreps[layer_index + 1],
                num_type=self.num_atomtype,
                basis_size=self.basis_size,
                sh_channel=1,
                para=para,
            )
            GauntConv_layer.append(layer)

            if self.using_layernorm:
                LayerNorm_node.append(E3LayerNormal(layer.irreps_out, floatdtype=self.para.torch_float_dtype))
                LayerNorm_edge.append(E3LayerNormal(layer.irreps_out, floatdtype=self.para.torch_float_dtype))

        self.GauntConv_layer = GauntConv_layer

        # SO2 Conv
        self.so2outputlayer = SO2OutputLayer(
            irreps_node=GauntConv_layer[-1].irreps_out,
            irreps_out=self.layer_irreps[-1],
            num_type=self.num_atomtype,
            basis_size=self.basis_size,
            para=self.para,
        )
        # self.so2outputlayer = GeneralLocalConv(irreps_node=GauntConv_layer[-1].irreps_out, irreps_out=self.layer_irreps[-1], num_type=self.num_atomtype, basis_size=self.basis_size, para=self.para)

        if self.using_layernorm:
            LayerNorm_node.append(E3LayerNormal(self.so2outputlayer.irreps_out, floatdtype=self.para.torch_float_dtype))
            LayerNorm_edge.append(E3LayerNormal(self.so2outputlayer.irreps_out, floatdtype=self.para.torch_float_dtype))
            self.LayerNorm_node = LayerNorm_node
            self.LayerNorm_edge = LayerNorm_edge

        self.magnetism = para.magnetism
        if self.magnetism:
            if para.get("to3") is None:
                raise ValueError("Missing required input parameter 'to3' for magnetic LCAO version 1")
            mmp_mid_to3_irreps_reduced_list = []
            for i in range(len(para["to3"])):
                channel = para["to3"][i]
                mmp_mid_irreps = MCST(self.lmax + 1, 1, channel) + MCST(self.lmax + 1, -1, channel)
                t_ps = [1 for _ in range(len(mmp_mid_irreps))]
                mmp_mid_irreps += mmp_mid_irreps
                t_ps += [-t_p for t_p in t_ps]
                mmp_mid_to3_irreps = TO3_Irreps([(mul, (l, p, t_p)) for (mul, (l, p)), t_p in zip(mmp_mid_irreps, t_ps)])
                mmp_mid_to3_irreps_reduced = TO3_Irreps(None)
                if para["reduced_to3"]:
                    for mul, (l, p, t_p) in mmp_mid_to3_irreps:
                        if TO3_Irrep((l, p, t_p)) in self.edge_to3_irreps_output:
                            mmp_mid_to3_irreps_reduced += TO3_Irreps([(mul, (l, p, t_p))])
                mmp_mid_to3_irreps_reduced_list.append(mmp_mid_to3_irreps_reduced)

            self.magmessagepassing = MagMessagePassing(
                irreps_node=self.so2outputlayer.irreps_out, irreps_mids=mmp_mid_to3_irreps_reduced_list, para=para
            )

        irreps_prev = self.so2outputlayer.irreps_out
        irreps_out = self.edge_irreps_output
        #if self.magmessagepassing is not None:
        if self.magnetism:
            irreps_prev = self.magmessagepassing.node_out
            irreps_out = self.edge_to3_irreps_output
        if para.using_rearrangement_linear:
            self.node_endlinear = OutputLinear(irreps_prev, irreps_out, intdtype=self.para.torch_int_dtype)
            self.edge_endlinear = OutputLinear(irreps_prev, irreps_out, intdtype=self.para.torch_int_dtype)
        else:
            self.node_endlinear = o3.Linear(irreps_prev, irreps_out)
            self.edge_endlinear = o3.Linear(irreps_prev, irreps_out)

    def forward(self, data: Data):
        # Embed "atom type", "edge length" and "direction"
        AtomType = data.AtomType
        edge_src, edge_dst = data.edge_index_hop
        edge_length = data.d_hop
        edge_vector = data.D_hop[:, [1, 2, 0]]

        node_emb = self.node_emb(AtomType)
        edge_emb = self.edge_emb(edge_length)
        edge_sh = self.edge_sh_emb(edge_vector)

        weight_f = self.weight_f(torch.cat([node_emb[edge_src], node_emb[edge_dst], edge_emb], dim=-1))
        f_edge = self.f_edge_lin(edge_sh, weight_f)
        f_node = self.scatter(f_edge, edge_dst, dim=0, dim_size=data.num_nodes)

        # Gaunt Conv
        for index_layer, fclayer in enumerate(self.GauntConv_layer):
            f_node, f_edge = fclayer(f_node, f_edge, edge_sh, node_emb, edge_emb, data)
            if self.using_layernorm:
                layerNorm_node = self.LayerNorm_node[index_layer]
                layerNorm_edge = self.LayerNorm_edge[index_layer]
                f_node = layerNorm_node(f_node, data.batch)
                if data.batch is None:
                    f_edge = layerNorm_edge(f_edge, None)
                else:
                    f_edge = layerNorm_edge(f_edge, data.batch[edge_src])

        # SO2 Conv
        f_node, f_edge = self.so2outputlayer(f_node, f_edge, edge_sh, node_emb, edge_emb, data)

        if self.using_layernorm:
            f_node = self.LayerNorm_node[-1](f_node, data.batch)
            if data.batch is None:
                f_edge = self.LayerNorm_edge[-1](f_edge, None)
            else:
                f_edge = self.LayerNorm_edge[-1](f_edge, data.batch[edge_src])

        if self.magnetism:
            f_node, f_edge = self.magmessagepassing(f_node=f_node, f_edge=f_edge, edge_sh=edge_sh, node_emb=node_emb, length_emb=edge_emb, data=data)

        # linear map to Hamilton's irreps
        f_node = self.node_endlinear(f_node)
        f_edge = self.edge_endlinear(f_edge)

        # l3 -> l1xl2
        DirectProduct = self.DirectSum_to_DirectProduct(f_node, f_edge, data)
        H_block = self.find_Hblock(DirectProduct, data)
        return H_block


if __name__ == "__main__":
    pass
