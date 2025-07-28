import torch
from torch_geometric.data import Data
from typing import Union
import e3nn
from e3nn import o3
from .Parameters import Parameters
from modules.Embedding import NodeEmbedding, EdgeEmbedding
from modules.SO2Conv import SO2OutputLayer
from modules.SelfInteraction import SelfInteraction
from modules.common import MCST, E3LayerNormal, MyScatter
from modules.OutputLinear import OutputLinear
from .basemodel import BaseModel


class Model(BaseModel):
    def __init__(self, para:  Union[dict, Parameters], **kwargs):
        super().__init__(para, **kwargs)

        self.lmax = para.L_max
        self.basis_size = self.para.basis_size
        self.Ce = para.Ce
        self.Co = para.Co
        assert len(self.Ce) == len(self.Co)

        self.layer_irreps = [MCST(self.lmax, 1, self.Ce[i])+MCST(self.lmax, -1, self.Co[i]) for i in range(len(self.Ce))]
        self.layer_irreps = [ir.remove_zero_multiplicities() for ir in self.layer_irreps]
        # restrict the last layer irreps
        if para.restrict_last_out:
            self.layer_irreps[-1] = o3.Irreps([irs for irs in self.layer_irreps[-1] if irs[1].l <= self.edge_irreps_output.lmax])

        # Define scatter function
        self.scatter = MyScatter(para.fix_average, para.N_average)

        # Embedding
        self.node_emb = NodeEmbedding(self.num_atomtype)
        self.edge_emb = EdgeEmbedding(self.basis_size,
                                      self.para.rc,
                                      self.layer_irreps[0].num_irreps)
        self.sh_emb = o3.SphericalHarmonics(irreps_out=o3.Irreps.spherical_harmonics(self.lmax),
                                            normalize=True)

        self.f_edge_lin = o3.Linear(self.sh_emb.irreps_out, self.layer_irreps[0], shared_weights=False)
        self.weight_f = e3nn.nn.FullyConnectedNet([2*self.node_emb.out_dim+self.edge_emb.out_dim,
                                                   64,
                                                   self.f_edge_lin.weight_numel],
                                                  torch.nn.functional.silu)

        # Gaunt Conv
        self.convblock_num = len(self.Ce)-2
        self.using_layernorm = para.using_layernorm

        if self.using_layernorm:
            LayerNorm_node, LayerNorm_edge = torch.nn.ModuleList([]), torch.nn.ModuleList([])
        GauntConv_layer = torch.nn.ModuleList([])

        if para.model_type == -1:
            from modules.E3Conv import E3Convolution as GeneralConvolution
        elif para.model_type == 0:
            from modules.GConv0 import GauntConvolution as GeneralConvolution
        elif para.model_type == 1:
            from modules.GConv1 import GauntConvolution as GeneralConvolution
        elif para.model_type == 2:
            from modules.SO2Conv import SO2OutputLayer as GeneralConvolution
        for layer_index in range(self.convblock_num):
            layer = GeneralConvolution(irreps_node=self.layer_irreps[layer_index],
                                       irreps_out=self.layer_irreps[layer_index+1],
                                       num_type=self.num_atomtype,
                                       basis_size=self.basis_size,
                                       sh_channel=1,
                                       split_stru=self.para.split_stru,
                                       is_final_layer=False,
                                       para=para)
            GauntConv_layer.append(layer)

            if self.using_layernorm:
                LayerNorm_node.append(E3LayerNormal(layer.irreps_out))
                LayerNorm_edge.append(E3LayerNormal(layer.irreps_out))

        self.GauntConv_layer = GauntConv_layer

        # SO2 Conv
        self.so2outputlayer = SO2OutputLayer(irreps_node=GauntConv_layer[-1].irreps_out,
                                             irreps_out=self.layer_irreps[-1],
                                             num_type=self.num_atomtype,
                                             basis_size=self.basis_size,
                                             para=self.para)

        if self.using_layernorm:
            LayerNorm_node.append(E3LayerNormal(self.so2outputlayer.irreps_out))
            LayerNorm_edge.append(E3LayerNormal(self.so2outputlayer.irreps_out))
            self.LayerNorm_node = LayerNorm_node
            self.LayerNorm_edge = LayerNorm_edge

        # self interaction
        if self.para.using_manybody:
            manybody_layer = torch.nn.ModuleList([])

            # before Gaunt Conv
            for layer_index in range(self.convblock_num):
                layer = SelfInteraction(v_max=self.para.v_max,
                                        irreps_in=GauntConv_layer[layer_index].irreps_node,
                                        irreps_out=GauntConv_layer[layer_index].irreps_node,
                                        num_type=self.num_atomtype,
                                        basis_size=self.basis_size,
                                        split_stru=self.para.split_stru,
                                        para=self.para)
                manybody_layer.append(layer)

            # before SO(2) conv
            layer = SelfInteraction(v_max=self.para.v_max,
                                    irreps_in=self.so2outputlayer.irreps_node,
                                    irreps_out=self.so2outputlayer.irreps_node,
                                    num_type=self.num_atomtype,
                                    basis_size=self.basis_size,
                                    split_stru=self.para.split_stru,
                                    para=self.para)
            manybody_layer.append(layer)
            self.manybody_layer = manybody_layer

        # readout
        if para.using_rearrangement_linear:
            self.node_endlinear = OutputLinear(self.so2outputlayer.irreps_out, self.edge_irreps_output)
            self.edge_endlinear = OutputLinear(self.so2outputlayer.irreps_out, self.edge_irreps_output)
        else:
            self.node_endlinear = o3.Linear(self.so2outputlayer.irreps_out, self.edge_irreps_output)
            self.edge_endlinear = o3.Linear(self.so2outputlayer.irreps_out, self.edge_irreps_output)

    def forward(self, data: Data):
        # Embed "atom type", "edge length" and "direction"
        AtomType = data.AtomType
        edge_src, edge_dst = data.edge_index_hop
        edge_length = data.d_hop
        edge_vector = data.D_hop[:, [1, 2, 0]]

        node_emb = self.node_emb(AtomType)
        edge_emb = self.edge_emb(edge_length)
        sh_emb = self.sh_emb(edge_vector)

        weight_f = self.weight_f(torch.cat([node_emb[edge_src], node_emb[edge_dst], edge_emb], dim=-1))
        f_edge = self.f_edge_lin(sh_emb, weight_f)
        f_node = self.scatter(f_edge, edge_dst, dim=0, dim_size=data.num_nodes)

        # General Conv
        for index_layer, fclayer in enumerate(self.GauntConv_layer):

            if self.para.using_manybody:
                f_node, f_edge = self.manybody_layer[index_layer](f_node, f_edge,  node_emb, edge_emb, data)

            f_node, f_edge = fclayer(f_node, f_edge, sh_emb, node_emb, edge_emb, data)
            if self.using_layernorm:
                layerNorm_node = self.LayerNorm_node[index_layer]
                layerNorm_edge = self.LayerNorm_edge[index_layer]
                f_node = layerNorm_node(f_node, data.batch)
                if data.batch is None:
                    f_edge = layerNorm_edge(f_edge, None)
                else:
                    f_edge = layerNorm_edge(f_edge, data.batch[edge_src])

        # SO2 Conv
        if self.para.using_manybody:
            f_node, f_edge = self.manybody_layer[-1](f_node, f_edge,  node_emb, edge_emb, data)
        f_node, f_edge = self.so2outputlayer(f_node, f_edge, sh_emb, node_emb, edge_emb, data)

        if self.using_layernorm:
            f_node = self.LayerNorm_node[-1](f_node, data.batch)
            if data.batch is None:
                f_edge = self.LayerNorm_edge[-1](f_edge, None)
            else:
                f_edge = self.LayerNorm_edge[-1](f_edge, data.batch[edge_src])

        # linear map to Hamilton's irreps
        f_node = self.node_endlinear(f_node)
        f_edge = self.edge_endlinear(f_edge)

        #########################
        # check equilvariance
        #########################
        # R_cpu = o3.rand_matrix(1).squeeze()
        # R_device = R_cpu.to(self.device)
        # D = self.node_endlinear.irreps_out.D_from_matrix(R_cpu)
        # data.wigner_D = [torch.einsum("ij,bjk->bik", o3.Irrep(l, -1).D_from_matrix(R_cpu).to(self.device), D) for l, D in zip(range(len(data.wigner_D)), data.wigner_D)]
        # if self.para.edge_include_sc:
        #     for l in range(len(data.wigner_D)):
        #         data.wigner_D[l][data.mask_sc] = torch.eye(n=2*l+1, dtype=data.wigner_D[l].dtype, device=data.wigner_D[l].device).unsqueeze(0)

        # edge_vector_rotated = torch.einsum("ij,zj->zi", R_device, edge_vector)
        # sh_emb_rotated = self.sh_emb(edge_vector_rotated)
        # f_edge_rotated = self.f_edge_lin(sh_emb_rotated, weight_f)
        # f_node_rotated = self.scatter(f_edge_rotated, edge_dst, dim=0, dim_size=data.num_nodes)

        # # Gaunt Conv
        # for index_layer, fclayer in enumerate(self.GauntConv_layer):

        #     if self.para.using_manybody:
        #         f_node_rotated, f_edge_rotated = self.manybody_layer[index_layer](f_node_rotated, f_edge_rotated,  node_emb, edge_emb, data)

        #     f_node_rotated, f_edge_rotated = fclayer(f_node_rotated, f_edge_rotated, sh_emb_rotated, node_emb, edge_emb, data)
        #     if self.using_layernorm:
        #         f_node_rotated = self.LayerNorm_node[index_layer](f_node_rotated)
        #         f_edge_rotated = self.LayerNorm_edge[index_layer](f_edge_rotated)

        # # SO2 Conv
        # if self.para.using_manybody:
        #     f_node_rotated, f_edge_rotated = self.manybody_layer[-1](f_node_rotated, f_edge_rotated,  node_emb, edge_emb, data)
        # f_node_rotated, f_edge_rotated = self.so2outputlayer(f_node_rotated, f_edge_rotated, sh_emb_rotated, node_emb, edge_emb, data)

        # if self.using_layernorm:
        #     f_node_rotated = self.LayerNorm_node[-1](f_node_rotated, data.batch)
        #     f_edge_rotated = self.LayerNorm_edge[-1](f_edge_rotated, data.batch[edge_src])

        # # linear map to Hamilton's irreps
        # f_node_rotated = self.node_endlinear(f_node_rotated)
        # f_edge_rotated = self.edge_endlinear(f_edge_rotated)

        # delta1 = (torch.einsum("ij,zj->zi", D.to(f_node.device), f_node)-f_node_rotated).abs().max()
        # delta2 = (torch.einsum("ij,zj->zi", D.to(f_edge.device), f_edge)-f_edge_rotated).abs().max()
        # ratio1 = delta1/torch.max(torch.tensor([f_node.abs().max(), f_node_rotated.abs().max()]))
        # ratio2 = delta2/torch.max(torch.tensor([f_edge.abs().max(), f_edge_rotated.abs().max()]))
        # print(delta1.item(), delta2.item(), ratio1.item(), ratio2.item())
        # #########################

        # l3 -> l1xl2
        DirectProduct = self.DirectSum_to_DirectProduct(f_node, f_edge, data)
        H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.find_Hblock(DirectProduct, data)
        return H_block, GraphEdgeIndex_to_BlockEdgeIndex


if __name__ == "__main__":
    pass
