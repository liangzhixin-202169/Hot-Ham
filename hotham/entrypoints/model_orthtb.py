import torch
from torch import nn
from torch_geometric.data import Data
from torch_scatter import scatter
from typing import Union
import re
import math
from typing import  Union
import e3nn
from e3nn import o3
from ase.atom import atomic_numbers
from .parameters import Parameters
from ..modules.embedding import NodeEmbedding, EdgeEmbedding
from ..modules.fourier_convolution import FourierConvelution
from ..modules.so2_convolution import SO2OutputLayer
from ..modules.common import MCST,  E3LayerNormal
from ..modules.output_linear import OutputLinear,OutputLinear_AtomType,OutputLinear_BondType

def find_common_AM(AtomType_OrbitalIrrepsList: dict):
    from collections import Counter
    from functools import reduce
    count_list = [Counter(v) for v in AtomType_OrbitalIrrepsList.values()]
    union_counter = reduce(lambda x, y: x | y, count_list)
    ir_list = [[ir]*num for ir, num in union_counter.items()]
    return o3.Irreps(reduce(lambda x, y: x+y, ir_list)).sort().irreps


def find_EdgeIrreps_output(irs):
    ir_out = o3.Irreps(None)
    for mul1, (l1, p1) in irs:
        for mul2, (l2, p2) in irs:
            assert mul1 == mul2 == 1
            p = p1*p2
            ir_out += o3.Irreps(list(map(lambda x: o3.Irrep(x, p), range(abs(l1-l2), l1+l2+1))))
    return ir_out

class Model(nn.Module):
    def __init__(self, para:  Union[dict, Parameters], **kwargs):
        super().__init__()
        self.para = para
        self.device = para.device

        self.init_info()

        self.lmax = para.L_max
        self.basis_size = self.para.basis_size
        self.Ce = para.Ce
        self.Co = para.Co
        self.register_buffer("N_average", torch.tensor(0.0, dtype=para.torch_float_dtype))
        self.N_average = torch.tensor(-1.0, dtype=para.torch_float_dtype)


        assert len(self.Ce) == len(self.Co)

        self.layer_irreps = [MCST(self.lmax, 1, self.Ce[i])+MCST(self.lmax, -1, self.Co[i]) for i in range(len(self.Ce))]
        self.layer_irreps = [ir.remove_zero_multiplicities() for ir in self.layer_irreps]
        # restrict the last layer irreps
        self.layer_irreps[-1] = o3.Irreps([irs for irs in self.layer_irreps[-1] if irs[1].l <= self.edge_irreps_output.lmax])

        # Embedding
        self.node_emb = NodeEmbedding(self.num_atomtype)
        self.edge_emb = EdgeEmbedding(self.basis_size,
                                      self.para.rc,
                                      self.layer_irreps[0].num_irreps)

        self.sh_emb = o3.SphericalHarmonics(irreps_out=o3.Irreps.spherical_harmonics(self.lmax), normalize=True)

        self.f_edge_lin = o3.Linear(self.sh_emb.irreps_out, self.layer_irreps[0], shared_weights=False)

        self.weight_f = e3nn.nn.FullyConnectedNet([2*self.node_emb.out_dim+self.edge_emb.out_dim,
                                                    64,
                                                   self.f_edge_lin.weight_numel],
                                                  torch.nn.functional.silu)

        # Gaunt Conv
        self.convblock_num = len(self.Ce)-2

        LayerNorm_node, LayerNorm_edge = torch.nn.ModuleList([]), torch.nn.ModuleList([])
        FourierConv_layer = torch.nn.ModuleList([])

        for layer_index in range(self.convblock_num):

            layer = FourierConvelution(irreps_node=self.layer_irreps[layer_index],
                                       irreps_out=self.layer_irreps[layer_index+1],
                                       num_type=self.num_atomtype,
                                       basis_size=self.basis_size,
                                       sh_channel=1,
                                       para=para,
                                       N_average=self.N_average)

            FourierConv_layer.append(layer)

            LayerNorm_node.append(E3LayerNormal(layer.irreps_out, floatdtype=self.para.torch_float_dtype))
            LayerNorm_edge.append(E3LayerNormal(layer.irreps_out, floatdtype=self.para.torch_float_dtype))

        self.FourierConv_layer = FourierConv_layer

        # SO2 Conv
        self.so2outputlayer = SO2OutputLayer(irreps_node=FourierConv_layer[-1].irreps_out,
                                             irreps_out=self.layer_irreps[-1],
                                             num_type=self.num_atomtype,
                                             basis_size=self.basis_size,
                                             para=self.para,
                                             N_average=self.N_average)

        LayerNorm_node.append(E3LayerNormal(self.so2outputlayer.irreps_out, floatdtype=self.para.torch_float_dtype))
        LayerNorm_edge.append(E3LayerNormal(self.so2outputlayer.irreps_out, floatdtype=self.para.torch_float_dtype))
        self.LayerNorm_node = LayerNorm_node
        self.LayerNorm_edge = LayerNorm_edge



        if  self.para.using_element_net:
            self.node_endlinear = OutputLinear_AtomType(self.so2outputlayer.irreps_out, self.edge_irreps_output,self.num_atomtype)
            self.edge_endlinear = OutputLinear_BondType(self.so2outputlayer.irreps_out, self.edge_irreps_output,self.num_atomtype)
        else:
            self.node_endlinear = OutputLinear(self.so2outputlayer.irreps_out, self.edge_irreps_output, intdtype=self.para.torch_int_dtype)
            self.edge_endlinear = OutputLinear(self.so2outputlayer.irreps_out, self.edge_irreps_output, intdtype=self.para.torch_int_dtype)

        if self.para.using_soc:
            input_length=[]
            for i in range(self.para.num_types):
                input_length.append(len(self.AtomType_AMList.get(i)))
            max_l=torch.as_tensor(input_length,dtype=self.para.torch_int_dtype,device=self.device).max().item()

            #self.lambdas_layer = o3.Linear(self.edge_irreps_output, (max_l*o3.Irreps("0e")).simplify())
            self.lambdas_layer = e3nn.nn.FullyConnectedNet([self.edge_irreps_output.dim,64,max_l],torch.nn.functional.silu)

    def init_info(self):
        self.num_atomtype = len(self.para.orbit)
        self.symbols = self.para.orbit.keys()
        self.AtomSymbol_to_AtomNumber = {atomsymbol: atomnumber for atomsymbol, atomnumber in atomic_numbers.items() if atomsymbol in self.symbols}
        self.AtomNumber_to_AtomSymbol = {atomnumber: atomsymbol for atomsymbol, atomnumber in self.AtomSymbol_to_AtomNumber.items()}
        unique_type = sorted(self.AtomNumber_to_AtomSymbol.keys())
        self.AtomNumber_to_AtomType = {num: i for i, num in enumerate(unique_type)}
        self.AtomSymbol_to_AtomType = {self.AtomNumber_to_AtomSymbol[atomnumber]: atomtype for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
        self.AtomType_to_AtomNumber = {atomtype: atomnumber for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
        self.AtomType_to_AtomSymbol = {atomtype: self.AtomNumber_to_AtomSymbol[atomnumber] for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}

        self.AMSymbol_to_AM = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
        self.AtomType_AMSymbolList = {self.AtomSymbol_to_AtomType[k]: list(map(lambda x: ''.join(re.findall(r'[A-Za-z]', x)), v)) for k, v in self.para.orbit.items()}
        self.AtomType_AMList = {k: list(map(lambda x: self.AMSymbol_to_AM[x], v)) for k, v in self.AtomType_AMSymbolList.items()}
        self.AtomType_OrbitalIrrepsList = {k: list(map(lambda x: o3.Irrep(x, (-1)**x), v)) for k, v in self.AtomType_AMList.items()}
        self.AtomType_OrbitalIrreps = {k: o3.Irreps(v).sort()[0].simplify() for k, v in self.AtomType_OrbitalIrrepsList.items()}
        # Index by angular momentum l, including zero-count lower-l orbitals.
        # A p-only basis must be [0, 1], not [1], for find_Hblock below.
        self.AtomType_OrbitalNum = [
            [sum(mul for mul, (l, _) in irreps if l == order)
             for order in range(irreps.lmax + 1)]
            for _, irreps in sorted(self.AtomType_OrbitalIrreps.items())
        ]
        self.AtomType_OrbitalSum = torch.tensor(list(map(lambda x: self.AtomType_OrbitalIrreps[x].dim, sorted(self.AtomType_OrbitalIrreps))), dtype=self.para.torch_int_dtype).to(self.device)

        self.CommonOrbitalIrreps = find_common_AM(self.AtomType_OrbitalIrrepsList)
        self.CommonAM = self.CommonOrbitalIrreps.ls
        self.cg = self.find_cg(self.CommonOrbitalIrreps)

        #########################
        self.irreps_node_attr = o3.Irreps([(len(self.symbols), (0, 1))])
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax=4)
        #########################

        self.edge_irreps_output = find_EdgeIrreps_output(self.CommonOrbitalIrreps)
        #self.OnsiteEnergy, self.OnsiteEnergy_offset = self.find_OnsitesEnergy()


    def find_soc_value(self,data,f_node):
        f_scatter=scatter(f_node,data.AtomType,dim=0,reduce="mean")
        result=self.lambdas_layer(f_scatter)
        return result

    def forward(self, data: Data):
        # 节点类型和边类型嵌入
        AtomType = data.AtomType
        edge_src, edge_dst = data.edge_index_hop
        edge_length = data.d_hop
        edge_vector = data.D_hop[:, [1, 2, 0]]

        node_emb = self.node_emb(AtomType)

        edge_emb = self.edge_emb(edge_length)

        sh_emb = self.sh_emb(edge_vector)

        weight_f = self.weight_f(torch.cat([node_emb[edge_src], node_emb[edge_dst], edge_emb], dim=-1))

        f_edge = self.f_edge_lin(sh_emb, weight_f)

        f_node = scatter(f_edge, edge_dst, dim=0, dim_size=data.num_nodes, reduce="mean")

        # Gaunt Conv
        for index_layer, fclayer in enumerate(self.FourierConv_layer):
            f_node, f_edge = fclayer(f_node, f_edge, sh_emb, node_emb, edge_emb, data)
            layerNorm_node = self.LayerNorm_node[index_layer]
            layerNorm_edge = self.LayerNorm_edge[index_layer]
            f_node = layerNorm_node(f_node, data.batch)
            if data.batch is None:
                f_edge = layerNorm_edge(f_edge, None)
            else:
                f_edge = layerNorm_edge(f_edge, data.batch[edge_src])


        # SO2 Conv
        f_node, f_edge = self.so2outputlayer(f_node, f_edge, sh_emb, node_emb, edge_emb, data)

        f_node = self.LayerNorm_node[-1](f_node, data.batch)
        if data.batch is None:
            f_edge = self.LayerNorm_edge[-1](f_edge, None)
        else:
            f_edge = self.LayerNorm_edge[-1](f_edge, data.batch[edge_src])

        # linear map to Hamilton's irreps
        if  self.para.using_element_net:
            f_node = self.node_endlinear(f_node,data)
            f_edge = self.edge_endlinear(f_edge,data)
        else:
            f_node = self.node_endlinear(f_node)
            f_edge = self.edge_endlinear(f_edge)

        # #########################
        # # 等变性检验 res version
        # #########################
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
        # f_node_rotated = scatter(f_edge_rotated, edge_dst, dim=0, dim_size=data.num_nodes, reduce="mean")

        # # f_edge_rotated_res=f_edge_rotated
        # # f_node_rotated_res=f_node_rotated

        # # 傅里叶卷积
        # for index_layer, fclayer in enumerate(self.FourierConv_layer):
        #   f_node_rotated, f_edge_rotated = fclayer(f_node_rotated, f_edge_rotated, sh_emb_rotated, node_emb, edge_emb, data)
        #   f_node_rotated = self.LayerNorm_node[index_layer](f_node_rotated)
        #   f_edge_rotated = self.LayerNorm_edge[index_layer](f_edge_rotated)

        # # f_edge_rotated=f_edge_rotated+f_edge_rotated_res
        # # f_node_rotated=f_node_rotated+f_node_rotated_res

        # # SO2输出层
        # f_node_rotated, f_edge_rotated = self.so2outputlayer(f_node_rotated, f_edge_rotated, sh_emb_rotated, node_emb, edge_emb, data)

        #   f_node_rotated = self.LayerNorm_node[-1](f_node_rotated, data.batch)
        #   f_edge_rotated = self.LayerNorm_edge[-1](f_edge_rotated, data.batch[edge_src])

        #     # linear map to Hamilton's irreps
        #f_node_rotated = self.node_endlinear(f_node_rotated)
        #f_edge_rotated = self.edge_endlinear(f_edge_rotated)

        # delta1 = (torch.einsum("ij,zj->zi", D.to(f_node.device), f_node)-f_node_rotated).abs().max()
        # delta2 = (torch.einsum("ij,zj->zi", D.to(f_edge.device), f_edge)-f_edge_rotated).abs().max()
        # ratio1 = delta1/torch.max(torch.tensor([f_node.abs().max(), f_node_rotated.abs().max()]))
        # ratio2 = delta2/torch.max(torch.tensor([f_edge.abs().max(), f_edge_rotated.abs().max()]))
        # print(delta1.item(), delta2.item(), ratio1.item(), ratio2.item())
        # #########################


        if self.para.using_soc:
            soc_val=self.find_soc_value(data,f_node)
            DirectProduct = self.DirectSum_to_DirectProduct(f_node, f_edge, data)
            H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.find_Hblock(DirectProduct, data)
            return H_block, GraphEdgeIndex_to_BlockEdgeIndex,soc_val.abs()#.sort(dim=1)[0]
        else:
            DirectProduct = self.DirectSum_to_DirectProduct(f_node, f_edge, data)
            H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.find_Hblock(DirectProduct, data)
            return H_block, GraphEdgeIndex_to_BlockEdgeIndex

    def find_cg(self, irs):
        l_list = sorted(set(irs.ls))
        cg = [[[] for j in range(max(l_list)+1)] for i in range(max(l_list)+1)]
        for l1 in l_list:
            for l2 in l_list:
                for l3 in range(abs(l1-l2), l1+l2+1):
                    cg[l1][l2].append(o3.wigner_3j(l1, l2, l3, dtype=self.para.torch_float_dtype))
                cg[l1][l2] = torch.cat(cg[l1][l2], dim=-1).to(device=self.device, dtype=self.para.torch_float_dtype)
        return cg

    def DirectSum_to_DirectProduct(self, feature_node, feature_edge, data):
        flag1 = data.edge_index_hop[0] == data.edge_index_hop[1]
        flag2 = torch.norm(data.S_hop.to(self.para.torch_float_dtype), dim=-1) < 1.0e-6
        selfconnect_index = torch.arange(feature_edge.shape[0], dtype=self.para.torch_int_dtype, device=feature_node.device)[flag1*flag2]
        feature_edge[selfconnect_index] += feature_node

        H_blocks = []
        offset = 0
        for l1 in self.CommonAM:
            for l2 in self.CommonAM:
                size_block = sum(list(map(lambda x: 2*x+1, range(abs(l1-l2), l1+l2+1))))
                block = torch.einsum("nk,ijk->nij", feature_edge[:, offset:offset+size_block], self.cg[l1][l2])
                H_blocks.append(block)
                offset += size_block
        return H_blocks

    def find_Hblock(self, DirectProduct, data):
        SqrtLength = int(math.sqrt(len(DirectProduct)))
        hams = []

        num_atomtype = torch.max(data.AtomType)+1
        n1, n2 = data.edge_index_hop
        num_edge = len(n1)

        GraphEdgeIndex_to_BlockEdgeIndex = torch.zeros(num_edge, dtype=self.para.torch_int_dtype, device=self.device)

        for atomtype_1 in range(num_atomtype):
            for atomtype_2 in range(num_atomtype):
                edge_index_seleced = torch.arange(num_edge, dtype=self.para.torch_int_dtype, device=self.device)[(data.AtomType[n1] == atomtype_1)*(data.AtomType[n2] == atomtype_2)]

                sum_orbit_1 = self.AtomType_OrbitalSum[atomtype_1]
                sum_orbit_2 = self.AtomType_OrbitalSum[atomtype_2]
                ham = torch.zeros((len(edge_index_seleced), sum_orbit_1, sum_orbit_2), device=self.device, dtype=self.para.torch_float_dtype)

                numlist_orbit_1 = self.AtomType_OrbitalNum[atomtype_1].copy()
                block_offset, row_offset1 = 0, 0

                for l1 in self.CommonAM:
                    if len(numlist_orbit_1) < l1+1 or numlist_orbit_1[l1] == 0:
                        block_offset += SqrtLength
                        continue
                    numlist_orbit_1[l1] -= 1

                    row_offset2 = 2*l1+1
                    col_offset1 = 0
                    numlist_orbit_2 = self.AtomType_OrbitalNum[atomtype_2].copy()

                    for l2 in self.CommonAM:
                        if len(numlist_orbit_2) < l2+1 or numlist_orbit_2[l2] == 0:
                            block_offset += 1
                            continue
                        numlist_orbit_2[l2] -= 1

                        col_offset2 = 2*l2+1

                        block = DirectProduct[block_offset][edge_index_seleced]
                        ham[:, row_offset1:row_offset1+row_offset2, col_offset1:col_offset1+col_offset2] = block

                        block_offset += 1
                        col_offset1 += col_offset2

                    row_offset1 += row_offset2

                hams.append(ham)
                GraphEdgeIndex_to_BlockEdgeIndex[edge_index_seleced] = torch.arange(edge_index_seleced.shape[0], dtype=self.para.torch_int_dtype, device=self.device)

        # symmetrize matrix
        if (hasattr(data, "num_graphs")) and (data.num_graphs != 1):
            edge_num = torch.tensor([data[i].num_edges for i in range(data.num_graphs)], dtype=self.para.torch_int_dtype, device=self.device)
            offset = torch.cat([edge_num.new_zeros(1), torch.cumsum(edge_num, dim=0)[:-1]])
            edge_inverse = torch.cat([data[i].edge_inverse+offset[i] for i in range(data.num_graphs)])
        else:
            edge_inverse = data.edge_inverse

        for atomtype_1 in range(num_atomtype):
            for atomtype_2 in range(atomtype_1, num_atomtype):
                edge_12_mask = (data.AtomType[n1] == atomtype_1)*(data.AtomType[n2] == atomtype_2)
                edge_12 = torch.arange(num_edge, dtype=self.para.torch_int_dtype, device=self.device)[edge_12_mask]
                edge_21 = edge_inverse[edge_12]
                edge_block21 = GraphEdgeIndex_to_BlockEdgeIndex[edge_21]

                index_HBlcok12 = atomtype_1*num_atomtype+atomtype_2
                index_HBlcok21 = atomtype_2*num_atomtype+atomtype_1
                hams[index_HBlcok12] = 0.5*(hams[index_HBlcok12]+hams[index_HBlcok21][edge_block21].transpose(-2, -1))
                if atomtype_1 != atomtype_2:
                    edge_21_mask = (data.AtomType[n1] == atomtype_2)*(data.AtomType[n2] == atomtype_1)
                    edge_21 = torch.arange(num_edge, dtype=self.para.torch_int_dtype, device=self.device)[edge_21_mask]
                    edge_12 = edge_inverse[edge_21]
                    edge_block12 = GraphEdgeIndex_to_BlockEdgeIndex[edge_12]
                    hams[index_HBlcok21] = hams[index_HBlcok12][edge_block12].transpose(-2, -1)

        return hams, GraphEdgeIndex_to_BlockEdgeIndex
