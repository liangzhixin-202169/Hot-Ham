import torch
from torch import nn
from typing import Union
import math
import re
from ase.atom import atomic_numbers
from e3nn import o3
from .Parameters import Parameters


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


class BaseModel(nn.Module):
    def __init__(self, para:  Union[dict, Parameters], **kwargs):
        super().__init__()
        self.para = para
        self.device = para.device

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
        self.AtomType_OrbitalNum = [[mul for mul, (l, p) in self.AtomType_OrbitalIrreps[i]] for i in sorted(self.AtomType_OrbitalIrreps)]
        self.AtomType_OrbitalSum = torch.tensor(list(map(lambda x: self.AtomType_OrbitalIrreps[x].dim, sorted(self.AtomType_OrbitalIrreps))), dtype=self.para.intdtype).to(self.device)

        self.CommonOrbitalIrreps = find_common_AM(self.AtomType_OrbitalIrrepsList)
        self.CommonAM = self.CommonOrbitalIrreps.ls
        self.cg = self.find_cg(self.CommonOrbitalIrreps)

        #########################
        self.irreps_node_attr = o3.Irreps([(len(self.symbols), (0, 1))])
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax=4)
        #########################

        self.edge_irreps_output = find_EdgeIrreps_output(self.CommonOrbitalIrreps)

    def find_cg(self, irs):
        l_list = sorted(set(irs.ls))
        cg = [[[] for j in range(max(l_list)+1)] for i in range(max(l_list)+1)]
        for l1 in l_list:
            for l2 in l_list:
                for l3 in range(abs(l1-l2), l1+l2+1):
                    cg[l1][l2].append(o3.wigner_3j(l1, l2, l3))
                cg[l1][l2] = torch.cat(cg[l1][l2], dim=-1).to(device=self.device)
        return cg

    def DirectSum_to_DirectProduct(self, feature_node, feature_edge, data):
        if self.para.edge_include_sc:
            flag1 = data.edge_index_hop[0] == data.edge_index_hop[1]
            flag2 = torch.norm(data.S_hop.to(torch.float32), dim=-1) < 1.0e-6
            selfconnect_index = torch.arange(feature_edge.shape[0], device=feature_node.device)[flag1*flag2]
            feature_edge[selfconnect_index] += feature_node
        else:
            feature_edge = torch.cat([feature_edge, feature_node], dim=0)

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
        if self.para.edge_include_sc:
            n1, n2 = data.edge_index_hop
        else:
            n1, n2 = data.edge_index_hop
            sc_node = torch.arange(data.num_nodes, dtype=n1.dtype, device=n1.device)
            n1 = torch.cat([n1, sc_node])
            n2 = torch.cat([n2, sc_node])
        num_edge = len(n1)

        GraphEdgeIndex_to_BlockEdgeIndex = torch.zeros(num_edge, dtype=self.para.intdtype, device=self.device)

        for atomtype_1 in range(num_atomtype):
            for atomtype_2 in range(num_atomtype):
                edge_index_seleced = torch.arange(num_edge, dtype=self.para.intdtype, device=self.device)[(data.AtomType[n1] == atomtype_1)*(data.AtomType[n2] == atomtype_2)]

                sum_orbit_1 = self.AtomType_OrbitalSum[atomtype_1]
                sum_orbit_2 = self.AtomType_OrbitalSum[atomtype_2]
                ham = torch.zeros((len(edge_index_seleced), sum_orbit_1, sum_orbit_2), device=self.device)

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
                GraphEdgeIndex_to_BlockEdgeIndex[edge_index_seleced] = torch.arange(edge_index_seleced.shape[0], dtype=self.para.intdtype, device=self.device)

        # symmetrize matrix
        if (hasattr(data, "num_graphs")) and (data.num_graphs != 1):
            edge_num = torch.tensor([data[i].num_edges for i in range(data.num_graphs)])
            offset = torch.cat([edge_num.new_zeros(1), torch.cumsum(edge_num, dim=0)[:-1]])
            if self.para.edge_include_sc:
                edge_inverse = torch.cat([data[i].edge_inverse+offset[i] for i in range(data.num_graphs)])
            else:
                edge_inverse = torch.cat([data[i].edge_inverse+offset[i] for i in range(data.num_graphs)]+[sc_node])
        else:
            if self.para.edge_include_sc:
                edge_inverse = data.edge_inverse
            else:
                edge_inverse = torch.cat([data.edge_inverse, sc_node])

        for atomtype_1 in range(num_atomtype):
            for atomtype_2 in range(atomtype_1, num_atomtype):
                edge_12_mask = (data.AtomType[n1] == atomtype_1)*(data.AtomType[n2] == atomtype_2)
                edge_12 = torch.arange(num_edge, dtype=self.para.intdtype, device=self.device)[edge_12_mask]
                edge_21 = edge_inverse[edge_12]
                edge_block21 = GraphEdgeIndex_to_BlockEdgeIndex[edge_21]

                index_HBlcok12 = atomtype_1*num_atomtype+atomtype_2
                index_HBlcok21 = atomtype_2*num_atomtype+atomtype_1
                hams[index_HBlcok12] = 0.5*(hams[index_HBlcok12]+hams[index_HBlcok21][edge_block21].transpose(-2, -1))
                if atomtype_1 != atomtype_2:
                    edge_21_mask = (data.AtomType[n1] == atomtype_2)*(data.AtomType[n2] == atomtype_1)
                    edge_21 = torch.arange(num_edge, dtype=self.para.intdtype, device=self.device)[edge_21_mask]
                    edge_12 = edge_inverse[edge_21]
                    edge_block12 = GraphEdgeIndex_to_BlockEdgeIndex[edge_12]
                    hams[index_HBlcok21] = hams[index_HBlcok12][edge_block12].transpose(-2, -1)

        return hams, GraphEdgeIndex_to_BlockEdgeIndex
