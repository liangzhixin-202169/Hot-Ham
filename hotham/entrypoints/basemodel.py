import torch
from torch import nn
from typing import Union
import math
from e3nn import o3
from .parameters import Parameters
from ..modules.common import BasicInfo
from ..modules.te3nn.to3 import TO3_Irreps


def find_common_AM(AtomType_OrbitalIrrepsList: dict):
    from collections import Counter
    from functools import reduce

    count_list = [Counter(v) for v in AtomType_OrbitalIrrepsList.values()]
    union_counter = reduce(lambda x, y: x | y, count_list)
    ir_list = [[ir] * num for ir, num in union_counter.items()]
    return o3.Irreps(reduce(lambda x, y: x + y, ir_list)).sort().irreps


def find_EdgeIrreps_output(irs):
    ir_out = o3.Irreps(None)
    for mul1, (l1, p1) in irs:
        for mul2, (l2, p2) in irs:
            assert mul1 == mul2 == 1
            p = p1 * p2
            ir_out += o3.Irreps(list(map(lambda x: o3.Irrep(x, p), range(abs(l1 - l2), l1 + l2 + 1))))
    return ir_out


class BaseModel(nn.Module):
    def __init__(self, para: Union[dict, Parameters], **kwargs):
        super().__init__()
        self.para = para
        self.device = para.device

        basicinfo = BasicInfo(orbit=para.orbit, device=self.device, intdtype=para.torch_int_dtype)
        self.num_atomtype = basicinfo.n_type
        self.symbols = self.para.orbit.keys()
        self.spin = self.para.spin
        self.AtomSymbol_to_AtomNumber = basicinfo.AtomSymbol_to_AtomNumber
        self.AtomNumber_to_AtomSymbol = basicinfo.AtomNumber_to_AtomSymbol
        self.AtomNumber_to_AtomType = basicinfo.AtomNumber_to_AtomType
        self.AtomSymbol_to_AtomType = basicinfo.AtomSymbol_to_AtomType
        self.AtomType_to_AtomNumber = basicinfo.AtomType_to_AtomNumber
        self.AtomType_to_AtomSymbol = basicinfo.AtomType_to_AtomSymbol

        self.AMSymbol_to_AM = basicinfo.AMSymbol_to_AM
        self.AtomType_AMSymbolList = basicinfo.AtomType_AMSymbolList
        self.AtomType_AMList = basicinfo.AtomType_AMList
        self.AtomType_OrbitalIrrepsList = {k: list(map(lambda x: o3.Irrep(x, (-1) ** x), v)) for k, v in self.AtomType_AMList.items()}
        self.AtomType_OrbitalIrreps = {k: o3.Irreps(v).sort()[0].simplify() for k, v in self.AtomType_OrbitalIrrepsList.items()}
        self.AtomType_OrbitalNum = [[mul for mul, (l, p) in self.AtomType_OrbitalIrreps[i]] for i in sorted(self.AtomType_OrbitalIrreps)]
        self.AtomType_OrbitalSum = torch.tensor(
            list(map(lambda x: self.AtomType_OrbitalIrreps[x].dim, sorted(self.AtomType_OrbitalIrreps))), dtype=self.para.torch_int_dtype
        ).to(self.device)
        self.AtomSymbol_to_OrbitlSum = basicinfo.AtomSymbol_to_OrbitlSum
        self.AtomSymbol_to_OrbitalNum = {self.AtomType_to_AtomSymbol[atomtype]: obnum for atomtype, obnum in enumerate(self.AtomType_OrbitalNum)}
        self.AtomSymbol_to_AMList = basicinfo.AtomSymbol_to_AMList

        self.CommonOrbitalIrreps = find_common_AM(self.AtomType_OrbitalIrrepsList)
        self.CommonAM = self.CommonOrbitalIrreps.ls
        self.find_cg(self.CommonOrbitalIrreps)
        self.find_final_irreps(self.CommonOrbitalIrreps)
        self.find_spin_base()

        #########################
        self.irreps_node_attr = o3.Irreps([(len(self.symbols), (0, 1))])
        self.irreps_edge_attr = o3.Irreps.spherical_harmonics(lmax=4)
        #########################

        # self.edge_irreps_output = find_EdgeIrreps_output(self.CommonOrbitalIrreps)

    def find_cg(self, irs):
        # l_list = sorted(set(irs.ls))
        # cg = [[[] for j in range(max(l_list)+1)] for i in range(max(l_list)+1)]
        # for l1 in l_list:
        #     for l2 in l_list:
        #         for l3 in range(abs(l1-l2), l1+l2+1):
        #             cg[l1][l2].append(o3.wigner_3j(l1, l2, l3))
        #         cg[l1][l2] = torch.cat(cg[l1][l2], dim=-1).to(device=self.device)
        l_list = sorted(irs.ls)
        edge_irreps_output = o3.Irreps(None)
        if self.spin:
            dtype_cg = self.para.torch_complex_dtype
        else:
            dtype_cg = self.para.torch_float_dtype
        for l1 in l_list:
            for l2 in l_list:
                # l1*l2 -> l3_list
                cg_l1_l2_l3s_key = f"cg_{l1}_{l2}_l3s"
                if not hasattr(self, cg_l1_l2_l3s_key):
                    cgs_l1_l2_l3s = []
                    for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                        cgs_l1_l2_l3s.append(o3.wigner_3j(l1, l2, l3, dtype=self.para.torch_float_dtype).to(dtype=dtype_cg))
                    self.register_buffer(cg_l1_l2_l3s_key, torch.cat(cgs_l1_l2_l3s, dim=-1))

                if self.spin:
                    # l3_list*spin0 -> l4_list
                    cg_l3s_spin0_l4s_key = f"cg_{l1}_{l2}_l3s_spin0_l4s"
                    if not hasattr(self, cg_l3s_spin0_l4s_key):
                        cgs_l3_spin0_l4s = []
                        for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                            cgs_l3_spin0_l4s.append(o3.wigner_3j(l3, 0, l3, dtype=self.para.torch_float_dtype).to(dtype=dtype_cg).squeeze(dim=1))
                        cgs_l3_spin0_l4s = torch.block_diag(*cgs_l3_spin0_l4s)
                        self.register_buffer(cg_l3s_spin0_l4s_key, cgs_l3_spin0_l4s.unsqueeze(1))

                    # l3_list*spin1 -> l4_list
                    for l3 in range(abs(l1 - l2), l1 + l2 + 1):
                        cg_l3_spin1_l4s_key = f"cg_{l3}_spin1_l4s"
                        if not hasattr(self, cg_l3_spin1_l4s_key):
                            cgs_l3_spin1_l4s = []
                            for l4 in range(abs(l3 - 1), l3 + 2):
                                cgs_l3_spin1_l4s.append(o3.wigner_3j(l3, 1, l4, dtype=self.para.torch_float_dtype).to(dtype=dtype_cg))
                            self.register_buffer(cg_l3_spin1_l4s_key, torch.cat(cgs_l3_spin1_l4s, dim=-1))

    def find_final_irreps(self, irs):
        edge_irreps_output = o3.Irreps(None)
        output_slice = {}
        offset = 0
        time_parities = []
        for index1, (mul1, (l1, p1)) in enumerate(irs):
            for index2, (mul2, (l2, p2)) in enumerate(irs):
                assert mul1 == mul2 == 1
                block_l1_l2 = o3.Irreps([(1, (l3, p1 * p2)) for l3 in range(abs(l1 - l2), l1 + l2 + 1)])
                if not self.spin:
                    # l1*l2 -> l3_list
                    edge_irreps_output += block_l1_l2
                    # l3_list's slice in edge_irreps_output
                    slice_name = ((index1, irs[index1].ir), (index2, irs[index2].ir))
                    output_slice[slice_name] = ((l1, l2), (offset, offset + block_l1_l2.dim))
                    offset += block_l1_l2.dim

                if self.spin:
                    # for spin==0
                    # l3_list*spin0 -> l4_list
                    edge_irreps_output += block_l1_l2
                    time_parities += [1] * len(block_l1_l2)
                    # l4_list's slice in edge_irreps_output
                    slice_name = ((index1, irs[index1].ir), (index2, irs[index2].ir), "spin0")
                    output_slice[slice_name] = ((l1, l2), (offset, offset + block_l1_l2.dim))
                    offset += block_l1_l2.dim

                    # for spin==1
                    # # l3*spin1 -> l4_list
                    for index3, (mul3, (l3, p3)) in enumerate(block_l1_l2):
                        block_l3_spin1 = o3.Irreps([(1, (l4, p3)) for l4 in range(abs(l3 - 1), l3 + 2)])
                        edge_irreps_output += block_l3_spin1
                        time_parities += [-1] * len(block_l3_spin1)
                        # l4_list's slice in edge_irreps_output
                        slice_name = ((index1, irs[index1].ir), (index2, irs[index2].ir), (index3, block_l1_l2[index3].ir), "spin1")
                        output_slice[slice_name] = ((l1, l2, l3), (offset, offset + block_l3_spin1.dim))
                        offset += block_l3_spin1.dim

        self.edge_irreps_output = edge_irreps_output
        self.time_parities = time_parities
        if self.spin:
            self.edge_to3_irreps_output_real = TO3_Irreps(
                [(mul, (l, p, t_p)) for (mul, (l, p)), t_p in zip(self.edge_irreps_output, self.time_parities)]
            )
            self.edge_to3_irreps_output_imag = TO3_Irreps([(mul, (l, p, -t_p)) for (mul, (l, p, t_p)) in self.edge_to3_irreps_output_real])
            self.edge_irreps_output += edge_irreps_output
            self.edge_to3_irreps_output = self.edge_to3_irreps_output_real + self.edge_to3_irreps_output_imag
            # self.time_parities += [-t_p for t_p in time_parities]
            # self.edge_to3_irreps_output = TO3_Irreps([(mul, (l, p, t_p)) for (mul, (l, p)), t_p in zip(self.edge_irreps_output, self.time_parities)])
        self.output_slice = output_slice

    def find_spin_base(self):
        oyzx2spin = torch.tensor([[1, 0, 1, 0], [0, -1j, 0, 1], [0, 1j, 0, 1], [1, 0, -1, 0]], dtype=self.para.torch_complex_dtype) / (2**0.5)
        self.register_buffer("oyzx2spin", oyzx2spin)

    def spin_block_reshape(self, x):
        n, _, dim0, dim1 = x.shape
        x = x.view(n, 2, 2, dim0, dim1)
        x = x.permute(0, 1, 3, 2, 4)
        x = x.contiguous().view(n, 2 * dim0, 2 * dim1)
        return x

    def DirectSum_to_DirectProduct(self, feature_node, feature_edge, data):
        if self.para.edge_include_sc:
            flag1 = data.edge_index_hop[0] == data.edge_index_hop[1]
            flag2 = torch.norm(data.S_hop.to(self.para.torch_float_dtype), dim=-1) < 1.0e-6
            selfconnect_index = torch.arange(feature_edge.shape[0], dtype=self.para.torch_int_dtype, device=feature_node.device)[flag1 * flag2]
            selfconnect_node = data.edge_index_hop[0, selfconnect_index]
            feature_edge[selfconnect_index] += feature_node[selfconnect_node]
        else:
            feature_edge = torch.cat([feature_edge, feature_node], dim=0)

        H_l1l2_blocks = []
        # H_l1l2_blocks = {(index1,irs1.ir):{(index2,irs2.ir):None for index2,irs2 in enumerate(self.CommonOrbitalIrreps)} for index1,irs1 in enumerate(self.CommonOrbitalIrreps)}
        offset = 0
        # if self.spin:
        #     n0, n1 = feature_edge.shape
        #     mid = n1 // 2
        #     feature_edge = feature_edge[:, :mid] + 1.0j * feature_edge[:, mid:]

        for index1, (mul1, (l1, p1)) in enumerate(self.CommonOrbitalIrreps):
            ir1 = self.CommonOrbitalIrreps[index1].ir
            for index2, (mul2, (l2, p2)) in enumerate(self.CommonOrbitalIrreps):
                ir2 = self.CommonOrbitalIrreps[index2].ir

                if not self.spin:
                    # l3_list -> l1*l2
                    slice_key = ((index1, ir1), (index2, ir2))
                    _, (offset_start, offset_end) = self.output_slice[slice_key]
                    sub_feature_edge = feature_edge[:, offset_start:offset_end]
                    cg_l1_l2_l3s = getattr(self, f"cg_{l1}_{l2}_l3s")
                    block = torch.einsum("nk,ijk->nij", sub_feature_edge, cg_l1_l2_l3s)
                    # H_l1l2_blocks[(index1, ir1)][(index2, ir2)] = block
                    H_l1l2_blocks.append(block)
                else:
                    l3_irreps = o3.Irreps([(1, (l3, p1 * p2)) for l3 in range(abs(l1 - l2), l1 + l2 + 1)])

                    # l4_list -> l3_list*spin0
                    slice_key = ((index1, ir1), (index2, ir2), "spin0")
                    _, (offset_start, offset_end) = self.output_slice[slice_key]
                    sub_feature_edge_spin0 = feature_edge[:, offset_start:offset_end]

                    cg_l3s_spin0_l4s = getattr(self, f"cg_{l1}_{l2}_l3s_spin0_l4s")
                    block_l3s_spin0 = torch.einsum("nk,ijk->nij", sub_feature_edge_spin0, cg_l3s_spin0_l4s)

                    # l4_list -> l3*spin1
                    blocks_l3_spin1 = []
                    for index3, (_, (l3, p3)) in enumerate(l3_irreps):
                        ir3 = l3_irreps[index3].ir

                        slice_key = ((index1, ir1), (index2, ir2), (index3, ir3), "spin1")
                        _, (offset_start, offset_end) = self.output_slice[slice_key]
                        sub_feature_edge_spin1 = feature_edge[:, offset_start:offset_end]

                        cg_l3_spin1_l4s = getattr(self, f"cg_{l3}_spin1_l4s")
                        block_l3_spin1 = torch.einsum("nk,ijk->nij", sub_feature_edge_spin1, cg_l3_spin1_l4s)
                        blocks_l3_spin1.append(block_l3_spin1)

                    # (1) stack l3*spin1 as l3_list*spin1
                    block_l3s_spin1 = torch.cat(blocks_l3_spin1, dim=-2)
                    # (2) stack l3_list*spin0 and l3_list*spin1 as l3_list*(spin0+spin1)
                    block_l3s_spins = torch.cat([block_l3s_spin0, block_l3s_spin1], dim=-1)
                    # (3) convert l3_list*(spin0+spin1) to (l1*l2)*(spin_half*spin_half)
                    cg_l1_l2_l3s = getattr(self, f"cg_{l1}_{l2}_l3s")
                    H_l1l2_block = torch.einsum("ijk,nkl,lm->nmij", cg_l1_l2_l3s, block_l3s_spins, self.oyzx2spin)
                    # # (4) convert (l1*l2)*(spin_half*spin_half) to (l1*spin_half)*(l2*spin_half)
                    # H_l1l2_block = self.spin_block_reshape(H_l1l2_block)
                    # H_l1l2_blocks[(index1, ir1)][(index2, ir2)] = H_l1l2_block
                    H_l1l2_blocks.append(H_l1l2_block)

        return H_l1l2_blocks

    def find_Hblock(self, DirectProduct, data):
        # SqrtLength = int(math.sqrt(len(DirectProduct)))
        SqrtLength = len(self.CommonOrbitalIrreps)
        hams = {s0: {s1: None for s1 in self.AtomSymbol_to_AtomType} for s0 in self.AtomSymbol_to_AtomType}

        num_atomtype = self.num_atomtype
        if self.para.edge_include_sc:
            n1, n2 = data.edge_index_hop
        else:
            n1, n2 = data.edge_index_hop
            sc_node = torch.arange(data.num_nodes, dtype=n1.dtype, device=n1.device)
            n1 = torch.cat([n1, sc_node])
            n2 = torch.cat([n2, sc_node])
        num_edge = len(n1)

        GraphEdgeIndex_to_BlockEdgeIndex = torch.zeros(num_edge, dtype=self.para.torch_int_dtype, device=self.device)
        edge_order = torch.arange(num_edge, dtype=self.para.torch_int_dtype, device=self.device)

        unique_atomtypes = torch.unique(data.AtomType)
        for atomtype_1 in range(self.num_atomtype):
            if atomtype_1 not in unique_atomtypes:
                continue
            atomsymbol_1 = self.AtomType_to_AtomSymbol[atomtype_1]
            for atomtype_2 in range(self.num_atomtype):
                if atomtype_2 not in unique_atomtypes:
                    continue
                atomsymbol_2 = self.AtomType_to_AtomSymbol[atomtype_2]
                edge_index_seleced = edge_order[(data.AtomType[n1] == atomtype_1) * (data.AtomType[n2] == atomtype_2)]

                sum_orbit_1 = self.AtomSymbol_to_OrbitlSum[atomsymbol_1]
                sum_orbit_2 = self.AtomSymbol_to_OrbitlSum[atomsymbol_2]
                if not self.spin:
                    ham = torch.zeros((len(edge_index_seleced), sum_orbit_1, sum_orbit_2), device=self.device, dtype=self.para.torch_float_dtype)
                else:
                    ham = torch.zeros((len(edge_index_seleced), 4, sum_orbit_1, sum_orbit_2), device=self.device, dtype=self.para.torch_complex_dtype)

                numlist_orbit_1 = self.AtomSymbol_to_OrbitalNum[atomsymbol_1].copy()
                block_offset, row_offset1 = 0, 0

                for l1 in self.CommonAM:
                    if len(numlist_orbit_1) < l1 + 1 or numlist_orbit_1[l1] == 0:
                        block_offset += SqrtLength
                        continue
                    numlist_orbit_1[l1] -= 1

                    row_offset2 = 2 * l1 + 1
                    col_offset1 = 0
                    numlist_orbit_2 = self.AtomSymbol_to_OrbitalNum[atomsymbol_2].copy()

                    for l2 in self.CommonAM:
                        if len(numlist_orbit_2) < l2 + 1 or numlist_orbit_2[l2] == 0:
                            block_offset += 1
                            continue
                        numlist_orbit_2[l2] -= 1

                        col_offset2 = 2 * l2 + 1

                        block = DirectProduct[block_offset][edge_index_seleced]
                        ham[..., row_offset1 : row_offset1 + row_offset2, col_offset1 : col_offset1 + col_offset2] = block

                        block_offset += 1
                        col_offset1 += col_offset2

                    row_offset1 += row_offset2

                hams[atomsymbol_1][atomsymbol_2] = ham
                GraphEdgeIndex_to_BlockEdgeIndex[edge_index_seleced] = torch.arange(
                    edge_index_seleced.shape[0], dtype=self.para.torch_int_dtype, device=self.device
                )
        # spin = True: (E, 4, n, m)
        # # spin=True: (E, 4, n, m) -> (E, 2n, 2m)
        # if self.spin:
        #     for atomsymbol_1 in hams:
        #         for atomsymbol_2 in hams[atomsymbol_1]:
        #             hams[atomsymbol_1][atomsymbol_2] = self.spin_block_reshape(hams[atomsymbol_1][atomsymbol_2])
        # spin=False: (E, n, m) -> (E, 1, n, m)
        if not self.spin:
            for symbol1 in hams:
                for symbol2 in hams[symbol1]:
                    hams[symbol1][symbol2] = hams[symbol1][symbol2].unsqueeze(1)

        # symmetrize matrix for (E,2n,2m)
        # if (hasattr(data, "num_graphs")) and (data.num_graphs != 1):
        #     edge_num = torch.tensor([data[i].num_edges for i in range(data.num_graphs)])
        #     offset = torch.cat([edge_num.new_zeros(1), torch.cumsum(edge_num, dim=0)[:-1]])
        #     if self.para.edge_include_sc:
        #         edge_inverse = torch.cat([data[i].edge_inverse + offset[i] for i in range(data.num_graphs)])
        #     else:
        #         edge_inverse = torch.cat([data[i].edge_inverse + offset[i] for i in range(data.num_graphs)] + [sc_node])
        # else:
        #     if self.para.edge_include_sc:
        #         edge_inverse = data.edge_inverse
        #     else:
        #         edge_inverse = torch.cat([data.edge_inverse, sc_node])

        # for atomtype_1 in range(self.num_atomtype):
        #     if atomtype_1 not in unique_atomtypes:
        #         continue
        #     atomsymbol_1 = self.AtomType_to_AtomSymbol[atomtype_1]
        #     for atomtype_2 in range(atomtype_1, self.num_atomtype):
        #         if atomtype_2 not in unique_atomtypes:
        #             continue
        #         atomsymbol_2 = self.AtomType_to_AtomSymbol[atomtype_2]
        #         edge_12_mask = (data.AtomType[n1] == atomtype_1) * (data.AtomType[n2] == atomtype_2)
        #         edge_12 = edge_order[edge_12_mask]
        #         edge_21 = edge_inverse[edge_12]
        #         edge_block21 = GraphEdgeIndex_to_BlockEdgeIndex[edge_21]

        #         if not self.spin:
        #             hams[atomsymbol_1][atomsymbol_2] = 0.5 * (
        #                 hams[atomsymbol_1][atomsymbol_2] + hams[atomsymbol_2][atomsymbol_1][edge_block21].transpose(-2, -1)
        #             )
        #         else:
        #             hams[atomsymbol_1][atomsymbol_2] = 0.5 * (
        #                 hams[atomsymbol_1][atomsymbol_2] + hams[atomsymbol_2][atomsymbol_1][edge_block21].transpose(-2, -1).conj()
        #             )
        #         if atomtype_1 != atomtype_2:
        #             edge_21_mask = (data.AtomType[n1] == atomtype_2) * (data.AtomType[n2] == atomtype_1)
        #             edge_21 = edge_order[edge_21_mask]
        #             edge_12 = edge_inverse[edge_21]
        #             edge_block12 = GraphEdgeIndex_to_BlockEdgeIndex[edge_12]
        #             if not self.spin:
        #                 hams[atomsymbol_2][atomsymbol_1] = hams[atomsymbol_1][atomsymbol_2][edge_block12].transpose(-2, -1)
        #             else:
        #                 hams[atomsymbol_2][atomsymbol_1] = hams[atomsymbol_1][atomsymbol_2][edge_block12].transpose(-2, -1).conj()
        
        # symmetrize matrix for (E,4,n,m)
        if (hasattr(data, "num_graphs")) and (data.num_graphs != 1):
            edge_num = torch.tensor([data[i].num_edges for i in range(data.num_graphs)], dtype=self.para.torch_int_dtype, device=self.device)
            offset = torch.cat([edge_num.new_zeros(1), torch.cumsum(edge_num, dim=0)[:-1]])
            if self.para.edge_include_sc:
                edge_inverse = torch.cat([data[i].edge_inverse + offset[i] for i in range(data.num_graphs)])
            else:
                edge_inverse = torch.cat([data[i].edge_inverse + offset[i] for i in range(data.num_graphs)] + [sc_node])
        else:
            if self.para.edge_include_sc:
                edge_inverse = data.edge_inverse
            else:
                edge_inverse = torch.cat([data.edge_inverse, sc_node])

        for atomtype_1 in range(self.num_atomtype):
            if atomtype_1 not in unique_atomtypes:
                continue
            atomsymbol_1 = self.AtomType_to_AtomSymbol[atomtype_1]
            for atomtype_2 in range(atomtype_1, self.num_atomtype):
                if atomtype_2 not in unique_atomtypes:
                    continue
                atomsymbol_2 = self.AtomType_to_AtomSymbol[atomtype_2]
                edge_12_mask = (data.AtomType[n1] == atomtype_1) * (data.AtomType[n2] == atomtype_2)
                edge_12 = edge_order[edge_12_mask]
                edge_21 = edge_inverse[edge_12]
                edge_block21 = GraphEdgeIndex_to_BlockEdgeIndex[edge_21]

                if not self.spin:
                    hams[atomsymbol_1][atomsymbol_2] = 0.5 * (
                        hams[atomsymbol_1][atomsymbol_2] + hams[atomsymbol_2][atomsymbol_1][edge_block21].transpose(-2, -1)
                    )
                else:
                    hams[atomsymbol_1][atomsymbol_2] = 0.5 * (
                        hams[atomsymbol_1][atomsymbol_2] + hams[atomsymbol_2][atomsymbol_1][edge_block21][:, [0, 2, 1, 3]].transpose(-2, -1).conj()
                    )
                if atomtype_1 != atomtype_2:
                    edge_21_mask = (data.AtomType[n1] == atomtype_2) * (data.AtomType[n2] == atomtype_1)
                    edge_21 = edge_order[edge_21_mask]
                    edge_12 = edge_inverse[edge_21]
                    edge_block12 = GraphEdgeIndex_to_BlockEdgeIndex[edge_12]
                    if not self.spin:
                        hams[atomsymbol_2][atomsymbol_1] = hams[atomsymbol_1][atomsymbol_2][edge_block12].transpose(-2, -1)
                    else:
                        hams[atomsymbol_2][atomsymbol_1] = hams[atomsymbol_1][atomsymbol_2][edge_block12][:, [0, 2, 1, 3]].transpose(-2, -1).conj()

        return hams
