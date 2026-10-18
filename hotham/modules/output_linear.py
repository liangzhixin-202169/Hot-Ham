from e3nn import nn, o3
from e3nn.o3 import Irreps
import torch
from typing import Union, Optional
from collections import defaultdict
from .te3nn.to3 import TO3_Irreps, TO3_Linear

# class OutputLinear(torch.nn.Module):
#     """linear and rearrangement, each ir of irreps_out should be found in irreps_in"""

#     def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
#         super().__init__()

#         self.irreps_in = irreps_in
#         self.irreps_out = irreps_out
#         self.slices_in = irreps_in.slices()

#         mask_outs, irreps_index = [], []
#         for index, (_, (l_in, p_in)) in enumerate(self.irreps_in):
#             mask_out = torch.zeros(self.irreps_out.dim, dtype=torch.bool)

#             for (_, (l_out, p_out)), s in zip(self.irreps_out, self.irreps_out.slices()):
#                 if (l_out == l_in) and (p_out == p_in):
#                     mask_out[s] = True

#             if mask_out.sum().item() > 0:
#                 mask_outs.append(mask_out)
#                 irreps_index.append(index)

#         mask_outs = torch.stack(mask_outs)
#         irreps_index = torch.tensor(irreps_index)
#         self.register_buffer("mask_outs", mask_outs)
#         self.register_buffer("irreps_index", irreps_index)

#         self.LinearList = torch.nn.ModuleList()
#         for index, mask_out in zip(irreps_index, mask_outs):
#             ir_in = self.irreps_in[index]

#             l_in, p_in = ir_in[1]
#             mul_out = int(mask_out.sum().item()/(2*l_in+1))
#             ir_out = o3.Irreps([(mul_out, (l_in, p_in))])
#             ir_in = o3.Irreps([(ir_in[0], ir_in[1])])

#             lin = o3.Linear(ir_in, ir_out)
#             self.LinearList.append(lin)


#     def forward(self, x: torch.tensor):
#         out = x.new_zeros(x.shape[0], self.irreps_out.dim)
#         for index, mask_out, lin in zip(self.irreps_index, self.mask_outs, self.LinearList):
#             slice_in = self.slices_in[index]
#             out[:, mask_out] += lin(x[:, slice_in])
#         return out


# class OutputLinear(torch.nn.Module):
#     """
#     linear and rearrangement, each ir of irreps_out should be found in irreps_in,
#     irreps_in has been sorted and simplified.
#     """

#     def __init__(self, irreps_in: Union[Irreps, TO3_Irreps], irreps_out: Union[Irreps, TO3_Irreps]):
#         super().__init__()

#         self.irreps_in = irreps_in
#         self.irreps_out = irreps_out
#         self.slices_in = irreps_in.slices()

#         if isinstance(irreps_in, Irreps):
#             irreps_class = Irreps
#             linear_class = o3.Linear
#         elif isinstance(irreps_in, TO3_Irreps):
#             irreps_class = TO3_Irreps
#             linear_class = TO3_Linear

#         mask_outs, irreps_index = [], []
#         for index, (_, ir_in) in enumerate(self.irreps_in):
#             mask_out = torch.zeros(self.irreps_out.dim, dtype=torch.bool)

#             for (_, ir_out), s in zip(self.irreps_out, self.irreps_out.slices()):
#                 if ir_in == ir_out:
#                     mask_out[s] = True

#             assert mask_out.sum().item() > 0, f"ir_in={ir_in} not found in irreps_out={self.irreps_out}"
#             mask_outs.append(mask_out)
#             irreps_index.append(index)

#         mask_outs = torch.stack(mask_outs)
#         irreps_index = torch.tensor(irreps_index)
#         self.register_buffer("mask_outs", mask_outs)
#         self.register_buffer("irreps_index", irreps_index)

#         self.LinearList = torch.nn.ModuleList()
#         for index, mask_out in zip(irreps_index, mask_outs):
#             ir_in = self.irreps_in[index]

#             mul_out = int(mask_out.sum().item() / (2 * ir_in.ir.l + 1))
#             ir_out = irreps_class([(mul_out, ir_in.ir)])
#             ir_in = irreps_class([ir_in])

#             lin = linear_class(ir_in, ir_out)
#             self.LinearList.append(lin)

#     def forward(self, x: torch.tensor):
#         out = x.new_zeros(x.shape[0], self.irreps_out.dim)
#         for index, mask_out, lin in zip(self.irreps_index, self.mask_outs, self.LinearList):
#             slice_in = self.slices_in[index]
#             out[:, mask_out] += lin(x[:, slice_in])
#         return out


class OutputLinear(torch.nn.Module):

    def __init__(self, irreps_in: Union[Irreps, TO3_Irreps], irreps_out: Union[Irreps, TO3_Irreps], shared_weights: Optional[bool] = True, intdtype=torch.int64):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.shared_weights = shared_weights

        if isinstance(irreps_in, Irreps):
            irreps_class = Irreps
            linear_class = o3.Linear
        elif isinstance(irreps_in, TO3_Irreps):
            irreps_class = TO3_Irreps
            linear_class = TO3_Linear
        else:
            raise TypeError()
        assert isinstance(irreps_out, irreps_class)

        unique_irreps_in = sorted(set([ir for _, ir in self.irreps_in]))
        unique_irreps_out = sorted(set([ir for _, ir in self.irreps_out]))

        selected_irs = []
        ir2mul_in = defaultdict(int)
        for unique_ir_in in unique_irreps_in:
            if unique_ir_in not in unique_irreps_out:
                continue
            selected_irs.append(unique_ir_in)
            offset = 0
            irs_idx = []
            for index, (mul, ir_in) in enumerate(self.irreps_in):
                start, end = offset, offset + mul * ir_in.dim
                if ir_in == unique_ir_in:
                    local_idx = list(range(start, end))
                    irs_idx.extend(local_idx)
                    ir2mul_in[ir_in] += mul
                offset = end
            self.register_buffer(f"{unique_ir_in}_input_idx", torch.tensor(irs_idx, dtype=intdtype))
        self.ir2mul_in = ir2mul_in

        ir2mul_out = defaultdict(int)
        for unique_ir_out in selected_irs:
            offset = 0
            irs_idx = []
            for index, (mul, ir_out) in enumerate(self.irreps_out):
                start, end = offset, offset + mul * ir_out.dim
                if ir_out == unique_ir_out:
                    local_idx = list(range(start, end))
                    irs_idx.extend(local_idx)
                    ir2mul_out[ir_out] += mul
                offset = end
            self.register_buffer(f"{unique_ir_out}_out_idx", torch.tensor(irs_idx, dtype=intdtype))
        self.ir2mul_out = ir2mul_out

        self.selected_irs = selected_irs
        self.LinearList = torch.nn.ModuleList()
        for index in range(len(self.selected_irs)):
            ir = self.selected_irs[index]
            mul_in = self.ir2mul_in[ir]
            irs_in = irreps_class([(mul_in, ir)])
            mul_out = self.ir2mul_out[ir]
            irs_out = irreps_class([(mul_out, ir)])
            lin = linear_class(irreps_in=irs_in, irreps_out=irs_out, shared_weights=self.shared_weights)
            self.LinearList.append(lin)

        # if not self.shared_weights:
        #     weights_slice = [0]
        #     for lin in self.LinearList:
        #         weights_slice.append(weights_slice[-1] + lin.weight_numel)
        #     self.weights_slice = weights_slice
        #     self.weight_numel = weights_slice[-1]

        weights_slice = [0]
        weight_numel = 0
        for lin in self.LinearList:
            if not self.shared_weights:
                weights_slice.append(weights_slice[-1] + lin.weight_numel)
            weight_numel += lin.weight_numel
        self.weights_slice = weights_slice
        self.weight_numel = weight_numel

    def forward(self, x: torch.tensor, weights=None):
        out = x.new_zeros(x.shape[0], self.irreps_out.dim)
        for index, ir in enumerate(self.selected_irs):
            input_idx = getattr(self, f"{ir}_input_idx")
            out_idx = getattr(self, f"{ir}_out_idx")
            lin = self.LinearList[index]
            if not self.shared_weights:
                sub_weights = weights[:, self.weights_slice[index] : self.weights_slice[index + 1]]
                out[:, out_idx] = lin(x[:, input_idx], sub_weights)
            else:
                out[:, out_idx] = lin(x[:, input_idx])
        return out


class OutputLinear_AtomType(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, num_atomtype: int):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.num_atomtype = num_atomtype

        self.outputlinear = torch.nn.ModuleList([])
        for _ in range(num_atomtype):
            self.outputlinear.append(OutputLinear(irreps_in, irreps_out))

    def forward(self, x: torch.tensor, data):
        out = x.new_zeros(size=(x.shape[0], self.irreps_out.dim))
        for atomtype in range(self.num_atomtype):
            node_mask = data.AtomType == atomtype
            out[node_mask] = self.outputlinear[atomtype](x[node_mask])
        return out


class OutputLinear_BondType(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps, num_atomtype: int):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.num_atomtype = num_atomtype

        self.outputlinear = torch.nn.ModuleList([])
        for _ in range(num_atomtype):
            for _ in range(num_atomtype):
                self.outputlinear.append(OutputLinear(irreps_in, irreps_out))

    def forward(self, x: torch.tensor, data):
        n1, n2 = data.edge_index_hop
        out = x.new_zeros(size=(x.shape[0], self.irreps_out.dim))
        for atomtype_1 in range(self.num_atomtype):
            for atomtype_2 in range(self.num_atomtype):
                bondtype = atomtype_1 * self.num_atomtype + atomtype_2
                edge_mask = (data.AtomType[n1] == atomtype_1) * (data.AtomType[n2] == atomtype_2)
                out[edge_mask] = self.outputlinear[bondtype](x[edge_mask])
        return out


if __name__ == "__main__":
    lmax = 6
    irreps_in = o3.Irreps([(32, (l, (-1) ** l)) for l in range(lmax + 1)]) + o3.Irreps([(32, (l, -1 * (-1) ** l)) for l in range(lmax + 1)])
    irreps_out = o3.Irreps.spherical_harmonics(lmax)
    outputlinear = OutputLinear(irreps_in, irreps_out)
    a = 1
