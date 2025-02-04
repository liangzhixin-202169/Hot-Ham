from e3nn import nn, o3
import torch


class OutputLinear(torch.nn.Module):
    """linear and rearrangement, each ir of irreps_out should be found in irreps_in"""

    def __init__(self, irreps_in: o3.Irreps, irreps_out: o3.Irreps):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.slices_in = irreps_in.slices()

        mask_outs, irreps_index = [], []
        for index, (_, (l_in, p_in)) in enumerate(self.irreps_in):
            mask_out = torch.zeros(self.irreps_out.dim, dtype=torch.bool)

            for (_, (l_out, p_out)), s in zip(self.irreps_out, self.irreps_out.slices()):
                if (l_out == l_in) and (p_out == p_in):
                    mask_out[s] = True

            if mask_out.sum().item() > 0:
                mask_outs.append(mask_out)
                irreps_index.append(index)

        mask_outs = torch.stack(mask_outs)
        irreps_index = torch.tensor(irreps_index)
        self.register_buffer("mask_outs", mask_outs)
        self.register_buffer("irreps_index", irreps_index)

        self.LinearList = torch.nn.ModuleList()
        for index, mask_out in zip(irreps_index, mask_outs):
            ir_in = self.irreps_in[index]

            l_in, p_in = ir_in[1]
            mul_out = int(mask_out.sum().item()/(2*l_in+1))
            ir_out = o3.Irreps([(mul_out, (l_in, p_in))])
            ir_in = o3.Irreps([(ir_in[0], ir_in[1])])

            lin = o3.Linear(ir_in, ir_out)
            self.LinearList.append(lin)

    def forward(self, x: torch.tensor):
        out = x.new_zeros(x.shape[0], self.irreps_out.dim)
        for index, mask_out, lin in zip(self.irreps_index, self.mask_outs, self.LinearList):
            slice_in = self.slices_in[index]
            out[:, mask_out] += lin(x[:, slice_in])
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
            node_mask = (data.AtomType == atomtype)
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
                bondtype = atomtype_1*self.num_atomtype+atomtype_2
                edge_mask = (data.AtomType[n1] == atomtype_1)*(data.AtomType[n2] == atomtype_2)
                out[edge_mask] = self.outputlinear[bondtype](x[edge_mask])
        return out


if __name__ == "__main__":
    lmax = 6
    irreps_in = o3.Irreps([(32, (l, (-1)**l)) for l in range(lmax+1)])+o3.Irreps([(32, (l, -1*(-1)**l)) for l in range(lmax+1)])
    irreps_out = o3.Irreps.spherical_harmonics(lmax)
    outputlinear = OutputLinear(irreps_in, irreps_out)
    a = 1
