import torch
from torch_geometric.data import Data
import numpy as np
import os
from ase.units import Rydberg, Bohr
from .base_calc import Base_Calc
from ..data.DatasetPreprocess import DatasetPrepocess


class ToPyatb(Base_Calc):
    def __init__(self, para: dict, **kwargs):
        super().__init__(para, **kwargs)

        self.datapreprocess = DatasetPrepocess(self.para)
        self.trainloader = self.datapreprocess.trainset_loader

    def get_wigner_D(self, order):
        """
        D @ Y_wiki == Y_abacus
        """
        D = [
            torch.tensor([[1.0]]),
            torch.tensor([[0, 1, 0],
                          [0, 0, -1],
                          [-1, 0, 0]]),
            torch.tensor([[0, 0, 1, 0, 0],
                          [0, 0, 0, -1, 0],
                          [0, -1, 0, 0, 0],
                          [0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0]]),
            torch.tensor([[0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, -1, 0, 0],
                          [0, 0, -1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, -1],
                          [-1, 0, 0, 0, 0, 0, 0]])
        ]

        if isinstance(order, int):
            order = [order]

        DirectSum = torch.block_diag(*[D[l] for l in order])
        return DirectSum

    def rotate2abacus(self, block: dict, AtomType: torch.Tensor):
        num_atomtype = self.model.num_atomtype
        unique_atomtypes = torch.unique(AtomType)
        AtomType_to_AtomSymbol = self.model.AtomType_to_AtomSymbol
        AtomSymbol_to_AMList = self.model.AtomSymbol_to_AMList

        for atomtype_1 in range(num_atomtype):
            if atomtype_1 not in unique_atomtypes:
                continue
            atomsymbol_1 = AtomType_to_AtomSymbol[atomtype_1]
            winger_D_1 = self.get_wigner_D(AtomSymbol_to_AMList[atomsymbol_1])

            for atomtype_2 in range(num_atomtype):
                if atomtype_2 not in unique_atomtypes:
                    continue
                atomsymbol_2 = AtomType_to_AtomSymbol[atomtype_2]
                winger_D_2 = self.get_wigner_D(AtomSymbol_to_AMList[atomsymbol_2])

                block[atomsymbol_1][atomsymbol_2] = block[atomsymbol_1][atomsymbol_2].to(winger_D_1.dtype)
                shape = block[atomsymbol_1][atomsymbol_2].shape
                block[atomsymbol_1][atomsymbol_2] = block[atomsymbol_1][atomsymbol_2].reshape((-1,)+shape[-2:])
                block[atomsymbol_1][atomsymbol_2] = torch.einsum("ij,zjk,kl->zil",
                                                                 winger_D_1,
                                                                 block[atomsymbol_1][atomsymbol_2],
                                                                 winger_D_2.T)
                block[atomsymbol_1][atomsymbol_2] = block[atomsymbol_1][atomsymbol_2].reshape(shape)
        return block

    def block_r(self, block: dict, data: Data):
        num_atomtype = self.model.num_atomtype
        AtomType = data.AtomType
        unique_atomtypes = torch.unique(AtomType)
        AtomType_to_AtomSymbol = self.model.AtomType_to_AtomSymbol
        index_edge = data.edge_index_hop
        num_edge = index_edge.shape[1]
        unique_cell_shift = data.unique_cell_shift
        cell_shift_index = data.cell_shift_index
        n_cell = len(unique_cell_shift)
        offset = data.offset
        AtomType_OrbitalSum = self.model.AtomType_OrbitalSum
        dim_matrix = sum([AtomType_OrbitalSum[atomtype] for atomtype in AtomType])
        Mr = torch.zeros((n_cell, dim_matrix, dim_matrix), dtype=torch.float32)
        EdgeNumber = torch.arange(num_edge, dtype=torch.long)

        for atomntype_1 in range(num_atomtype):
            if atomntype_1 not in unique_atomtypes:
                continue
            atomsymbol_1 = AtomType_to_AtomSymbol[atomntype_1]

            for atomntype_2 in range(num_atomtype):
                if atomntype_2 not in unique_atomtypes:
                    continue
                atomsymbol_2 = AtomType_to_AtomSymbol[atomntype_2]

                mr = block[atomsymbol_1][atomsymbol_2]
                mr = mr.squeeze(1)
                mask_12 = (AtomType[index_edge[0, :]] == atomntype_1)*(AtomType[index_edge[1, :]] == atomntype_2)
                edge_12 = EdgeNumber[mask_12]
                sub_cell_shift_index = cell_shift_index[edge_12]
                offset0 = offset[index_edge[0, edge_12]]
                offset1 = offset[index_edge[1, edge_12]]
                dim0, dim1 = mr.shape[-2:]
                for i in range(dim0):
                    for j in range(dim1):
                        Mr[sub_cell_shift_index, offset0+i, offset1+j] = mr[:, i, j]
        return Mr, dim_matrix

    def calculation(self, data: Data, structure_idx: int):
        AtomType = data.AtomType
        unique_cell_shift = data.unique_cell_shift.numpy()
        n_cell = len(unique_cell_shift)

        if "h_ref" in self.para.write:
            h_ref = self.rotate2abacus(data.HR, AtomType)
            h_ref, dim_matrix = self.block_r(h_ref, data)
            with open("h_ref.csr", "w") as f:
                f.write(f"STEP: 0\n")
                f.write(f"Matrix Dimension of H(R): {dim_matrix}\n")
                f.write(f"Matrix number of H(R): {n_cell}\n")

                for i_cell in range(n_cell):
                    cell_shift = unique_cell_shift[i_cell].tolist()
                    mr = h_ref[i_cell]
                    mr = (mr/Rydberg).to_sparse_csr()
                    row_ptr = mr.crow_indices()
                    col_ind = mr.col_indices()
                    values = mr.values()
                    nnz = len(values)

                    f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]} {nnz}\n")
                    if nnz != 0:
                        f.write(" ".join(f"{x.item():.8e}" for x in values)+"\n")
                        f.write(" ".join(str(x.item()) for x in col_ind)+"\n")
                        f.write(" ".join(str(x.item()) for x in row_ptr)+"\n")
                    else:
                        pass
            del h_ref

        if "olp" in self.para.write:
            olp = self.rotate2abacus(data.SR, AtomType)
            olp, dim_matrix = self.block_r(olp, data)
            with open("olp.csr", "w") as f:
                f.write(f"STEP: 0\n")
                f.write(f"Matrix Dimension of S(R): {dim_matrix}\n")
                f.write(f"Matrix number of S(R): {n_cell}\n")

                for i_cell in range(n_cell):
                    cell_shift = unique_cell_shift[i_cell].tolist()
                    mr = olp[i_cell]
                    mr = mr.to_sparse_csr()
                    row_ptr = mr.crow_indices()
                    col_ind = mr.col_indices()
                    values = mr.values()
                    nnz = len(values)

                    f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]} {nnz}\n")
                    if nnz != 0:
                        f.write(" ".join(f"{x.item():.8e}" for x in values)+"\n")
                        f.write(" ".join(str(x.item()) for x in col_ind)+"\n")
                        f.write(" ".join(str(x.item()) for x in row_ptr)+"\n")
                    else:
                        pass
            del olp

        if "rR" in self.para.write:
            rR_x = self.rotate2abacus(data.rR['x'], AtomType)
            rR_y = self.rotate2abacus(data.rR['y'], AtomType)
            rR_z = self.rotate2abacus(data.rR['z'], AtomType)
            rR_x, dim_matrix = self.block_r(rR_x, data)
            rR_y, dim_matrix = self.block_r(rR_y, data)
            rR_z, dim_matrix = self.block_r(rR_z, data)

            with open("rR.csr", "w") as f:
                f.write(f"STEP: 0\n")
                f.write(f"Matrix Dimension of r(R): {dim_matrix}\n")
                f.write(f"Matrix number of r(R): {n_cell}\n")

                for i_cell in range(n_cell):
                    cell_shift = unique_cell_shift[i_cell].tolist()
                    mr_x = rR_x[i_cell]
                    mr_y = rR_y[i_cell]
                    mr_z = rR_z[i_cell]

                    mr_x = (mr_x/Bohr).to_sparse_csr()
                    mr_y = (mr_y/Bohr).to_sparse_csr()
                    mr_z = (mr_z/Bohr).to_sparse_csr()

                    row_ptr_x = mr_x.crow_indices()
                    col_ind_x = mr_x.col_indices()
                    values_x = mr_x.values()
                    nnz_x = len(values_x)

                    f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]}\n")
                    f.write(f"{nnz_x}\n")
                    if nnz_x != 0:
                        f.write(" ".join(f"{x.item():.8e}" for x in values_x)+"\n")
                        f.write(" ".join(str(x.item()) for x in col_ind_x)+"\n")
                        f.write(" ".join(str(x.item()) for x in row_ptr_x)+"\n")
                    else:
                        pass

                    row_ptr_y = mr_y.crow_indices()
                    col_ind_y = mr_y.col_indices()
                    values_y = mr_y.values()
                    nnz_y = len(values_y)

                    f.write(f"{nnz_y}\n")
                    if nnz_y != 0:
                        f.write(" ".join(f"{x.item():.8e}" for x in values_y)+"\n")
                        f.write(" ".join(str(x.item()) for x in col_ind_y)+"\n")
                        f.write(" ".join(str(x.item()) for x in row_ptr_y)+"\n")
                    else:
                        pass

                    row_ptr_z = mr_z.crow_indices()
                    col_ind_z = mr_z.col_indices()
                    values_z = mr_z.values()
                    nnz_z = len(values_z)

                    f.write(f"{nnz_z}\n")
                    if nnz_z != 0:
                        f.write(" ".join(f"{x.item():.8e}" for x in values_z)+"\n")
                        f.write(" ".join(str(x.item()) for x in col_ind_z)+"\n")
                        f.write(" ".join(str(x.item()) for x in row_ptr_z)+"\n")
                    else:
                        pass

            del rR_x, rR_y, rR_z

        if "h_pred" in self.para.write:
            data.to(self.device)
            h_pred, _ = self.model(data)
            h_pred = self.tensor2tensor(h_pred, "cpu")
            h_pred = self.rotate2abacus(h_pred, AtomType)
            data.to("cpu")
            h_pred, dim_matrix = self.block_r(h_pred, data)
            with open("h_pred.csr", "w") as f:
                f.write(f"STEP: 0\n")
                f.write(f"Matrix Dimension of H(R): {dim_matrix}\n")
                f.write(f"Matrix number of H(R): {n_cell}\n")

                for i_cell in range(n_cell):
                    cell_shift = unique_cell_shift[i_cell].tolist()
                    mr = h_pred[i_cell]
                    mr = (mr/Rydberg).to_sparse_csr()
                    row_ptr = mr.crow_indices()
                    col_ind = mr.col_indices()
                    values = mr.values()
                    nnz = len(values)

                    f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]} {nnz}\n")
                    if nnz != 0:
                        f.write(" ".join(f"{x.item():.8e}" for x in values)+"\n")
                        f.write(" ".join(str(x.item()) for x in col_ind)+"\n")
                        f.write(" ".join(str(x.item()) for x in row_ptr)+"\n")
                    else:
                        pass
            del h_pred

    def run(self):
        with torch.no_grad():
            for batch_idx, data in enumerate(self.trainloader):
                self.calculation(data=data, structure_idx=batch_idx)


if __name__ == "__main__":
    pass
