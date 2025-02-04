import h5py
import numpy as np
import torch
import os
import sys
from tqdm import tqdm
import json5


class Diagonal():
    def __init__(self, inputfile: dict):
        self.inputfile = inputfile
        self.device = torch.device(inputfile["device"])
        self.KPAR = inputfile["KPAR"]
        self.has_olp = False
        self.has_href = False

    def get_kpoints(self, kpath_list):
        kpath = np.asarray(kpath_list)
        if kpath.shape[-1] == 4:
            kpoints = kpath[:, 0:3]
            num_kp = kpath[:, 3].astype(int)

            kpath_list = []
            for i in range(len(kpoints)-1):
                tmp = np.linspace(kpoints[i], kpoints[i+1],
                                  num_kp[i]+1)[0:num_kp[i]]
                kpath_list.append(tmp)

            kpath_list.append(kpoints[-1:])
            kpath_list = np.concatenate(kpath_list, axis=0)
            kpath_list = torch.from_numpy(kpath_list).to(torch.float32).to(self.device)

        elif kpath.shape[-1] == 3:
            kpath_list = torch.from_numpy(kpath).to(torch.float32).to(self.device)

        chunks = kpath_list.shape[0]//self.KPAR+1
        kpath_list = torch.chunk(input=kpath_list, chunks=chunks, dim=0)
        return kpath_list

    def read_matrixes(self, filename):
        matrixes = {}
        with h5py.File(filename, "r") as fp:
            # matrixes["num_node"] = fp["num_node"][()]
            matrixes["num_atomtype"] = fp["num_atomtype"][()]
            matrixes["num_edge"] = fp["num_edge"][()]
            matrixes["dim_matrix"] = fp["dim_matrix"][()]
            matrixes["AtomType"] = torch.from_numpy(fp["AtomType"][()]).to(self.device)
            # matrixes["GraphEdgeIndex_to_BlockEdgeIndex"] = torch.from_numpy(fp["GraphEdgeIndex_to_BlockEdgeIndex"][()])
            matrixes["Node_offset"] = torch.from_numpy(fp["Node_offset"][()]).to(self.device)
            matrixes["index_edge"] = torch.from_numpy(fp["index_edge"][()]).to(self.device)

            for atomtype_1 in range(matrixes["num_atomtype"]):
                for atomtype_2 in range(matrixes["num_atomtype"]):
                    index_HBlcok = atomtype_1*matrixes["num_atomtype"]+atomtype_2
                    key = f"bondtype_matrix-{index_HBlcok}"
                    matrixes[key] = torch.from_numpy(fp[key][()]).to(self.device)

            if matrixes["bondtype_matrix-0"].shape[0] == 2:
                self.has_olp = True
            elif matrixes["bondtype_matrix-0"].shape[0] == 3:
                self.has_olp = True
                self.has_href = True

            return matrixes

    def matrixes_k(self, matrixes: dict, k: torch.tensor):
        dim_matrix = matrixes["dim_matrix"]

        num_matrix = 1
        if self.has_olp:
            num_matrix += 1
        if self.has_href:
            num_matrix += 1

        Mk = torch.zeros((k.shape[0], num_matrix, dim_matrix, dim_matrix), dtype=torch.complex64, device=self.device)
        num_atomtype = matrixes["num_atomtype"]
        num_edge = matrixes["num_edge"]
        index_edge = matrixes["index_edge"]
        AtomType = matrixes["AtomType"]
        Node_offset = matrixes["Node_offset"]
        cell_shift = index_edge[:, 2:].to(k.dtype)
        for atomtype_1 in range(num_atomtype):
            for atomtype_2 in range(num_atomtype):
                index_HBlcok = atomtype_1*num_atomtype+atomtype_2
                key = f"bondtype_matrix-{index_HBlcok}"
                Ms = matrixes[key]

                edge_12 = torch.arange(num_edge, dtype=torch.long, device=self.device)[(AtomType[index_edge[:, 0]] == atomtype_1)*(AtomType[index_edge[:, 1]] == atomtype_2)]
                exp_kr = torch.exp(-2.j*torch.pi*(k@(cell_shift[edge_12].T)))
                Ms_k = torch.einsum("anij,kn->kanij", Ms, exp_kr)
                dim0, dim1 = Ms.shape[-2:]
                Mk_flatten = Mk.reshape(Mk.shape[:-2]+(-1,))
                offset0 = Node_offset[index_edge[edge_12, 0]]
                offset1 = Node_offset[index_edge[edge_12, 1]]
                for i in range(dim0):
                    for j in range(dim1):
                        index = (offset0+i)*dim_matrix+(offset1+j)
                        Mk_flatten.index_add_(-1, index=index, source=Ms_k[..., i, j])
        return Mk

    def compute(self):
        dirname = self.inputfile["dirname"]
        kpath_list = self.inputfile["kpath_list"]

        # convert kpath
        kpath_list = self.get_kpoints(kpath_list)

        # read hamilton and overlap matrix
        Matrix_file = os.path.join(dirname, "Matrix-0.hdf5")
        assert os.path.exists(Matrix_file)

        self.matrixes = self.read_matrixes(Matrix_file)

        # strart diagnalizing
        EigPred, EigRef = [], []
        for k in tqdm(kpath_list, desc="Diagonalizing"):
            if len(k.shape) != 2:
                k = k.reshape(-1, 3)
            Mk = self.matrixes_k(self.matrixes, k)
            HPred_k = Mk[:, 0, :, :]
            OLP_k = Mk[:, 1, :, :]
            L_inv = torch.linalg.inv(torch.linalg.cholesky(OLP_k))
            LHL = L_inv @ HPred_k @ torch.transpose(L_inv, dim0=-1, dim1=-2).conj()
            eig = torch.linalg.eigvalsh(LHL)
            EigPred.append(eig)

            if self.has_href:
                HRef_k = Mk[:, 2, :, :]
                LHL = L_inv @ HRef_k @ torch.transpose(L_inv, dim0=-1, dim1=-2).conj()
                eig = torch.linalg.eigvalsh(LHL)
                EigRef.append(eig)

        kpath_list = torch.cat(kpath_list, dim=0)
        # save eigs
        EigPred_txt = os.path.join(dirname, f"EigPred.txt")
        EigPred = torch.cat([kpath_list, torch.cat(EigPred, dim=0)], dim=-1).to("cpu").numpy()
        np.savetxt(EigPred_txt, EigPred, header="Kx Ky Kz Eigs")

        if self.has_href:
            EigRed_txt = os.path.join(dirname, f"EigRef.txt")
            EigRef = torch.cat([kpath_list, torch.cat(EigRef, dim=0)], dim=-1).to("cpu").numpy()
            np.savetxt(EigRed_txt, EigRef, header="Kx Ky Kz Eigs")


if __name__ == "__main__":
    inputfile = sys.argv[1]
    with open(inputfile, "r") as f:
        inputfile = json5.load(f)
    with torch.no_grad():
        diagnal = Diagonal(inputfile)
        diagnal.compute()
