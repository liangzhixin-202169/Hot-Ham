import numpy as np
import torch
from torch_geometric.data import Data
from scipy.sparse import csr_matrix
import ase
from ase.data import atomic_numbers
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.units import Hartree, Bohr
import os
import re
from typing import Union, List
from abc import ABC
import h5py
from e3nn import o3


def find_inverse_index(I, J, S):
    index_inv = {}
    for index in range(len(I)):
        i, j = I[index], J[index]
        s1, s2, s3 = S[index]
        ijs = (i, j, s1, s2, s3)
        ijs_inv = (j, i, -s1, -s2, -s3)

        index_inv[ijs] = [index]+index_inv.setdefault(ijs, [])
        index_inv[ijs_inv] = index_inv.setdefault(ijs_inv, [])+[index]

    return np.array(sorted(index_inv.values()))[:, 1]


def numpy2tensor(data, device):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data).to(device)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = numpy2tensor(v, device)
        return data
    elif isinstance(data, list):
        for i, e in enumerate(data):
            data[i] = numpy2tensor(e, device)
        return data
    elif isinstance(data, Data):
        data = data.to(torch.device(device))
        return data
    else:
        return data


def tensor2device(data, device):
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = tensor2device(v, device)
        return data
    elif isinstance(data, list):
        for i, e in enumerate(data):
            data[i] = tensor2device(e, device)
        return data
    else:
        return data


class Parameters(dict):
    def __init__(self, para: dict):
        super().__init__()
        self.update(self.set_default_parameters())
        self.update(para)
        self.atomic_numbers = [ase.data.atomic_numbers[ele] for ele in self.orbit.keys()]
        self.num_types = len(self.atomic_numbers)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Parameters' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value

    def set_default_parameters(self):
        default_dict = {}
        # Set dtype and device
        default_dict["intdtype"] = torch.int64
        default_dict["floatdtype"] = torch.float32
        default_dict["device"] = "cpu"
        # Dataset path
        default_dict["trainset"] = None
        default_dict["valset"] = None
        default_dict["testset"] = None
        default_dict["dft"] = None
        default_dict["edge_include_sc"] = True
        # Data preprocess
        default_dict["using_CoordinateTransformation"] = True
        return default_dict


class DataBase(ABC):
    def __init__(self, para, dataset):
        self.para = para
        self.dataset = dataset
        self.intdtype = para.intdtype
        self.floatdtype = para.floatdtype
        self.device = para.device
        self.rc = para.rc

        self.AtomSymbol_to_AtomNumber = {atomsymbol: atomnumber for atomsymbol, atomnumber in atomic_numbers.items() if atomsymbol in para.orbit.keys()}
        self.AtomNumber_to_AtomSymbol = {atomnumber: atomsymbol for atomsymbol, atomnumber in self.AtomSymbol_to_AtomNumber.items()}
        unique_type = sorted(self.AtomNumber_to_AtomSymbol.keys())
        self.AtomNumber_to_AtomType = {num: i for i, num in enumerate(unique_type)}
        self.AtomSymbol_to_AtomType = {self.AtomNumber_to_AtomSymbol[atomnumber]: atomtype for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
        self.AtomType_to_AtomNumber = {atomtype: atomnumber for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
        self.AtomType_to_AtomSymbol = {atomtype: self.AtomNumber_to_AtomSymbol[atomnumber] for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}

        self.AMSymbol_to_AM = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
        self.AtomType_AMSymbolList = {self.AtomSymbol_to_AtomType[k]: list(map(lambda x: ''.join(re.findall(r'[A-Za-z]', x)), v)) for k, v in para.orbit.items()}
        self.AtomType_OrbitalSum = torch.tensor([sum(list(map(lambda x: 2*self.AMSymbol_to_AM[x]+1, self.AtomType_AMSymbolList[k])))
                                                for k in sorted(self.AtomType_AMSymbolList)]).to(self.intdtype).to(para.device)
        self.AtomType_AMList = {atomtype: torch.tensor(list(map(lambda x: self.AMSymbol_to_AM[x], amsymbollist))) for atomtype, amsymbollist in self.AtomType_AMSymbolList.items()}

    def find_neigbhor(self, frame: ase.Atoms, cutoff):
        i, j, d, D, S = neighbor_list("ijdDS", a=frame, cutoff=cutoff, self_interaction=self.para.edge_include_sc)
        edge_index = np.concatenate([i.reshape(1, -1), j.reshape(1, -1)], axis=0)
        edge_inverse = find_inverse_index(i, j, S)
        return [torch.from_numpy(ele).to(self.device) for ele in [i, j, d, D, S, edge_index, edge_inverse]]

    def get_wigner_Ds(self, lmax, edge_vec):
        # edge_vec should be yzx order
        # R@((0,1,0).T) = (y,z,x).T
        # self._Jd = torch.load(os.path.join(os.path.dirname(__file__), "../utilities/Jd.pt"))
        # self._Jd = torch.load("D:/Users/17183/repo/hotham-mace/Hot-Ham/hotham/utilities/Jd.pt")
        self._Jd = torch.load("/fs08/home/js_liangzx/anaconda3/envs/deep/apps/hotham/utilities/Jd.pt")
        alpha, beta = o3.xyz_to_angles(edge_vec)
        wigner_D = [[] for _ in range(lmax+1)]
        for l in range(lmax+1):
            D = self.wigner_D(l, alpha, beta, torch.zeros_like(alpha))
            wigner_D[l] = D
        return wigner_D

    def wigner_D(self, l, alpha, beta, gamma):
        if not l < len(self._Jd):
            raise NotImplementedError(
                f"wigner D maximum l implemented is {len(self._Jd) - 1}"
            )

        alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
        J = self._Jd[l].to(dtype=alpha.dtype, device=alpha.device)
        Xa = self._z_rot_mat(alpha, l)
        Xb = self._z_rot_mat(beta, l)
        Xc = self._z_rot_mat(gamma, l)
        return Xa @ J @ Xb @ J @ Xc

    def _z_rot_mat(self, angle, l):
        shape, device, dtype = angle.shape, angle.device, angle.dtype
        M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
        inds = torch.arange(0, 2 * l + 1, 1, device=device)
        reversed_inds = torch.arange(2 * l, -1, -1, device=device)
        frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
        M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
        M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
        return M


class AbacusData(DataBase):
    def __init__(self, para: dict, dataset):
        super().__init__(para, dataset)
        self.device = para.device
        self.target = para.train_target
        if self.target == "hamiltonian":
            self.dataset = self.Hamilton()
        elif self.target == "band":
            self.dataset = self.Band()

        if self.para.using_CoordinateTransformation:
            for data in self.dataset:
                data.wigner_D = self.get_wigner_Ds(self.para.L_max, data.D_hop[:, [1, 2, 0]])
                if self.para.edge_include_sc:
                    data.mask_edge = (data.d_hop > 1.0e-6)
                    data.mask_sc = ~data.mask_edge
                    for index in range(len(data.wigner_D)):
                        data.wigner_D[index][data.mask_sc] = torch.eye(2*index+1, dtype=data.wigner_D[index].dtype, device=data.wigner_D[index].device).unsqueeze(0)

    def get_wigner_D(self, order: Union[int, List[int]]):
        """
        D @ Y_wiki == Y_abacus
        """
        D = [
            torch.tensor([[1.0]], dtype=self.floatdtype, device=self.device),
            torch.tensor([[0, 1, 0],
                          [0, 0, -1],
                          [-1, 0, 0]], dtype=self.floatdtype, device=self.device),
            torch.tensor([[0, 0, 1, 0, 0],
                          [0, 0, 0, -1, 0],
                          [0, -1, 0, 0, 0],
                          [0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0]], dtype=self.floatdtype, device=self.device)
        ]

        if isinstance(order, int):
            order = [order]

        DirectSum = torch.block_diag(*[D[l] for l in order])
        return DirectSum

    def read_HS(self, hsfile, S, flag="H"):
        '''read non-orthogonal Hamiltonian and overlap'''
        S_tuple = set([tuple(i.tolist()) for i in S])
        if flag == "H":
            factor = 13.605698
        elif flag == "S":
            factor = 1

        hsR = {}
        with open(hsfile, "r") as fid:
            line = fid.readline()
            csr_dim = int(fid.readline().split()[-1])
            line = fid.readline()
            line = fid.readline()
            while line:
                s1, s2, s3, nnz = [int(i) for i in line.split()]
                s_key = (s1, s2, s3)
                if nnz == 0:
                    line = fid.readline()
                    # hsr = torch.zeros((csr_dim, csr_dim))
                    continue
                else:
                    line_V = fid.readline().split()
                    line_COL_INDEX = fid.readline().split()
                    line_ROW_INDEX = fid.readline().split()
                    line = fid.readline()
                    if s_key not in S_tuple:  # flag == "H" and (s_key not in S_tuple):
                        continue
                    hsr = torch.from_numpy(csr_matrix((np.array(line_V).astype(float),
                                                       np.array(line_COL_INDEX).astype(int),
                                                       np.array(line_ROW_INDEX).astype(int)),
                                                      shape=(csr_dim, csr_dim)).toarray()).to(torch.float32).to(device=self.device)
                hsR[s_key] = hsr * factor
        return hsR

    def HS_preprocess(self, hsR: dict, AtomType, edge_index, S):
        if self.target == "hamiltonian":
            num_atomtype = len(self.AtomSymbol_to_AtomType)
            H_block = [[] for _ in range(num_atomtype**2)]
            # if not self.para.edge_include_sc:
            #     Onsite_block=[[] for _ in range(num_atomtype)]
            offset1 = torch.cumsum(self.AtomType_OrbitalSum[AtomType], dim=0).to(self.device)
            offset2 = torch.cat([torch.tensor([0]).to(self.device), offset1[:-1]], dim=0)

            for index in range(edge_index.shape[-1]):
                s1, s2, s3 = S[index]
                s_key = (s1.item(), s2.item(), s3.item())
                n1, n2 = edge_index.T[index]
                atomtype_1 = AtomType[n1].item()
                atomtype_2 = AtomType[n2].item()
                index_HBlcok = atomtype_1*num_atomtype+atomtype_2
                offset_row1 = offset1[n1].item()
                offset_col1 = offset1[n2].item()
                offset_row2 = offset2[n1].item()
                offset_col2 = offset2[n2].item()

                if s_key not in hsR.keys():
                    block = torch.zeros((1, offset_row1-offset_row2, offset_col1-offset_col2))
                else:
                    block = hsR[s_key][offset_row2:offset_row1, offset_col2:offset_col1].unsqueeze(0)

                H_block[index_HBlcok].append(block)

            if not self.para.edge_include_sc:
                s_key = (0, 0, 0)
                for n in range(AtomType.shape[0]):
                    offset_row1 = offset1[n].item()
                    offset_row2 = offset2[n].item()
                    block = hsR[s_key][offset_row2:offset_row1, offset_row2:offset_row1].unsqueeze(0)
                    atomtype = AtomType[n].item()
                    index_HBlock = atomtype*num_atomtype+atomtype
                    H_block[index_HBlock].append(block)

            H_block = [torch.cat(sub_block).to(self.device) for sub_block in H_block]

            """
            B_wiki == D_i.T @ B_abacus @ D_j
            """
            for atomtype_i in range(num_atomtype):
                for atomtype_j in range(num_atomtype):
                    winger_D_i = self.get_wigner_D(self.AtomType_AMList[atomtype_i])
                    winger_D_j = self.get_wigner_D(self.AtomType_AMList[atomtype_j])

                    index_HBlcok = atomtype_i*num_atomtype+atomtype_j
                    H_block[index_HBlcok] = torch.einsum("ij,zjk,kl->zil", winger_D_i.T, H_block[index_HBlcok], winger_D_j)

            return H_block

        elif self.target == "band":
            offset1 = torch.cumsum(self.AtomType_OrbitalSum[AtomType], dim=0).to(self.device)
            offset2 = torch.cat([torch.tensor([0]).to(self.device), offset1[:-1]], dim=0)
            hsR_rotated = torch.stack(list(hsR.values()))
            for ni in range(len(AtomType)):
                for nj in range(len(AtomType)):
                    winger_D_i = self.get_wigner_D(self.AtomType_AMList[AtomType[ni].item()])
                    winger_D_j = self.get_wigner_D(self.AtomType_AMList[AtomType[nj].item()])

                    offset_row1 = offset1[ni].item()
                    offset_col1 = offset1[nj].item()
                    offset_row2 = offset2[ni].item()
                    offset_col2 = offset2[nj].item()

                    block_ij = hsR_rotated[:, offset_row2:offset_row1, offset_col2:offset_col1]
                    hsR_rotated[:, offset_row2:offset_row1, offset_col2:offset_col1] = torch.einsum("ij,zjk,kl->zil", winger_D_i.T, block_ij, winger_D_j)
            H_block = {key: value for key, value in zip(hsR.keys(), hsR_rotated)}

        return H_block

    def Hamilton(self):
        dataset = []

        for root, _, files in os.walk(self.dataset):
            if "data-HR-sparse_SPIN0.csr" in files:
                HRFile = os.path.join(root, "data-HR-sparse_SPIN0.csr")
                STRUFile = os.path.join(root, "STRU")
                assert os.path.exists(f"{STRUFile}")
                frame = read(STRUFile)

                _, _, d, D, S, edge_index, edge_inverse = self.find_neigbhor(frame, self.rc)
                AtomType = torch.tensor([self.AtomSymbol_to_AtomType[symbol] for symbol in frame.symbols], dtype=torch.long)
                lattice = torch.tensor(frame.cell.array).unsqueeze(0).to(self.floatdtype)
                pos = torch.from_numpy(frame.positions).to(self.floatdtype)
                HR = self.read_HS(HRFile, S)
                HR = self.HS_preprocess(HR, AtomType, edge_index, S)

                data = Data(AtomType=AtomType,
                            # lattice=lattice,
                            pos=pos,
                            HR=HR,
                            edge_index_hop=edge_index.to(self.intdtype),
                            d_hop=d.to(self.floatdtype),
                            D_hop=D.to(self.floatdtype),
                            S_hop=S.to(self.intdtype),
                            edge_inverse=edge_inverse.to(self.intdtype))

                dataset.append(data.to(device=self.device))

        return dataset

    def Band(self):
        dataset = []

        for root, _, files in os.walk(self.dataset):
            if "STRU" in files or "model.xyz" in files or "running_scf.log" in files:
                # if "running_scf.log" in files:
                kpoints_file = os.path.join(root, "kpoints.npy")
                eigs_file = os.path.join(root, "BANDS_1.dat")
                structure_file = os.path.join(root, "model.xyz") if "model.xyz" in files else os.path.join(root, "STRU")
                # structure_file = os.path.join(root, "running_scf.log")
                SRFile = os.path.join(root, "data-SR-sparse_SPIN0.csr")
                HRFile = os.path.join(root, "data-HR-sparse_SPIN0.csr")

                if os.path.exists(kpoints_file):
                    kpoints = torch.from_numpy(np.load(kpoints_file)).to(self.para.floatdtype)
                if os.path.exists(eigs_file):
                    eigs_ref = torch.from_numpy(np.loadtxt(eigs_file)).to(self.para.floatdtype)
                assert os.path.exists(structure_file)
                frame = read(structure_file)

                _, _, d, D, S, edge_index, edge_inverse = self.find_neigbhor(frame, self.rc)
                AtomType = torch.tensor([self.AtomSymbol_to_AtomType[symbol] for symbol in frame.symbols], dtype=self.intdtype)
                lattice = torch.tensor(frame.cell.array).unsqueeze(0).to(self.floatdtype)
                pos = torch.from_numpy(frame.positions).to(self.floatdtype)

                data = Data(AtomType=AtomType,
                            # lattice=lattice,
                            pos=pos,
                            edge_index_hop=edge_index.to(self.intdtype),
                            d_hop=d.to(self.floatdtype),
                            D_hop=D.to(self.floatdtype),
                            S_hop=S.to(self.intdtype),
                            edge_inverse=edge_inverse.to(self.intdtype))

                if os.path.exists(kpoints_file):
                    data.kpoints = kpoints.unsqueeze(0)
                if os.path.exists(eigs_file):
                    data.eigs_ref = eigs_ref.unsqueeze(0)
                if os.path.exists(SRFile):
                    SR = self.read_HS(SRFile, S, "S")
                    SR = self.HS_preprocess(SR, AtomType, edge_index, S)
                    data.SR = [SR]
                if os.path.exists(HRFile):
                    HR = self.read_HS(HRFile, S)
                    HR = self.HS_preprocess(HR, AtomType, edge_index, S)
                    data.HR = [HR]

                dataset.append(data.to(device=self.device))

        return dataset


class OpenmxData(DataBase):
    def __init__(self, para: Union[dict, Parameters], dataset):
        super().__init__(para, dataset)
        self.device = para.device
        self.target = para.train_target
        if self.target == "hamiltonian":
            self.dataset = self.Hamilton()
        elif self.target == "band":
            self.dataset = self.Band()
        elif self.target == "hamiltonian_from_deeph":
            self.dataset = self.Hamilton_from_deeph()

        if self.para.using_CoordinateTransformation:
            for data in self.dataset:
                data.wigner_D = self.get_wigner_Ds(self.para.L_max, data.D_hop[:, [1, 2, 0]])
                if self.para.edge_include_sc:
                    data.mask_edge = (data.d_hop > 1.0e-6)
                    data.mask_sc = ~data.mask_edge
                    for index in range(len(data.wigner_D)):
                        data.wigner_D[index][data.mask_sc] = torch.eye(2*index+1, dtype=data.wigner_D[index].dtype, device=data.wigner_D[index].device).unsqueeze(0)

    def get_wigner_D(self, order: Union[int, List[int]]):
        """
        D @ Y_wiki == Y_openmx
        """
        D = [
            torch.tensor([[1.0]], dtype=self.floatdtype, device=self.device),
            torch.tensor([[0, 0, 1],
                          [1, 0, 0],
                          [0, 1, 0]], dtype=self.floatdtype, device=self.device),
            torch.tensor([[0, 0, 1, 0, 0],
                          [0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0],
                          [0, 0, 0, 1, 0],
                          [0, 1, 0, 0, 0]], dtype=self.floatdtype, device=self.device),
            torch.tensor([[0, 0, 0, 1, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0, 0],
                          [0, 0, 1, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 1, 0],
                          [0, 1, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 1],
                          [1, 0, 0, 0, 0, 0, 0]], dtype=self.floatdtype, device=self.device)
        ]

        if isinstance(order, int):
            order = [order]

        DirectSum = torch.block_diag(*[D[l] for l in order])
        return DirectSum

    def read_HS(self, hsfile):
        if self.target == "hamiltonian":
            hsR, onsite = {}, {}
            with open(hsfile, "r") as fid:
                line = fid.readline()
                while line:
                    if "atomnum" in line:
                        atomnum = int(line[:-1].split("=")[1])
                        line = fid.readline()
                    elif "SpinP_switch" in line:
                        SpinP_switch = int(line[:-1].split("=")[1])
                        if SpinP_switch > 0:
                            raise NotImplementedError("Spin polarized and non-collinear calculation is not implemented.")
                        line = fid.readline()
                    elif "Kohn-Sham Hamiltonian" in line:
                        spin = int(line[:-1].split("=")[1])
                        line = fid.readline().split()
                        while line and (line != "\n"):
                            assert line[0] == "Block:"
                            n1, n2, s1, s2, s3, dim0, dim1 = [int(i) for i in line[1:]]
                            s_key = (s1, s2, s3, n1-1, n2-1)
                            block = np.zeros(shape=(dim0, dim1))
                            for i in range(dim0):
                                line = fid.readline().split()
                                block[i] = np.array(line)
                            block = block*Hartree
                            if self.para.edge_include_sc:
                                hsR[s_key] = torch.from_numpy(block).to(self.device)
                            else:
                                if (s1 == s2 == s3 == 0) and (n1 == n2):
                                    onsite[n1-1] = torch.from_numpy(block).to(self.device)
                                else:
                                    hsR[s_key] = torch.from_numpy(block).to(self.device)
                            line = fid.readline().split()
                    else:
                        line = fid.readline()
            return hsR, onsite

        elif self.target == "band":
            HAM, OLP = {}, {}
            with open(hsfile, "r") as fid:
                line = fid.readline()
                while line:
                    if "atomnum" in line:
                        atomnum = int(line[:-1].split("=")[1])
                        line = fid.readline()
                    elif "SpinP_switch" in line:
                        SpinP_switch = int(line[:-1].split("=")[1])
                        if SpinP_switch > 0:
                            raise NotImplementedError("Spin polarized and non-collinear calculation is not implemented.")
                        line = fid.readline()
                    elif "Kohn-Sham Hamiltonian" in line:
                        spin = int(line[:-1].split("=")[1])
                        line = fid.readline()
                        while line and (line != "\n"):
                            line = line.split()
                            assert line[0] == "Block:"
                            n1, n2, s1, s2, s3, dim0, dim1 = [int(i) for i in line[1:]]
                            s_key = (s1, s2, s3, n1-1, n2-1)
                            block = np.zeros(shape=(dim0, dim1))
                            for i in range(dim0):
                                line = fid.readline().split()
                                block[i] = np.array(line)
                            block = block*Hartree
                            HAM[s_key] = block
                            line = fid.readline()
                    elif "Overlap matrix" in line:
                        line = fid.readline()
                        while line and (line != "\n"):
                            line = line.split()
                            assert line[0] == "Block:"
                            n1, n2, s1, s2, s3, dim0, dim1 = [int(i) for i in line[1:]]
                            s_key = (s1, s2, s3, n1-1, n2-1)
                            block = np.zeros(shape=(dim0, dim1))
                            for i in range(dim0):
                                line = fid.readline().split()
                                block[i] = np.array(line)
                            block = block
                            OLP[s_key] = block
                            line = fid.readline()
                    else:
                        line = fid.readline()
            return HAM, OLP

        elif self.target == "hamiltonian_from_deeph":
            """
            if read hamiltonian, hamiltonian can be split into hopping and onsite, or not split,
            if read overlap, not split
            """
            hsR, onsite = {}, {}
            with h5py.File(os.path.join(hsfile), "r") as fid:
                for key in fid.keys():
                    *S, n1, n2 = eval(key)
                    s_key = tuple(S)+(n1-1, n2-1)
                    if self.para.edge_include_sc:
                        hsR[s_key] = torch.from_numpy(np.array(fid[key])).to(self.device)
                    else:
                        if (torch.tensor(S).pow(2).sum() == 0) and (n1 == n2):
                            onsite[n1-1] = torch.from_numpy(np.array(fid[key])).to(self.device)
                        else:
                            hsR[s_key] = torch.from_numpy(np.array(fid[key])).to(self.device)

            return hsR, onsite

    def HS_preprocess(self, hsR: dict, onsite: dict, AtomType, edge_index, S):
        if self.target in ["hamiltonian", "hamiltonian_from_deeph"]:
            num_atomtype = len(self.AtomSymbol_to_AtomType)
            H_block = [[] for _ in range(num_atomtype**2)]
            offset1 = torch.cumsum(self.AtomType_OrbitalSum[AtomType], dim=0).to(self.device)
            offset2 = torch.cat([torch.tensor([0]).to(self.device), offset1[:-1]], dim=0)

            for index in range(edge_index.shape[-1]):
                s1, s2, s3 = S[index]
                # s_key = (s1.item(), s2.item(), s3.item())
                n1, n2 = edge_index.T[index]
                atomtype_1 = AtomType[n1].item()
                atomtype_2 = AtomType[n2].item()
                index_HBlock = atomtype_1*num_atomtype+atomtype_2
                offset_row1 = offset1[n1].item()
                offset_col1 = offset1[n2].item()
                offset_row2 = offset2[n1].item()
                offset_col2 = offset2[n2].item()

                s_key = (s1.item(), s2.item(), s3.item(), n1.item(), n2.item())
                if s_key not in hsR.keys():
                    block = torch.tensor((1, offset_row1-offset_row2, offset_col1-offset_col2))
                else:
                    # block = hsR[s_key][np.newaxis, ...]
                    block = hsR[s_key].unsqueeze(0)

                H_block[index_HBlock].append(block)

            if onsite:
                for n in range(AtomType.shape[0]):
                    block = onsite[n]
                    atomtype = AtomType[n].item()
                    index_HBlock = atomtype*num_atomtype+atomtype
                    H_block[index_HBlock].append(block.unsqueeze(0))

            H_block = [torch.cat(sub_block).to(self.device).to(self.floatdtype) for sub_block in H_block]

            """
            B_wiki == D_i.T @ B_abacus @ D_j
            """
            for atomtype_i in range(num_atomtype):
                for atomtype_j in range(num_atomtype):
                    winger_D_i = self.get_wigner_D(self.AtomType_AMList[atomtype_i])
                    winger_D_j = self.get_wigner_D(self.AtomType_AMList[atomtype_j])

                    index_HBlcok = atomtype_i*num_atomtype+atomtype_j
                    H_block[index_HBlcok] = torch.einsum("ij,zjk,kl->zil", winger_D_i.T, H_block[index_HBlcok], winger_D_j)

            return H_block

        elif self.target == "band":
            num_atomtype = len(self.AtomSymbol_to_AtomType)
            H_block = [[] for _ in range(num_atomtype**2)]
            offset1 = torch.cumsum(self.AtomType_OrbitalSum[AtomType], dim=0).to(self.device)
            offset2 = torch.cat([torch.tensor([0]).to(self.device), offset1[:-1]], dim=0)

            for index in range(edge_index.shape[-1]):
                s1, s2, s3 = S[index]
                n1, n2 = edge_index.T[index]
                atomtype_1 = AtomType[n1].item()
                atomtype_2 = AtomType[n2].item()
                index_HBlock = atomtype_1*num_atomtype+atomtype_2
                offset_row1 = offset1[n1].item()
                offset_col1 = offset1[n2].item()
                offset_row2 = offset2[n1].item()
                offset_col2 = offset2[n2].item()

                s_key = (s1.item(), s2.item(), s3.item(), n1.item(), n2.item())
                assert s_key in hsR.keys()
                block = torch.from_numpy(hsR[s_key]).unsqueeze(0)

                H_block[index_HBlock].append(block)
            H_block = [torch.cat(sub_block).to(self.device).to(self.floatdtype) for sub_block in H_block]

            for atomtype_i in range(num_atomtype):
                for atomtype_j in range(num_atomtype):
                    winger_D_i = self.get_wigner_D(self.AtomType_AMList[atomtype_i])
                    winger_D_j = self.get_wigner_D(self.AtomType_AMList[atomtype_j])

                    index_HBlcok = atomtype_i*num_atomtype+atomtype_j
                    H_block[index_HBlcok] = torch.einsum("ij,zjk,kl->zil", winger_D_i.T, H_block[index_HBlcok], winger_D_j)

            # hsR_rotated = torch.stack(list(hsR.values()))
            # for ni in range(len(AtomType)):
            #     for nj in range(len(AtomType)):
            #         winger_D_i = self.get_wigner_D(self.AtomType_AMList[AtomType[ni].item()])
            #         winger_D_j = self.get_wigner_D(self.AtomType_AMList[AtomType[nj].item()])

            #         offset_row1 = offset1[ni].item()
            #         offset_col1 = offset1[nj].item()
            #         offset_row2 = offset2[ni].item()
            #         offset_col2 = offset2[nj].item()

            #         block_ij = hsR_rotated[:, offset_row2:offset_row1, offset_col2:offset_col1]
            #         hsR_rotated[:, offset_row2:offset_row1, offset_col2:offset_col1] = torch.einsum("ij,zjk,kl->zil", winger_D_i.T, block_ij, winger_D_j)
            # H_block = {key: value for key, value in zip(hsR.keys(), hsR_rotated)}

        return H_block

    def Hamilton(self):
        dataset = []

        for root, _, files in os.walk(self.dataset):
            if "Hks.txt" in files:
                HRFile = os.path.join(root, "Hks.txt")
                structure_file = os.path.join(root, "model.xyz")
                assert os.path.exists(structure_file)
                frame = read(structure_file)

                AtomType = torch.tensor([self.AtomNumber_to_AtomType[atomnumber] for atomnumber in frame.numbers])
                lattice = torch.from_numpy(np.array(frame.cell)).to(self.device)
                pos = torch.from_numpy(frame.positions).to(self.device)

                HR, onsite = self.read_HS(HRFile)
                keys = np.array(list(HR.keys()))
                S, edge_index = keys[:, :3], keys[:, 3:].T
                edge_inverse = find_inverse_index(edge_index[0], edge_index[1], S)
                S = torch.from_numpy(S).to(self.intdtype).to(self.device)
                edge_index = torch.from_numpy(edge_index).to(self.intdtype).to(self.device)
                HR = self.HS_preprocess(HR, onsite, AtomType, edge_index, S)
                D = (pos[edge_index[1]]-pos[edge_index[0]]+S.to(pos.dtype)@lattice).to(self.floatdtype)
                d = torch.norm(D, dim=1)

                data = Data(AtomType=AtomType,
                            # lattice=lattice.to(self.floatdtype),
                            pos=pos.to(self.floatdtype),
                            HR=HR,
                            edge_index_hop=edge_index,
                            d_hop=d.to(self.floatdtype),
                            D_hop=D,
                            S_hop=S,
                            edge_inverse=torch.from_numpy(edge_inverse).to(self.intdtype))

                dataset.append(data.to(device=self.device))
        return dataset

    def Band(self):
        dataset = []

        for root, _, files in os.walk(self.dataset):
            if "overlap.txt" in files:
                OLP = os.path.join(root, "overlap.txt")
                HRFile = os.path.join(root, "Hks.txt")
                structure_file = os.path.join(root, "model.xyz")
                assert os.path.exists(OLP) and os.path.exists(structure_file)
                _, OLP = self.read_HS(OLP)
                frame = read(structure_file)

                AtomType = torch.tensor([self.AtomNumber_to_AtomType[atomnumber] for atomnumber in frame.numbers])
                lattice = torch.from_numpy(np.array(frame.cell)).to(self.device)
                pos = torch.from_numpy(frame.positions).to(self.device)

                cutoff = [self.para.cutoff[symbol]*Bohr for symbol in frame.get_chemical_symbols()]
                _, _, d, D, S, edge_index, edge_inverse = self.find_neigbhor(frame=frame, cutoff=cutoff)

                OLP = self.HS_preprocess(OLP, {}, AtomType, edge_index, S)

                data = Data(AtomType=AtomType,
                            pos=pos.to(self.floatdtype),
                            # HR=HR,
                            edge_index_hop=edge_index.to(self.intdtype),
                            d_hop=d.to(self.floatdtype),
                            D_hop=D.to(self.floatdtype),
                            S_hop=S.to(self.intdtype),
                            edge_inverse=edge_inverse.to(self.intdtype),
                            # SR=[OLP])
                            SR=OLP)

                if os.path.exists(HRFile):
                    HR, _ = self.read_HS(HRFile)
                    HR = self.HS_preprocess(HR, {}, AtomType, edge_index, S)
                    # Data.HR = [HR]
                    Data.HR = HR

                dataset.append(data.to(device=self.device))
        return dataset

    def Hamilton_from_deeph(self):
        dataset = []

        for root, _, files in os.walk(self.dataset):
            if "hamiltonians.h5" in files:
                element = os.path.join(root, "element.dat")
                HRFile = os.path.join(root, "hamiltonians.h5")
                lat = os.path.join(root, "lat.dat")
                site_positions = os.path.join(root, "site_positions.dat")

                assert os.path.exists(f"{element}")
                assert os.path.exists(f"{lat}")
                assert os.path.exists(f"{site_positions}")

                AtomType = torch.tensor([self.AtomNumber_to_AtomType[atomnumber] for atomnumber in np.loadtxt(element)], dtype=self.intdtype).to(self.device)
                lattice = torch.from_numpy(np.loadtxt(lat).T[np.newaxis, ...]).to(self.device)
                pos = torch.from_numpy(np.loadtxt(site_positions).T).to(self.device)

                # HR实际上就是hopping,包括self-connection的话onsite就是空的
                HR, onsite = self.read_HS(HRFile)
                keys = np.array(list(HR.keys()))
                S, edge_index = keys[:, :3], keys[:, 3:].T
                edge_inverse = find_inverse_index(edge_index[0], edge_index[1], S)
                S = torch.from_numpy(S).to(self.intdtype).to(self.device)
                edge_index = torch.from_numpy(edge_index).to(self.intdtype).to(self.device)
                HR = self.HS_preprocess(HR, onsite, AtomType, edge_index, S)
                D = (pos[edge_index[1]]-pos[edge_index[0]]+S.to(pos.dtype)@lattice[0]).to(self.floatdtype)
                d = torch.norm(D, dim=1)
                assert (d < self.para.rc).all(), f"rc = {self.para.rc} smaller than d_max: {torch.max(d)}"

                data = Data(AtomType=AtomType,
                            # lattice=lattice.to(self.floatdtype),
                            pos=pos.to(self.floatdtype),
                            HR=HR,
                            edge_index_hop=edge_index,
                            d_hop=d.to(self.floatdtype),
                            D_hop=D,
                            S_hop=S,
                            edge_inverse=torch.from_numpy(edge_inverse).to(self.intdtype))

                dataset.append(data.to(device=self.device))

        return dataset


class GraphData(DataBase):
    def __init__(self, para: Union[dict, Parameters], dataset):
        super().__init__(para, dataset)
        self.device = para.device
        self.dataset = self.get_graph()

        if self.para.using_CoordinateTransformation:
            for data in self.dataset:
                data.wigner_D = self.get_wigner_Ds(self.para.L_max, data.D_hop[:, [1, 2, 0]])
                if self.para.edge_include_sc:
                    data.mask_edge = (data.d_hop > 1.0e-6)
                    data.mask_sc = ~data.mask_edge
                    for index in range(len(data.wigner_D)):
                        data.wigner_D[index][data.mask_sc] = torch.eye(2*index+1, dtype=data.wigner_D[index].dtype, device=data.wigner_D[index].device).unsqueeze(0)

    def get_graph(self):
        dataset = []

        for root, _, files in os.walk(self.dataset):
            if "model.xyz" in files:
                structure_file = os.path.join(root, "model.xyz")
                frame = read(structure_file)

                AtomType = torch.tensor([self.AtomNumber_to_AtomType[atomnumber] for atomnumber in frame.numbers])
                lattice = torch.from_numpy(np.array(frame.cell))
                pos = torch.from_numpy(frame.positions)

                cutoff = [self.para.cutoff[symbol]*Bohr for symbol in frame.get_chemical_symbols()]
                _, _, d, D, S, edge_index, edge_inverse = self.find_neigbhor(frame=frame, cutoff=cutoff)

                data = Data(
                    AtomType=AtomType,
                    lattice=lattice,
                    pos=pos.to(self.floatdtype),
                    edge_index_hop=edge_index.to(self.intdtype),
                    d_hop=d.to(self.floatdtype),
                    D_hop=D.to(self.floatdtype),
                    S_hop=S.to(self.intdtype),
                    edge_inverse=edge_inverse.to(self.intdtype)
                )

                dataset.append(data)
        return dataset


if __name__ == "__main__":
    inputfile = {
        # "trainset": ,
        # "valset": ,
        "testset": "/fs08/home/js_liangzx/anaconda3/envs/deep/neptb/dataset/GTB/openmx-water5/getdata/data/valset",
        "train_target": "hamiltonian",
        "dft": "openmx",
        "orbit": {
            "H": [
                "1s",
                "2s",
                "2p"
            ],
            "O": [
                "1s",
                "2s",
                "2p",
                "3p",
                "3d"
            ]
        },
        "rc": 7.5,
        "L_max": 5
    }

    param = Parameters(inputfile)
    if param.dft == "abacus":
        DATACLASS = AbacusData
    elif param.dft == "openmx":
        DATACLASS = OpenmxData
    elif param.dft is None:
        DATACLASS = GraphData

    for dataset in ["trainset", "valset", "testset"]:
        if param[dataset] is not None:
            data = DATACLASS(param, param[dataset])
            torch.save(tensor2device(data.dataset, "cpu"), dataset+".pth")
