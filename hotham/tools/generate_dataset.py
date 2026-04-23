import numpy as np
import torch
from torch_geometric.data import Data
import scipy
from scipy.sparse import csr_matrix
import ase
from ase.data import atomic_numbers
from ase.io import read
from ase.neighborlist import neighbor_list
from ase.units import Hartree, Rydberg, Bohr
import os
import re
from typing import Union, List
from abc import ABC
from io import TextIOWrapper
from e3nn import o3
from tqdm import tqdm


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


def find_cell_shfit_index(S):
    unique_S = np.unique(S, axis=0)
    unique_S_tuple = [tuple(s.tolist()) for s in unique_S]
    # unique_S_tuple_sort = sorted(unique_S_tuple)
    mapping = {s: i for i, s in enumerate(unique_S_tuple)}
    S_index = np.array([mapping[tuple(s.tolist())] for s in S])
    return unique_S, S_index


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


class BasicInfo():
    def __init__(self, orbit: dict, device="cpu", intdtype=torch.int64):
        self.orbit = orbit
        self.device = device
        self.intdtype = intdtype

        # Atom type informatio
        self.AtomSymbol_to_AtomNumber = {atomsymbol: atomnumber for atomsymbol, atomnumber in atomic_numbers.items() if atomsymbol in orbit.keys()}
        self.AtomNumber_to_AtomSymbol = {atomnumber: atomsymbol for atomsymbol, atomnumber in self.AtomSymbol_to_AtomNumber.items()}
        unique_type = sorted(self.AtomNumber_to_AtomSymbol.keys())
        self.AtomNumber_to_AtomType = {num: i for i, num in enumerate(unique_type)}
        self.AtomSymbol_to_AtomType = {self.AtomNumber_to_AtomSymbol[atomnumber]: atomtype for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
        self.AtomType_to_AtomNumber = {atomtype: atomnumber for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
        self.AtomType_to_AtomSymbol = {atomtype: self.AtomNumber_to_AtomSymbol[atomnumber] for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
        self.n_type = len(self.AtomType_to_AtomSymbol)

        # Orbit information
        self.AMSymbol_to_AM = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
        self.AtomType_AMSymbolList = {self.AtomSymbol_to_AtomType[k]: list(map(lambda x: ''.join(re.findall(r'[A-Za-z]', x)), v)) for k, v in orbit.items()}
        self.AtomType_OrbitalSum = torch.tensor([sum(list(map(lambda x: 2*self.AMSymbol_to_AM[x]+1, self.AtomType_AMSymbolList[k])))
                                                for k in sorted(self.AtomType_AMSymbolList)]).to(self.intdtype).to(self.device)
        self.AtomType_AMList = {atomtype: torch.tensor(list(map(lambda x: self.AMSymbol_to_AM[x], amsymbollist))) for atomtype, amsymbollist in self.AtomType_AMSymbolList.items()}
        self.AtomSymbol_to_AMList = {self.AtomType_to_AtomSymbol[atomtype]: amlist for atomtype, amlist in self.AtomType_AMList.items()}


class DataBase(ABC):
    def __init__(self, para, dataset):
        self.para = para
        self.dataset = dataset
        self.intdtype = para.intdtype
        self.floatdtype = para.floatdtype
        self.device = para.device

        basicinfo = BasicInfo(para.orbit, self.device, self.intdtype)
        self.AtomSymbol_to_AtomNumber = basicinfo.AtomSymbol_to_AtomNumber
        self.AtomNumber_to_AtomSymbol = basicinfo.AtomNumber_to_AtomSymbol
        self.AtomNumber_to_AtomType = basicinfo.AtomNumber_to_AtomType
        self.AtomSymbol_to_AtomType = basicinfo.AtomSymbol_to_AtomType
        self.AtomType_to_AtomNumber = basicinfo.AtomType_to_AtomNumber
        self.AtomType_to_AtomSymbol = basicinfo.AtomType_to_AtomSymbol
        self.n_type = basicinfo.n_type
        self.AMSymbol_to_AM = basicinfo.AMSymbol_to_AM
        self.AtomType_AMSymbolList = basicinfo.AtomType_AMSymbolList
        self.AtomType_OrbitalSum = basicinfo.AtomType_OrbitalSum
        self.AtomType_AMList = basicinfo.AtomType_AMList
        self.AtomSymbol_to_AMList = basicinfo.AtomSymbol_to_AMList

    def find_neigbhor(self, frame: ase.Atoms, cutoff):
        i, j, d, D, S = neighbor_list("ijdDS", a=frame, cutoff=cutoff, self_interaction=self.para.edge_include_sc)
        edge_index = np.concatenate([i.reshape(1, -1), j.reshape(1, -1)], axis=0)
        edge_inverse = find_inverse_index(i, j, S)
        return [torch.from_numpy(ele).to(self.device) for ele in [i, j, d, D, S, edge_index, edge_inverse]]

    def get_wigner_Ds(self, lmax, edge_vec):
        # edge_vec should be yzx order
        # R@((0,1,0).T) = (y,z,x).T
        try:
            import hotham
            self._Jd = torch.load(os.path.join(hotham.__path__[0], "utilities/Jd.pt"))
        except:
            self._Jd = torch.load("D:/Users/lzx/source/repos/Hot-Ham/hotham/utilities/Jd.pt")
        # self._Jd = torch.load("/fs08/home/js_liangzx/anaconda3/envs/deep/apps/hotham/utilities/Jd.pt")
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
    def __init__(self, para: dict, dataset: str):
        super().__init__(para, dataset)
        self.dataset = self.get_data()

        if self.para.using_CoordinateTransformation:
            for data in self.dataset:
                data.wigner_D = self.get_wigner_Ds(self.para.L_max, data.D_hop[:, [1, 2, 0]])
                if self.para.edge_include_sc:
                    data.mask_edge = (data.d_hop > 1.0e-6)
                    data.mask_sc = ~data.mask_edge
                    for index in range(len(data.wigner_D)):
                        data.wigner_D[index][data.mask_sc] = torch.eye(2*index+1, dtype=data.wigner_D[index].dtype, device=data.wigner_D[index].device).unsqueeze(0)

    @staticmethod
    def read_real(fid: TextIOWrapper):
        return list(map(float, fid.readline().split()))

    @staticmethod
    def read_complex(fid: TextIOWrapper):
        def tuple2complex(t: str):
            t = eval(t)
            return t[0]+t[1]*1.j
        return list(map(tuple2complex, fid.readline().split()))

    def get_Hamiltonian(self, filename: str, TotalOrbital: int):
        HR = {}
        with open(filename, "r") as fid:
            line = fid.readline()
            csr_dim = int(fid.readline().split()[-1])
            if csr_dim == TotalOrbital:
                read_func = self.read_real
            elif csr_dim == 2*TotalOrbital:
                read_func = self.read_complex
            csr_number = int(fid.readline().split()[-1])
            line = fid.readline()
            while line:
                s1, s2, s3, nnz = [int(i) for i in line.split()]
                key = (s1, s2, s3)
                if nnz == 0:
                    line = fid.readline()
                else:
                    line_V = read_func(fid)
                    line_COL_INDEX = list(map(int, fid.readline().split()))
                    line_ROW_INDEX = list(map(int, fid.readline().split()))
                    block = csr_matrix((line_V,
                                        line_COL_INDEX,
                                        line_ROW_INDEX),
                                       shape=(csr_dim, csr_dim)).toarray()
                    HR[key] = torch.from_numpy(block) * Rydberg
                    line = fid.readline()
        return HR

    def get_Overlap(self, filename: str, TotalOrbital: int):
        SR = {}
        with open(filename, "r") as fid:
            line = fid.readline()
            csr_dim = int(fid.readline().split()[-1])
            if csr_dim == TotalOrbital:
                read_func = self.read_real
            elif csr_dim == 2*TotalOrbital:
                read_func = self.read_complex
            csr_number = int(fid.readline().split()[-1])
            line = fid.readline()
            while line:
                s1, s2, s3, nnz = [int(i) for i in line.split()]
                key = (s1, s2, s3)
                if nnz == 0:
                    line = fid.readline()
                else:
                    line_V = read_func(fid)
                    line_COL_INDEX = list(map(int, fid.readline().split()))
                    line_ROW_INDEX = list(map(int, fid.readline().split()))
                    block = csr_matrix((line_V,
                                        line_COL_INDEX,
                                        line_ROW_INDEX),
                                       shape=(csr_dim, csr_dim)).toarray()
                    SR[key] = torch.from_numpy(block)
                    line = fid.readline()
        return SR

    def get_rR(self, filename: str, TotalOrbital: int):
        rR = {"x": {}, "y": {}, "z": {}}
        with open(filename, "r") as fid:
            line = fid.readline()
            csr_dim = int(fid.readline().split()[-1])
            if csr_dim == TotalOrbital:
                read_func = self.read_real
            elif csr_dim == 2*TotalOrbital:
                read_func = self.read_complex
            csr_number = int(fid.readline().split()[-1])
            line = fid.readline()
            while line:
                s1, s2, s3 = [int(i) for i in line.split()]
                key = (s1, s2, s3)

                for dirction in ["x", "y", "z"]:
                    nnz = int(fid.readline())
                    if nnz == 0:
                        pass
                    else:
                        line_V = read_func(fid)
                        line_COL_INDEX = list(map(int, fid.readline().split()))
                        line_ROW_INDEX = list(map(int, fid.readline().split()))
                        block = csr_matrix((line_V,
                                            line_COL_INDEX,
                                            line_ROW_INDEX),
                                           shape=(csr_dim, csr_dim)).toarray()
                        rR[dirction][key] = torch.from_numpy(block) * Bohr
                line = fid.readline()
        return rR

    def get_HS(self, HS_file: dict, TotalOrbital: int):
        filedata = dict()
        for file_key, file_value in HS_file.items():
            if file_value is None:
                filedata[file_key] = {}

            elif file_key in ["H0_file", "H1_file"]:
                filedata[file_key] = self.get_Hamiltonian(file_value, TotalOrbital)

            elif file_key in ["S_file"]:
                filedata[file_key] = self.get_Overlap(file_value, TotalOrbital)

            elif file_key in ["rR_file"]:
                filedata[file_key] = self.get_rR(file_value, TotalOrbital)
        return list(filedata.values())

    def get_wigner_D(self, order: Union[int, List[int]]):
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

    def abacus2hotham(self,
                      # matrix
                      HR: dict,
                      iHR: dict,
                      SR: dict,
                      rR: dict,
                      # structure information
                      AtomType: torch.tensor,
                      cell_shift: torch.tensor,
                      edge_index: torch.tensor,
                      # orbital information
                      offset: torch.tensor,
                      TotalOrbital: int):
        has_HR, has_iHR, has_SR, has_rR = False, False, False, False
        H_block, iH_block, S_block, rR_block = {}, {}, {}, {}

        def create_pair_dict(symbols):
            return {s0: {s1: [] for s1 in symbols} for s0 in symbols}

        if len(HR) > 0:
            has_HR = True
            H_block = create_pair_dict(self.AtomSymbol_to_AtomType.keys())
        if len(iHR) > 0:
            has_iHR = True
            iH_block = create_pair_dict(self.AtomSymbol_to_AtomType.keys())
        if len(SR) > 0:
            has_SR = True
            S_block = create_pair_dict(self.AtomSymbol_to_AtomType.keys())
        if len(rR) > 0:
            has_rR = True
            rR_block = {k: create_pair_dict(self.AtomSymbol_to_AtomType.keys()) for k in rR}

        nspin = 1
        if len(iHR):
            nspin = 2
        elif len(HR) or len(SR):
            MR = HR if len(HR) else SR
            csr_dim = list(MR.values())[0].shape[0]
            if csr_dim == 2*TotalOrbital:
                nspin = 4

        for i in range(edge_index.shape[1]):
            s1, s2, s3 = cell_shift[i]
            n1, n2 = edge_index[:, i]
            n1, n2 = n1.item(), n2.item()
            start1, start2 = offset[[n1, n2]]
            atomtype_1 = AtomType[n1].item()
            atomtype_2 = AtomType[n2].item()
            atomsymbol_1 = self.AtomType_to_AtomSymbol[atomtype_1]
            atomsymbol_2 = self.AtomType_to_AtomSymbol[atomtype_2]
            o1 = self.AtomType_OrbitalSum[atomtype_1]
            o2 = self.AtomType_OrbitalSum[atomtype_2]
            key_dft = (s1.item(), s2.item(), s3.item())
            # key_hotham = (s1.item(), s2.item(), s3.item(), n1, n2)
            if has_HR:
                if key_dft in HR:
                    if nspin == 1:
                        # H_spin_o1_o2 = HR[key_dft][start1:start1+o1, start2:start2+o2][None, ...]
                        H_spin_o1_o2 = HR[key_dft][start1:start1+o1, start2:start2+o2]
                        H_block[atomsymbol_1][atomsymbol_2].append(H_spin_o1_o2)
                    elif nspin > 1:
                        raise NotImplementedError("Interface for nspin>1 is under development")
                else:
                    H_spin_o1_o2 = torch.zeros((o1, o2))
                    # H_spin_o1_o2 = np.zeros((o1,o2))[None, ...]
                    H_block[atomsymbol_1][atomsymbol_2].append(H_spin_o1_o2)
                    # raise KeyError(f"Can't find {key_dft} derived by ase in abacus's neighbor list")
            if has_iHR:
                if key_dft in iHR:
                    if nspin == 1:
                        iH_spin_o1_o2 = iHR[key_dft][start1:start1+o1, start2:start2+o2][None, ...]
                        iH_block[atomsymbol_1][atomsymbol_2].append(iH_spin_o1_o2)
                    elif nspin > 1:
                        raise NotImplementedError("Interface for nspin>1 is under development")
                else:
                    raise KeyError(f"Can't find {key_dft} derived by ase in abacus's neighbor list")
            if has_SR:
                if key_dft in SR:
                    s_o1_o2 = SR[key_dft][start1:start1+o1, start2:start2+o2]
                    S_block[atomsymbol_1][atomsymbol_2].append(s_o1_o2)
                else:
                    s_o1_o2 = torch.zeros((o1, o2))
                    S_block[atomsymbol_1][atomsymbol_2].append(s_o1_o2)
                    # raise KeyError(f"Can't find {key_dft} derived by ase in abacus's neighbor list")
            if has_rR:
                for direction in ["x", "y", "z"]:
                    if key_dft in rR[direction]:
                        rr_o1_o2 = rR[direction][key_dft][start1:start1+o1, start2:start2+o2]
                        rR_block[direction][atomsymbol_1][atomsymbol_2].append(rr_o1_o2)
                    else:
                        rr_o1_o2 = torch.zeros((o1, o2))
                        rR_block[direction][atomsymbol_1][atomsymbol_2].append(rr_o1_o2)

        for atomtype_1 in range(self.n_type):
            atomsymbol_1 = self.AtomType_to_AtomSymbol[atomtype_1]
            winger_D_1 = self.get_wigner_D(self.AtomSymbol_to_AMList[atomsymbol_1])
            for atomtype_2 in range(self.n_type):
                atomsymbol_2 = self.AtomType_to_AtomSymbol[atomtype_2]
                winger_D_2 = self.get_wigner_D(self.AtomSymbol_to_AMList[atomsymbol_2])
                if has_HR and (len(H_block[atomsymbol_1][atomsymbol_2]) != 0):
                    H_block[atomsymbol_1][atomsymbol_2] = torch.stack(H_block[atomsymbol_1][atomsymbol_2]).to(torch.float32)
                    # H_block[atomtype_1][atomtype_2] = torch.einsum("ij,zsjk,kl->zsil", winger_D_1.T, H_block[atomtype_1][atomtype_2], winger_D_2)
                    H_block[atomsymbol_1][atomsymbol_2] = torch.einsum("ij,zjk,kl->zil", winger_D_1.T, H_block[atomsymbol_1][atomsymbol_2], winger_D_2)
                if has_iHR and (len(iH_block[atomsymbol_1][atomsymbol_2]) != 0):
                    iH_block[atomsymbol_1][atomsymbol_2] = torch.stack(iH_block[atomsymbol_1][atomsymbol_2])
                    iH_block[atomsymbol_1][atomsymbol_2] = torch.einsum("ij,zsjk,kl->zsil", winger_D_1.T, iH_block[atomsymbol_1][atomsymbol_2], winger_D_2)
                if has_SR and (len(S_block[atomsymbol_1][atomsymbol_2]) != 0):
                    S_block[atomsymbol_1][atomsymbol_2] = torch.stack(S_block[atomsymbol_1][atomsymbol_2]).to(torch.float32)
                    S_block[atomsymbol_1][atomsymbol_2] = torch.einsum("ij,zjk,kl->zil", winger_D_1.T, S_block[atomsymbol_1][atomsymbol_2], winger_D_2)
                if has_rR:
                    for direction in rR_block:
                        if len(rR_block[direction][atomsymbol_1][atomsymbol_2]) != 0:
                            rR_block[direction][atomsymbol_1][atomsymbol_2] = np.stack(rR_block[direction][atomsymbol_1][atomsymbol_2])
                            rR_block[direction][atomsymbol_1][atomsymbol_2] = np.einsum("ij,zjk,kl->zil", winger_D_1.T, rR_block[direction][atomsymbol_1][atomsymbol_2], winger_D_2)

        return H_block, iH_block, S_block, rR_block

    def get_data(self):
        dataset = []
        paths = []
        for root, _, files in os.walk(self.dataset, followlinks=True):
            HS_file = {"H0_file": None, "H1_file": None, "S_file": None, "rR_file": None}
            if "data-HR-sparse_SPIN0.csr" in files:
                HS_file["H0_file"] = os.path.join(root, "data-HR-sparse_SPIN0.csr")
            if "data-HR-sparse_SPIN1.csr" in files:
                HS_file["H1_file"] = os.path.join(root, "data-HR-sparse_SPIN1.csr")
            if "data-SR-sparse_SPIN0.csr" in files:
                HS_file["S_file"] = os.path.join(root, "data-SR-sparse_SPIN0.csr")
            elif "SR.csr" in files:
                HS_file["S_file"] = os.path.join(root, "SR.csr")
            if "data-rR-sparse.csr" in files:
                HS_file["rR_file"] = os.path.join(root, "data-rR-sparse.csr")
            if all([f is None for f in list(HS_file.values())]):
                continue
            paths.append((root, HS_file))

        for root, HS_file in tqdm(paths):
            structure = read(os.path.join(root, "../model.xyz"))

            # atom_type, n_type, lattice, position
            AtomType = torch.tensor([self.AtomNumber_to_AtomType[atomnumber] for atomnumber in structure.numbers])
            n_type = self.n_type
            lattice = torch.tensor(np.array(structure.cell))
            pos = torch.tensor(structure.positions)

            # atom_type's orbit number
            # Hamiltonian and overlap start index for each atom
            Node_OrbitalSum = self.AtomType_OrbitalSum[AtomType]
            offset = torch.cumsum(Node_OrbitalSum, dim=0)-Node_OrbitalSum
            TotalOrbital = Node_OrbitalSum.sum()

            # Hamiltonian (spin, key, orbit_0, oribit_1)
            # overlap     (key, orbit_0, oribit_1)
            HR, iHR, SR, rR = self.get_HS(HS_file=HS_file, TotalOrbital=TotalOrbital)

            # 1.calculate and check neighbor list
            # 2.convert abacus's Hamiltonian and overlap to hotham's order
            #   Hamiltonian  (n_type, n_type, edge, spin, orbit_0, oribit_1)
            #   iHamiltonian (n_type, n_type, edge, spin, orbit_0, oribit_1)
            #   overlap      (n_type, n_type, edge,       orbit_0, oribit_1)
            cutoff = [self.para["cutoff"][symbol]*Bohr for symbol in structure.get_chemical_symbols()]
            _, _, d, D, S, edge_index, edge_inverse = self.find_neigbhor(frame=structure, cutoff=cutoff)
            unique_cell_shift, cell_shift_index = find_cell_shfit_index(S)
            HR, iHR, SR, rR = self.abacus2hotham(HR=HR,
                                                 iHR=iHR,
                                                 SR=SR,
                                                 rR=rR,
                                                 AtomType=AtomType,
                                                 edge_index=edge_index,
                                                 cell_shift=S,
                                                 offset=offset,
                                                 TotalOrbital=TotalOrbital)

            # save as dict
            data = Data(
                AtomType=AtomType,
                AtomType_OrbitalSum=self.AtomType_OrbitalSum,
                offset=offset,
                n_type=n_type,
                lattice=lattice,
                pos=pos.to(self.floatdtype),
                edge_index_hop=edge_index.to(self.intdtype),
                edge_inverse=edge_inverse.to(self.intdtype),
                D_hop=D.to(self.floatdtype),
                d_hop=d.to(self.floatdtype),
                S_hop=S.to(self.floatdtype),
                unique_cell_shift=torch.from_numpy(unique_cell_shift).to(self.intdtype),
                cell_shift_index=torch.from_numpy(cell_shift_index).to(self.intdtype)
            )
            # data = {"AtomType": AtomType,
            #         "AtomType_OrbitalSum": self.AtomType_OrbitalSum,
            #         "offset": offset,
            #         "n_type": n_type,
            #         "lattice": lattice,
            #         "pos": pos,
            #         "edge_index": edge_index,
            #         "inv_edge_index": edge_inverse,
            #         "D": D,
            #         "d": d,
            #         "S": S,
            #         "unique_cell_shift": unique_cell_shift,
            #         "cell_shift_index": cell_shift_index}
            if len(HR) != 0:
                data["HR"] = numpy2tensor(HR, "cpu")
            if len(iHR) != 0:
                data["iHR"] = numpy2tensor(iHR, "cpu")
            if len(SR) != 0:
                data["SR"] = numpy2tensor(SR, "cpu")
            if len(rR) != 0:
                data["rR"] = numpy2tensor(rR, "cpu")
            dataset.append(data)
        return dataset


class OpenmxData(DataBase):
    def __init__(self, para: dict, dataset: str):
        super().__init__(para, dataset)
        self.dataset = self.get_data()

        if self.para.using_CoordinateTransformation:
            for data in self.dataset:
                data.wigner_D = self.get_wigner_Ds(self.para.L_max, data.D_hop[:, [1, 2, 0]])
                if self.para.edge_include_sc:
                    data.mask_edge = (data.d_hop > 1.0e-6)
                    data.mask_sc = ~data.mask_edge
                    for index in range(len(data.wigner_D)):
                        data.wigner_D[index][data.mask_sc] = torch.eye(2*index+1, dtype=data.wigner_D[index].dtype, device=data.wigner_D[index].device).unsqueeze(0)

    def get_Hamiltonian(self, fid: TextIOWrapper):
        HR = {}
        line = fid.readline()
        # line must start with "Block"
        while line:
            if "Block" not in line:
                break
            n1, n2, s1, s2, s3, dim0, dim1 = [int(i) for i in line.split()[1:]]
            key = (s1, s2, s3, n1-1, n2-1)
            block = np.zeros(shape=(dim0, dim1))
            for i in range(dim0):
                block[i] = np.array(fid.readline().split())
            block = block*Hartree
            HR[key] = block
            line = fid.readline()
        return HR

    def get_Overlap(self, fid: TextIOWrapper):
        SR = {}
        line = fid.readline()
        # line must start with "Block"
        while line:
            if "Block" not in line:
                break
            n1, n2, s1, s2, s3, dim0, dim1 = [int(i) for i in line.split()[1:]]
            key = (s1, s2, s3, n1-1, n2-1)
            block = np.zeros(shape=(dim0, dim1))
            for i in range(dim0):
                block[i] = np.array(fid.readline().split())
            SR[key] = block
            line = fid.readline()
        return SR

    def get_rR(self, fid: TextIOWrapper):
        rR = {}
        line = fid.readline()
        # line must start with "Block"
        while line:
            if "Block" not in line:
                break
            n1, n2, s1, s2, s3, dim0, dim1 = [int(i) for i in line.split()[1:]]
            key = (s1, s2, s3, n1-1, n2-1)
            block = np.zeros(shape=(dim0, dim1))
            for i in range(dim0):
                block[i] = np.array(fid.readline().split())
            rR[key] = block*Bohr
            line = fid.readline()
        return rR

    def get_HS(self, filename):
        HR, iHR, SR, rR = {}, {}, {}, {}
        SpinP_switch = -1
        with open(filename, "r") as fid:
            line = fid.readline()
            while line:
                # read SpinP_switch
                if "SpinP_switch" in line:
                    SpinP_switch = int(line.split("=")[1])

                # read HR
                elif ("Kohn-Sham Hamiltonian" in line) and ("iHks" not in line):
                    spin = int(line.split("=")[1])
                    HR[spin] = self.get_Hamiltonian(fid)

                # read iHR
                elif "iHks: Kohn-Sham Hamiltonian" in line:
                    spin = int(line.split("=")[1])
                    iHR[spin] = self.get_Hamiltonian(fid)

                # read SR
                elif "Overlap matrix" in line:
                    SR = self.get_Overlap(fid)

                # read rR x
                elif "Overlap x matrix" in line:
                    rR["x"] = self.get_rR(fid)

                # read rR y
                elif "Overlap y matrix" in line:
                    rR["y"] = self.get_rR(fid)

                # read rR z
                elif "Overlap z matrix" in line:
                    rR["z"] = self.get_rR(fid)

                line = fid.readline()
        # assert len(HR) == (SpinP_switch+1)
        return HR, iHR, SR, rR

    def get_wigner_D(self, order: Union[int, List[int]]):
        """
        D @ Y_wiki == Y_openmx
        """
        D = [
            np.array([[1.0]]),
            np.array([[0, 0, 1],
                      [1, 0, 0],
                      [0, 1, 0]]),
            np.array([[0, 0, 1, 0, 0],
                      [0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0]]),
            np.array([[0, 0, 0, 1, 0, 0, 0],
                      [0, 0, 0, 0, 1, 0, 0],
                      [0, 0, 1, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 1, 0],
                      [0, 1, 0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0, 0, 1],
                      [1, 0, 0, 0, 0, 0, 0]])
        ]

        if isinstance(order, int):
            order = [order]

        DirectSum = scipy.linalg.block_diag(*[D[l] for l in order])
        return DirectSum

    def openmx2hotham(self, HR: dict, iHR: dict, SR: dict, rR: dict, AtomType: np.array, cell_shift: np.array, edge_index: np.array):
        # Hamiltonian  (spin, key, orbit_0, oribit_1)->(n_type, n_type, edge, spin, orbit_0, oribit_1)
        # iHamiltonian (spin, key, orbit_0, oribit_1)->(n_type, n_type, edge, spin, orbit_0, oribit_1)
        # overlap            (key, orbit_0, oribit_1)->(n_type, n_type, edge,       orbit_0, oribit_1)
        has_HR, has_iHR, has_SR, has_rR = False, False, False, False
        H_block, iH_block, S_block, rR_block = {}, {}, {}, {}

        def create_pair_dict(symbols):
            return {s0: {s1: [] for s1 in symbols} for s0 in symbols}

        if len(HR) > 0:
            has_HR = True
            H_block = create_pair_dict(self.AtomSymbol_to_AtomType.keys())
        if len(iHR) > 0:
            has_iHR = True
            iH_block = create_pair_dict(self.AtomSymbol_to_AtomType.keys())
        if len(SR) > 0:
            has_SR = True
            S_block = create_pair_dict(self.AtomSymbol_to_AtomType.keys())
        if len(rR) > 0:
            has_rR = True
            rR_block = {k: create_pair_dict(self.AtomSymbol_to_AtomType.keys()) for k in rR}
        for index in range(edge_index.shape[1]):
            s1, s2, s3 = cell_shift[index]
            n1, n2 = edge_index.T[index]
            atomtype_1 = AtomType[n1].item()
            atomtype_2 = AtomType[n2].item()
            atomsymbol_1 = self.AtomType_to_AtomSymbol[atomtype_1]
            atomsymbol_2 = self.AtomType_to_AtomSymbol[atomtype_2]
            key = (s1.item(), s2.item(), s3.item(), n1.item(), n2.item())
            if has_HR:
                H_spin_o1_o2 = np.stack([HR[spin][key] for spin in HR])
                H_block[atomsymbol_1][atomsymbol_2].append(H_spin_o1_o2)
            if has_iHR:
                iH_spin_o1_o2 = np.stack([iHR[spin][key] for spin in iHR])
                iH_block[atomsymbol_1][atomsymbol_2].append(iH_spin_o1_o2)
            if has_SR:
                S_o1_o2 = SR[key]
                S_block[atomsymbol_1][atomsymbol_2].append(S_o1_o2)
            if has_rR:
                for direction in rR:
                    rR_d_o1_o2 = rR[direction][key]
                    rR_block[direction][atomsymbol_1][atomsymbol_2].append(rR_d_o1_o2)

        for atomtype_1 in range(self.n_type):
            atomsymbol_1 = self.AtomType_to_AtomSymbol[atomtype_1]
            winger_D_1 = self.get_wigner_D(self.AtomSymbol_to_AMList[atomsymbol_1])
            for atomtype_2 in range(self.n_type):
                atomsymbol_2 = self.AtomType_to_AtomSymbol[atomtype_2]
                winger_D_2 = self.get_wigner_D(self.AtomSymbol_to_AMList[atomsymbol_2])
                if has_HR and (len(H_block[atomsymbol_1][atomsymbol_2]) != 0):
                    H_block[atomsymbol_1][atomsymbol_2] = np.stack(H_block[atomsymbol_1][atomsymbol_2])
                    H_block[atomsymbol_1][atomsymbol_2] = np.einsum("ij,zsjk,kl->zsil", winger_D_1.T, H_block[atomsymbol_1][atomsymbol_2], winger_D_2)
                if has_iHR and (len(iH_block[atomsymbol_1][atomsymbol_2]) != 0):
                    iH_block[atomsymbol_1][atomsymbol_2] = np.stack(iH_block[atomsymbol_1][atomsymbol_2])
                    iH_block[atomsymbol_1][atomsymbol_2] = np.einsum("ij,zsjk,kl->zsil", winger_D_1.T, iH_block[atomsymbol_1][atomsymbol_2], winger_D_2)
                if has_SR and (len(S_block[atomsymbol_1][atomsymbol_2]) != 0):
                    S_block[atomsymbol_1][atomsymbol_2] = np.stack(S_block[atomsymbol_1][atomsymbol_2])
                    S_block[atomsymbol_1][atomsymbol_2] = np.einsum("ij,zjk,kl->zil", winger_D_1.T, S_block[atomsymbol_1][atomsymbol_2], winger_D_2)
                if has_rR:
                    for direction in rR_block:
                        if len(rR_block[direction][atomsymbol_1][atomsymbol_2]) != 0:
                            rR_block[direction][atomsymbol_1][atomsymbol_2] = np.stack(rR_block[direction][atomsymbol_1][atomsymbol_2])
                            rR_block[direction][atomsymbol_1][atomsymbol_2] = np.einsum("ij,zjk,kl->zil", winger_D_1.T, rR_block[direction][atomsymbol_1][atomsymbol_2], winger_D_2)

        return H_block, iH_block, S_block, rR_block

    def get_data(self):
        dataset = []
        paths = []
        for root, _, files in os.walk(self.dataset, followlinks=True):
            if "Hks.txt" in files:
                HS_file = os.path.join(root, "Hks.txt")
            elif "overlap.txt" in files:
                HS_file = os.path.join(root, "overlap.txt")
            else:
                continue
            paths.append((root, HS_file))

        for root, HS_file in tqdm(paths):
            structure = read(os.path.join(root, "model.xyz"))

            # atom_type, n_type, lattice, position
            AtomType = np.array([self.AtomNumber_to_AtomType[atomnumber] for atomnumber in structure.numbers])
            n_type = self.n_type
            lattice = np.array(structure.cell)
            pos = structure.positions

            # Hamiltonian (spin, key, orbit_0, oribit_1)
            # overlap     (key, orbit_0, oribit_1)
            HR, iHR, SR, rR = self.get_HS(HS_file)

            # infer neighbor list from HR[0]
            if len(HR) > 0:
                keys = np.array(list(HR[0].keys()))
            elif len(SR) > 0:
                keys = np.array(list(SR.keys()))
            cell_shift, edge_index = keys[:, :3], keys[:, 3:].T
            inv_edge_index = find_inverse_index(I=edge_index[0], J=edge_index[1], S=cell_shift)
            unique_cell_shift, cell_shift_index = find_cell_shfit_index(cell_shift)
            D = (pos[edge_index[1]]-pos[edge_index[0]]+cell_shift@lattice)
            d = np.linalg.norm(D, axis=1)

            # convert openmx's Hamiltonian and overlap to hotham's order
            # Hamiltonian  (n_type, n_type, edge, spin, orbit_0, oribit_1)
            # iHamiltonian (n_type, n_type, edge, spin, orbit_0, oribit_1)
            # overlap      (n_type, n_type, edge,       orbit_0, oribit_1)
            HR, iHR, SR, rR = self.openmx2hotham(HR=HR, iHR=iHR, SR=SR, rR=rR, AtomType=AtomType, cell_shift=cell_shift, edge_index=edge_index)

            # atom_type's orbit number
            # Hamiltonian and overlap start index for each atom
            AtomType_OrbitalSum = self.AtomType_OrbitalSum
            offset = torch.cumsum(self.AtomType_OrbitalSum[AtomType], dim=0)-self.AtomType_OrbitalSum[AtomType]

            # save as dict
            data = Data(
                AtomType=torch.from_numpy(AtomType).to(self.intdtype),
                AtomType_OrbitalSum=self.AtomType_OrbitalSum,
                offset=offset,
                n_type=n_type,
                lattice=torch.from_numpy(lattice),
                pos=torch.from_numpy(pos).to(self.floatdtype),
                edge_index_hop=torch.from_numpy(edge_index).to(self.intdtype),
                edge_inverse=torch.from_numpy(inv_edge_index).to(self.intdtype),
                D_hop=torch.from_numpy(D).to(self.floatdtype),
                d_hop=torch.from_numpy(d).to(self.floatdtype),
                S_hop=torch.from_numpy(cell_shift).to(self.floatdtype),
                unique_cell_shift=torch.from_numpy(unique_cell_shift).to(self.intdtype),
                cell_shift_index=torch.from_numpy(cell_shift_index).to(self.intdtype)
            )
            # data = {"AtomType": AtomType,
            #         "AtomType_OrbitalSum": AtomType_OrbitalSum,
            #         "offset": offset,
            #         "n_type": n_type,
            #         "lattice": lattice,
            #         "pos": pos,
            #         "edge_index": edge_index,
            #         "inv_edge_index": inv_edge_index,
            #         "D": D,
            #         "d": d,
            #         "S": cell_shift,
            #         "unique_cell_shift": unique_cell_shift,
            #         "cell_shift_index": cell_shift_index}
            if len(HR) != 0:
                data["HR"] = numpy2tensor(HR, "cpu")
            if len(iHR) != 0:
                data["iHR"] = numpy2tensor(iHR, "cpu")
            if len(SR) != 0:
                data["SR"] = numpy2tensor(SR, "cpu")
            if len(rR) != 0:
                data["rR"] = numpy2tensor(rR, "cpu")
            dataset.append(data)
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
        "trainset": ".",
        # "testset": "./data/testset",
        # "valset": "./data/valset",
        "dft": "abacus",
        "orbit": {
            "H": ["1s", "2s", "2p"],
            "C": ["1s", "2s", "2p", "3p", "3d"],
            "N": ["1s", "2s", "2p", "3p", "3d"],
            "O": ["1s", "2s", "2p", "3p", "3d"],
        },
        "L_max": 5,
        # only used by AbacusData
        "cutoff": {
            "H": 8,
            "C": 8,
            "N": 8,
            "O": 8,
        }
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
