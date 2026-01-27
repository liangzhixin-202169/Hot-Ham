import numpy as np
import torch
from torch_geometric.data import Data
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
import h5py
from io import TextIOWrapper
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

def find_cell_shfit_index(S):
    unique_S = np.unique(S, axis=0)
    unique_S_tuple = [tuple(s.tolist()) for s in unique_S]
    unique_S_tuple_sort = sorted(unique_S_tuple)
    mapping = {s: i for i, s in enumerate(unique_S_tuple_sort)}
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
        self.n_type = len(self.AtomType_to_AtomSymbol)

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

    def get_HS(self, HS_file: dict, TotalOrbital: int):
        filedata = dict()
        for file_key, file_value in HS_file.items():
            if file_value is None:
                filedata[file_key] = None

            elif file_key in ["H0_file", "H1_file"]:
                filedata[file_key] = self.get_Hamiltonian(file_value, TotalOrbital)

            elif file_key in ["S_file"]:
                filedata[file_key] = self.get_Overlap(file_value, TotalOrbital)
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
                      # structure information
                      AtomType: torch.tensor,
                      cell_shift: torch.tensor,
                      edge_index: torch.tensor,
                      # orbital information
                      offset: torch.tensor,
                      TotalOrbital: int):
        has_HR, has_iHR, has_SR = False, False, False
        H_block, iH_block, S_block = [], [], []
        if HR is not None:
            has_HR = True
            H_block = [[[] for _ in range(self.n_type)] for _ in range(self.n_type)]
        if iHR is not None:
            has_iHR = True
            iH_block = [[[] for _ in range(self.n_type)] for _ in range(self.n_type)]
        if SR is not None:
            has_SR = True
            S_block = [[[] for _ in range(self.n_type)] for _ in range(self.n_type)]

        nspin = 1
        if iHR is not None:
            nspin = 2
        elif HR is not None or SR is not None:
            MR = HR if HR is not None else SR
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
            o1 = self.AtomType_OrbitalSum[atomtype_1]
            o2 = self.AtomType_OrbitalSum[atomtype_2]
            key_dft = (s1.item(), s2.item(), s3.item())
            key_hotham = (s1.item(), s2.item(), s3.item(), n1, n2)
            if has_HR:
                if key_dft in HR:
                    if nspin == 1:
                        # H_spin_o1_o2 = HR[key_dft][start1:start1+o1, start2:start2+o2][None, ...]
                        H_spin_o1_o2 = HR[key_dft][start1:start1+o1, start2:start2+o2]
                        H_block[atomtype_1][atomtype_2].append(H_spin_o1_o2)
                    elif nspin > 1:
                        raise NotImplementedError("Interface for nspin>1 is under development")
                else:
                    H_spin_o1_o2 = torch.zeros((o1,o2))
                    #H_spin_o1_o2 = np.zeros((o1,o2))[None, ...]
                    H_block[atomtype_1][atomtype_2].append(H_spin_o1_o2)
                    # raise KeyError(f"Can't find {key_dft} derived by ase in abacus's neighbor list")
            if has_iHR:
                if key_dft in iHR:
                    if nspin == 1:
                        iH_spin_o1_o2 = iHR[key_dft][start1:start1+o1, start2:start2+o2][None, ...]
                        iH_block[atomtype_1][atomtype_2].append(iH_spin_o1_o2)
                    elif nspin > 1:
                        raise NotImplementedError("Interface for nspin>1 is under development")
                else:
                    raise KeyError(f"Can't find {key_dft} derived by ase in abacus's neighbor list")
            if has_SR:
                if key_dft in SR:
                    s_o1_o2 = SR[key_dft][start1:start1+o1, start2:start2+o2]
                    S_block[atomtype_1][atomtype_2].append(s_o1_o2)
                else:
                    s_o1_o2 = torch.zeros((o1,o2))
                    S_block[atomtype_1][atomtype_2].append(s_o1_o2)
                    #raise KeyError(f"Can't find {key_dft} derived by ase in abacus's neighbor list")

        for atomtype_1 in range(self.n_type):
            for atomtype_2 in range(self.n_type):
                winger_D_1 = self.get_wigner_D(self.AtomType_AMList[atomtype_1])
                winger_D_2 = self.get_wigner_D(self.AtomType_AMList[atomtype_2])
                if has_HR and (len(H_block[atomtype_1][atomtype_2]) != 0):
                    H_block[atomtype_1][atomtype_2] = torch.stack(H_block[atomtype_1][atomtype_2]).to(torch.float32)
                    # H_block[atomtype_1][atomtype_2] = torch.einsum("ij,zsjk,kl->zsil", winger_D_1.T, H_block[atomtype_1][atomtype_2], winger_D_2)
                    H_block[atomtype_1][atomtype_2] = torch.einsum("ij,zjk,kl->zil", winger_D_1.T, H_block[atomtype_1][atomtype_2], winger_D_2)
                if has_iHR and (len(iH_block[atomtype_1][atomtype_2]) != 0):
                    iH_block[atomtype_1][atomtype_2] = torch.stack(iH_block[atomtype_1][atomtype_2])
                    iH_block[atomtype_1][atomtype_2] = torch.einsum("ij,zsjk,kl->zsil", winger_D_1.T, iH_block[atomtype_1][atomtype_2], winger_D_2)
                if has_SR and (len(S_block[atomtype_1][atomtype_2]) != 0):
                    S_block[atomtype_1][atomtype_2] = torch.stack(S_block[atomtype_1][atomtype_2]).to(torch.float32)
                    S_block[atomtype_1][atomtype_2] = torch.einsum("ij,zjk,kl->zil", winger_D_1.T, S_block[atomtype_1][atomtype_2], winger_D_2)

        return H_block, iH_block, S_block

    def get_data(self):
        dataset = []
        for root, _, files in os.walk(self.dataset):
            HS_file = {"H0_file": None, "H1_file": None, "S_file": None}
            if "data-HR-sparse_SPIN0.csr" in files:
                HS_file["H0_file"] = os.path.join(root, "data-HR-sparse_SPIN0.csr")
            if "data-HR-sparse_SPIN1.csr" in files:
                HS_file["H1_file"] = os.path.join(root, "data-HR-sparse_SPIN1.csr")
            if "data-SR-sparse_SPIN0.csr" in files:
                HS_file["S_file"] = os.path.join(root, "data-SR-sparse_SPIN0.csr")
            if all([f is None for f in list(HS_file.values())]):
                continue

            structure = read(os.path.join(root, "../model.xyz"))

            # atom_type, n_type, lattice, position
            AtomType = torch.tensor([self.AtomNumber_to_AtomType[atomnumber] for atomnumber in structure.numbers])
            n_type = self.n_type
            lattice = torch.tensor(np.array(structure.cell))
            pos = torch.tensor(structure.positions)

            # atom_type's orbit number
            # Hamiltonian and overlap start index for each atom
            Node_OrbitalSum = self.AtomType_OrbitalSum[AtomType]
            offset = torch.cumsum(Node_OrbitalSum,dim=0)-Node_OrbitalSum
            TotalOrbital = Node_OrbitalSum.sum()

            # Hamiltonian (spin, key, orbit_0, oribit_1)
            # overlap     (key, orbit_0, oribit_1)
            HR, iHR, SR = self.get_HS(HS_file=HS_file, TotalOrbital=TotalOrbital)

            # 1.calculate and check neighbor list
            # 2.convert abacus's Hamiltonian and overlap to hotham's order
            #   Hamiltonian  (n_type, n_type, edge, spin, orbit_0, oribit_1)
            #   iHamiltonian (n_type, n_type, edge, spin, orbit_0, oribit_1)
            #   overlap      (n_type, n_type, edge,       orbit_0, oribit_1)
            cutoff = [self.para["cutoff"][symbol]*Bohr for symbol in structure.get_chemical_symbols()]
            _, _, d, D, S, edge_index, edge_inverse = self.find_neigbhor(frame=structure, cutoff=cutoff)
            unique_cell_shift, cell_shift_index = find_cell_shfit_index(S)
            HR, iHR, SR = self.abacus2hotham(HR=HR,
                                             iHR=iHR,
                                             SR=SR,
                                             AtomType=AtomType,
                                             edge_index=edge_index,
                                             cell_shift=S,
                                             offset=offset,
                                             TotalOrbital=TotalOrbital)

            # save as dict
            data=Data(
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
                unique_cell_shift=unique_cell_shift,
                cell_shift_index=cell_shift_index
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
                data["HR"] = HR
            if len(iHR) != 0:
                data["iHR"] = iHR
            if len(SR) != 0:
                data["SR"] = SR
            dataset.append(data)
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
        "trainset": "./tmp",
        "train_target": "hamiltonian",
        "dft": "abacus",
        "orbit": {
            "H": [
                "1s",
                "2s",
                "2p"
            ]
        },
        "cutoff": {
            "H": 6.0
        },
        "rc": 6,
        "L_max": 5,
        "using_CoordinateTransformation": True,
        "edge_include_sc": True,
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
