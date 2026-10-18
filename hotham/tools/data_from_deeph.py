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
import json5
import h5py


def find_inverse_index(I, J, S, numpy_int_dtype):
    index_inv = {}
    for index in range(len(I)):
        i, j = I[index], J[index]
        s1, s2, s3 = S[index]
        ijs = (i, j, s1, s2, s3)
        ijs_inv = (j, i, -s1, -s2, -s3)

        index_inv[ijs] = [index] + index_inv.setdefault(ijs, [])
        index_inv[ijs_inv] = index_inv.setdefault(ijs_inv, []) + [index]

    return np.array(sorted(index_inv.values()), dtype=numpy_int_dtype)[:, 1]


def find_cell_shfit_index(S, numpy_int_dtype):
    unique_S = np.unique(S, axis=0).astype(numpy_int_dtype, copy=False)
    unique_S_tuple = [tuple(s.tolist()) for s in unique_S]
    # unique_S_tuple_sort = sorted(unique_S_tuple)
    mapping = {s: i for i, s in enumerate(unique_S_tuple)}
    S_index = np.array([mapping[tuple(s.tolist())] for s in S], dtype=numpy_int_dtype)
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
        if self["int_dtype"] == "int32":
            self.numpy_int_dtype = np.int32
            self.torch_int_dtype = torch.int32
        elif self["int_dtype"] == "int64":
            self.numpy_int_dtype = np.int64
            self.torch_int_dtype = torch.int64
        if self["float_dtype"] == "float32":
            self.numpy_float_dtype = np.float32
            self.numpy_complex_dtype = np.complex64
            self.torch_float_dtype = torch.float32
            self.torch_complex_dtype = torch.complex64
        elif self["float_dtype"] == "float64":
            self.numpy_float_dtype = np.float64
            self.numpy_complex_dtype = np.complex128
            self.torch_float_dtype = torch.float64
            self.torch_complex_dtype = torch.complex128
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
        default_dict["device"] = "cpu"
        # Dataset path
        default_dict["trainset"] = None
        default_dict["testset"] = None
        default_dict["dft"] = None
        default_dict["edge_include_sc"] = True
        # Data preprocess
        default_dict["using_CoordinateTransformation"] = True
        return default_dict


class BasicInfo:
    def __init__(self, orbit: dict, device="cpu", intdtype=torch.int64):
        self.orbit = orbit
        self.device = device
        self.intdtype = intdtype

        # Atom type informatio
        self.AtomSymbol_to_AtomNumber = {atomsymbol: atomnumber for atomsymbol, atomnumber in atomic_numbers.items() if atomsymbol in orbit.keys()}
        self.AtomNumber_to_AtomSymbol = {atomnumber: atomsymbol for atomsymbol, atomnumber in self.AtomSymbol_to_AtomNumber.items()}
        unique_type = sorted(self.AtomNumber_to_AtomSymbol.keys())
        self.AtomNumber_to_AtomType = {num: i for i, num in enumerate(unique_type)}
        self.AtomSymbol_to_AtomType = {
            self.AtomNumber_to_AtomSymbol[atomnumber]: atomtype for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()
        }
        self.AtomType_to_AtomNumber = {atomtype: atomnumber for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()}
        self.AtomType_to_AtomSymbol = {
            atomtype: self.AtomNumber_to_AtomSymbol[atomnumber] for atomnumber, atomtype in self.AtomNumber_to_AtomType.items()
        }
        self.n_type = len(self.AtomType_to_AtomSymbol)

        # Orbit information
        self.AMSymbol_to_AM = {"s": 0, "p": 1, "d": 2, "f": 3, "g": 4}
        self.AtomType_AMSymbolList = {
            self.AtomSymbol_to_AtomType[k]: list(map(lambda x: "".join(re.findall(r"[A-Za-z]", x)), v)) for k, v in orbit.items()
        }
        self.AtomType_OrbitalSum = (
            torch.tensor(
                [sum(list(map(lambda x: 2 * self.AMSymbol_to_AM[x] + 1, self.AtomType_AMSymbolList[k]))) for k in sorted(self.AtomType_AMSymbolList)]
            )
            .to(self.intdtype)
            .to(self.device)
        )
        self.AtomType_AMList = {
            atomtype: torch.tensor(list(map(lambda x: self.AMSymbol_to_AM[x], amsymbollist)), dtype=self.intdtype)
            for atomtype, amsymbollist in self.AtomType_AMSymbolList.items()
        }
        self.AtomSymbol_to_AMList = {self.AtomType_to_AtomSymbol[atomtype]: amlist for atomtype, amlist in self.AtomType_AMList.items()}


class DataBase(ABC):
    def __init__(self, para, dataset):
        self.para = para
        self.dataset = dataset
        self.intdtype = para.torch_int_dtype
        self.floatdtype = para.torch_float_dtype
        self.complexdtype = para.torch_complex_dtype
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
        edge_inverse = find_inverse_index(i, j, S, self.para.numpy_int_dtype)
        return [torch.from_numpy(ele).to(device=self.device, dtype=self.floatdtype if ele.dtype.kind == "f" else self.intdtype) for ele in [i, j, d, D, S, edge_index, edge_inverse]]

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
        wigner_D = [[] for _ in range(lmax + 1)]
        for l in range(lmax + 1):
            D = self.wigner_D(l, alpha, beta, torch.zeros_like(alpha))
            wigner_D[l] = D
        return wigner_D

    def wigner_D(self, l, alpha, beta, gamma):
        if not l < len(self._Jd):
            raise NotImplementedError(f"wigner D maximum l implemented is {len(self._Jd) - 1}")

        alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
        J = self._Jd[l].to(dtype=alpha.dtype, device=alpha.device)
        Xa = self._z_rot_mat(alpha, l)
        Xb = self._z_rot_mat(beta, l)
        Xc = self._z_rot_mat(gamma, l)
        return Xa @ J @ Xb @ J @ Xc

    def _z_rot_mat(self, angle, l):
        shape, device, dtype = angle.shape, angle.device, angle.dtype
        M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
        inds = torch.arange(0, 2 * l + 1, 1, dtype=self.intdtype, device=device)
        reversed_inds = torch.arange(2 * l, -1, -1, dtype=self.intdtype, device=device)
        frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
        M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
        M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
        return M


class DeepHData(DataBase):
    def __init__(self, para: dict, dataset: str):
        super().__init__(para, dataset)
        self.dataset = self.get_data()

        if self.para.using_CoordinateTransformation:
            for data in self.dataset:
                data.wigner_D = self.get_wigner_Ds(self.para.L_max, data.D_hop[:, [1, 2, 0]])
                if self.para.edge_include_sc:
                    data.mask_edge = data.d_hop > 1.0e-6
                    data.mask_sc = ~data.mask_edge
                    assert data.mask_sc.sum() == data.num_nodes
                    for index in range(len(data.wigner_D)):
                        data.wigner_D[index][data.mask_sc] = torch.eye(
                            2 * index + 1, dtype=data.wigner_D[index].dtype, device=data.wigner_D[index].device
                        ).unsqueeze(0)

    def get_HS(self, path: str, spin: bool = False):
        HR = {}
        with h5py.File(path, "r") as fid:
            for k, v in fid.items():
                key = json5.loads(k)
                key = (key[0], key[1], key[2], key[3] - 1, key[4] - 1)
                v = np.array(v)  # first read to np.ndarray format to enable faster conversion to torch.Tensor
                if spin:
                    HR[key] = torch.tensor(v, dtype=self.complexdtype)
                else:
                    HR[key] = torch.tensor(v, dtype=self.floatdtype)
        return HR

    def get_wigner_D(self, order: Union[int, List[int]]):
        """
        D @ Y_wiki == Y_openmx
        """
        D = [
            np.array([[1.0]], dtype=self.para.numpy_float_dtype),
            np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=self.para.numpy_float_dtype),
            np.array([[0, 0, 1, 0, 0], [0, 0, 0, 0, 1], [1, 0, 0, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 0]], dtype=self.para.numpy_float_dtype),
            np.array(
                [
                    [0, 0, 0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 1, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1, 0],
                    [0, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 0, 0],
                ], dtype=self.para.numpy_float_dtype
            ),
        ]

        if isinstance(order, int):
            order = [order]

        DirectSum = scipy.linalg.block_diag(*[D[l] for l in order])
        return DirectSum

    def deeph2hotham(self, HR: dict, AtomType: np.array, cell_shift: np.array, edge_index: np.array, spin: bool):
        def create_pair_dict(symbols):
            return {s0: {s1: [] for s1 in symbols} for s0 in symbols}

        H_block = create_pair_dict(self.AtomSymbol_to_AtomType.keys())

        for index in range(edge_index.shape[1]):
            s1, s2, s3 = cell_shift[index]
            n1, n2 = edge_index.T[index]
            atomtype_1 = AtomType[n1].item()
            atomtype_2 = AtomType[n2].item()
            atomsymbol_1 = self.AtomType_to_AtomSymbol[atomtype_1]
            atomsymbol_2 = self.AtomType_to_AtomSymbol[atomtype_2]
            key = (s1.item(), s2.item(), s3.item(), n1.item(), n2.item())
            H_spin_o1_o2 = HR[key]
            H_block[atomsymbol_1][atomsymbol_2].append(H_spin_o1_o2)

        for atomtype_1 in range(self.n_type):
            atomsymbol_1 = self.AtomType_to_AtomSymbol[atomtype_1]
            winger_D_1 = self.get_wigner_D(self.AtomSymbol_to_AMList[atomsymbol_1])
            for atomtype_2 in range(self.n_type):
                atomsymbol_2 = self.AtomType_to_AtomSymbol[atomtype_2]
                winger_D_2 = self.get_wigner_D(self.AtomSymbol_to_AMList[atomsymbol_2])
                H_block[atomsymbol_1][atomsymbol_2] = np.stack(H_block[atomsymbol_1][atomsymbol_2])
                blocks = H_block[atomsymbol_1][atomsymbol_2]
                n, dim0, dim1 = blocks.shape
                winger_D_1 = winger_D_1.astype(blocks.dtype)
                winger_D_2 = winger_D_2.astype(blocks.dtype)

                if spin:
                    if dim0 != 2 * winger_D_1.shape[0] or dim1 != 2 * winger_D_2.shape[0]:
                        raise ValueError(f"Unexpected spinful Hamiltonian shape for {atomsymbol_1}-{atomsymbol_2}: {blocks.shape}")
                    # DeepH interleaves spin within each orbital: (E, m, 2, n, 2).
                    blocks = blocks.reshape(n, dim0 // 2, 2, dim1 // 2, 2)
                    blocks = np.einsum("ij,zjakb,kl->zialb", winger_D_1.T, blocks, winger_D_2)
                    # Channel order: up-up, up-down, down-up, down-down.
                    blocks = blocks.transpose(0, 2, 4, 1, 3).reshape(n, 4, dim0 // 2, dim1 // 2)
                else:
                    blocks = np.einsum("ij,zjk,kl->zil", winger_D_1.T, blocks, winger_D_2)
                    blocks = blocks[:, None, :, :]
                H_block[atomsymbol_1][atomsymbol_2] = blocks

        return H_block

    def get_data(self):
        dataset = []
        paths = []
        for root, dirname, files in os.walk(self.dataset, followlinks=True):
            if {"element.dat", "orbital_types.dat", "lat.dat", "site_positions.dat"}.issubset(files):
                paths.append(root)

        for root in tqdm(paths):
            # atom type
            AtomNumber = np.loadtxt(os.path.join(root, "element.dat"), dtype=self.para.numpy_int_dtype)
            AtomType = np.array([self.AtomNumber_to_AtomType[num] for num in AtomNumber], dtype=self.para.numpy_int_dtype)
            n_type = self.n_type

            # lattice
            lattice = np.loadtxt(os.path.join(root, "lat.dat"), dtype=self.para.numpy_float_dtype).T

            # site positions
            pos = np.loadtxt(os.path.join(root, "site_positions.dat"), dtype=self.para.numpy_float_dtype).T

            # info.json
            with open(os.path.join(root, "info.json"), "r") as fid:
                spin: bool = json5.load(fid)["isspinful"]

            # Hamiltonian (key, orbit_0*(spin+1), oribit_1*(spin+1))
            HR = self.get_HS(path=os.path.join(root, "hamiltonians.h5"), spin=spin)

            # infer neighbor list from HR
            keys = np.array(list(HR.keys()))
            cell_shift, edge_index = keys[:, :3], keys[:, 3:].T
            inv_edge_index = find_inverse_index(I=edge_index[0], J=edge_index[1], S=cell_shift, numpy_int_dtype=self.para.numpy_int_dtype)
            unique_cell_shift, cell_shift_index = find_cell_shfit_index(cell_shift, self.para.numpy_int_dtype)
            D = pos[edge_index[1]] - pos[edge_index[0]] + cell_shift @ lattice
            d = np.linalg.norm(D, axis=1)

            # convert deeph's Hamiltonian and overlap to hotham's order
            HR = self.deeph2hotham(HR=HR, AtomType=AtomType, cell_shift=cell_shift, edge_index=edge_index, spin=spin)

            # atom_type's orbit number
            # Hamiltonian start index for each atom
            AtomType_OrbitalSum = self.AtomType_OrbitalSum
            offset = torch.cumsum(self.AtomType_OrbitalSum[AtomType], dim=0) - self.AtomType_OrbitalSum[AtomType]

            # save as dict
            data = Data(
                AtomType=torch.from_numpy(AtomType).to(self.intdtype),
                AtomType_OrbitalSum=self.AtomType_OrbitalSum,
                offset=offset,
                n_type=n_type,
                lattice=torch.from_numpy(lattice).to(self.floatdtype),
                pos=torch.from_numpy(pos).to(self.floatdtype),
                edge_index_hop=torch.from_numpy(edge_index).to(self.intdtype),
                edge_inverse=torch.from_numpy(inv_edge_index).to(self.intdtype),
                D_hop=torch.from_numpy(D).to(self.floatdtype),
                d_hop=torch.from_numpy(d).to(self.floatdtype),
                S_hop=torch.from_numpy(cell_shift).to(self.intdtype),
                unique_cell_shift=torch.from_numpy(unique_cell_shift).to(self.intdtype),
                cell_shift_index=torch.from_numpy(cell_shift_index).to(self.intdtype),
                spin=spin,
            )

            # HR
            data["HR"] = numpy2tensor(HR, "cpu")

            # magmon
            if os.path.exists(os.path.join(root, "magmom.txt")):
                magmom = torch.tensor(np.loadtxt(os.path.join(root, "magmom.txt"), dtype=self.para.numpy_float_dtype), dtype=self.floatdtype, device=self.device)
                data["magmom"] = magmom
            dataset.append(data)
        return dataset


if __name__ == "__main__":
    inputfile = {
        "int_dtype": "int64",
        "float_dtype": "float32",
        "trainset": "/data/home/js_liangzx/cc-test/xDeepH/data/100_10",
        # "trainset": "/data/home/js_liangzx/cc-test/xDeepH/data/",
        # "testset": "./data/testset",
        "dft": "deeph",
        "orbit": {"Cr": ["1s", "2s", "3s", "2p", "3p", "3d"], "I": ["1s", "2s", "3s", "2p", "3p", "3d", "4d"]},
        "L_max": 5,
        # only used by AbacusData
        "cutoff": {
            "Cr": 8,
            "I": 8,
        },
    }

    param = Parameters(inputfile)
    DATACLASS = DeepHData

    for dataset in ["trainset", "testset"]:
        if param[dataset] is not None:
            data = DATACLASS(param, param[dataset])
            torch.save(tensor2device(data.dataset, "cpu"), dataset + ".pth")
