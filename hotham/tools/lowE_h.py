import os
import h5py
import torch
import spglib
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase import Atoms
from ase.io import read
from ase.neighborlist import neighbor_list
from typing import Literal
from collections import Counter
from sklearn.cluster import KMeans
from scipy.sparse import csc_matrix

def load_csc_matrix(npz_path: str) -> sp.csc_matrix:
    loader = np.load(npz_path)
    data = loader["data"]
    indices = loader["indices"]
    indptr = loader["indptr"]
    shape = tuple(loader["shape"])
    matrix = sp.csc_matrix((data, indices, indptr), shape=shape)
    return matrix

def seek_kpath(structure: Atoms, path: str, pbc: list, npoints: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    band_path = structure.cell.bandpath(path=path, pbc=pbc, npoints=npoints)
    kpoints_frac = band_path.kpts
    kpoints_cart = kpoints_frac @ band_path.icell
    kpath_distances = np.insert(np.cumsum(np.linalg.norm(np.diff(kpoints_cart, axis=0), axis=1)), 0, 0)
    special_points_frac = np.array([band_path.special_points[p] for p in band_path.path])
    special_points_cart = special_points_frac @ band_path.icell
    special_k_distances = np.insert(np.cumsum(np.linalg.norm(np.diff(special_points_cart, axis=0), axis=1)), 0, 0)
    return kpoints_frac, kpath_distances, special_k_distances

def get_dense_grid(structure: Atoms, mesh: list, using_ir_mesh: bool) -> tuple[np.ndarray, np.ndarray]:
    cell = (structure.cell.array, structure.get_scaled_positions(), structure.numbers)
    grid_mapping_table, grid_address = spglib.get_ir_reciprocal_mesh(mesh, cell)
    knum = len(grid_mapping_table)
    if using_ir_mesh:
        ir_counter = Counter(grid_mapping_table)
        kweight = np.array(list(ir_counter.values()))/knum
        kpoints = grid_address[np.array(list(ir_counter.keys()))] / np.array(mesh, dtype=float)
    else:
        kweight = np.ones(shape=knum)/knum
        kpoints = grid_address / np.array(mesh, dtype=float)
    return kpoints, kweight

def get_element_orbit_dict(orbitals: dict) -> dict:
    degeneracy = {"s": 1, "p": 3, "d": 5, "f": 7}
    orbital_dict = {}
    for element, orbs in orbitals.items():
        total = 0
        for orb in orbs:
            l = orb[-1]
            total += degeneracy[l]
        orbital_dict[element] = total
    return orbital_dict

def split_layers(structure: Atoms) -> tuple[list[str], list[Atoms], list[float]]:
    tol = 1.2 #Ångstrom
    layer_atoms = []
    if set(structure.get_chemical_symbols()) == {"Mo", "Te"}:
        #direct split
        mo_z = sorted(structure.get_positions()[i, 2] for i, s in enumerate(structure.get_chemical_symbols()) if s == "Mo")
        mo_layers, current = [], [mo_z[0]]
        for z in mo_z[1:]:
            if abs(z - np.mean(current)) < tol:
                current.append(z)
            else:
                mo_layers.append(current)
                current = [z]
        mo_layers.append(current)
        n_layers = len(mo_layers)
        total = len(structure)
        chunk = total // n_layers
        for i in range(n_layers):
            start = i * chunk
            end = (i + 1) * chunk if i < n_layers - 1 else total
            layer_atoms.append(structure[start:end])
    elif set(structure.get_chemical_symbols()) <= {"C", "B", "N"}:
        z_coords = structure.get_positions()[:, 2]
        layers, current = [], [0]
        for i in range(1, len(z_coords)):
            if abs(z_coords[i] - np.mean(z_coords[current])) < tol:
                current.append(i)
            else:
                layers.append(current)
                current = [i]
        layers.append(current)
        layer_atoms = [structure[idx] for idx in layers]

    layer_system = []
    avg_bond_lengths = []
    for layer in layer_atoms:
        s = set(layer.get_chemical_symbols())
        if s <= {"C"}:
            layer_system.append("graphene")
        elif s <= {"B", "N"}:
            layer_system.append("h-BN")
        elif "Mo" in s:
            layer_system.append("MoTe2")
        dists = layer.get_all_distances(mic=True)
        nearest = []
        for i in range(len(layer)):
            row = dists[i].copy()
            row[i] = np.inf
            min_dist = np.min(row[row > 1e-6]) 
            nearest.append(min_dist)
        avg_bond_lengths.append(float(np.mean(nearest)))
    return layer_system, layer_atoms, avg_bond_lengths

def compute_dos(eigenvalues: np.ndarray, kweight: np.ndarray, e_grid: np.ndarray, sigma: float) -> np.ndarray:
    dos = np.zeros_like(e_grid)
    for evals, w in zip(eigenvalues, kweight):
        diff = e_grid[:, None] - evals[None, :]
        dos += w * np.sum(np.exp(-1 * (diff / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi)), axis=1)
    return dos

class LowEnergyHamiltonian():
    def __init__(self, inputfile: dict):
        self.inputfile = inputfile
        self.device = inputfile["device"]
        self.numpy_dtype_int = np.int32
        if self.inputfile["precision"] == "float32":
            self.numpy_dtype_float = np.float32
            self.numpy_dtype_complex = np.complex64
            self.torch_dtype_float = torch.float32
            self.torch_dtype_complex = torch.complex64
        elif self.inputfile["precision"] == "float64":
            self.numpy_dtype_float = np.float64
            self.numpy_dtype_complex = np.complex128
            self.torch_dtype_float = torch.float64
            self.torch_dtype_complex = torch.complex128
        
        self.structure = read(self.inputfile["structure"])
        self.layer_system, self.layer_atoms, self.avg_bond_lengths = split_layers(structure=self.structure)
        self.num_layers = len(self.layer_system)
        print("Layer systems:", self.layer_system)
        print("Number of Atoms per layer:", [len(atoms) for atoms in self.layer_atoms])
        print("Average bond lengths per layer:", [f"{x:.6f}" for x in self.avg_bond_lengths])

        self.orbitals = inputfile["orbitals"]
        self.element_num_dict = self.structure.symbols.formula.count()
        #element_num_dict {C:100,...}
        self.element_orbital_dict = get_element_orbit_dict(self.orbitals)
        #element_orbital_dict {C:13,...}
        self.dim_origin = sum(self.element_num_dict[elem] * self.element_orbital_dict[elem] for elem in self.element_num_dict)
        self.dim_TAPW = None
        self.dim_lowE = None
        self.num_Gpoints_per_layer = None
        print("Dimension of H_origin:", self.dim_origin,"*",self.dim_origin)

        self.fermi_energy = self.inputfile["fermi_energy"]
        self.matrix_source = self.inputfile["matrix_source"]
        #==========valley_projection part==========#
        self.project_harmonics = self.inputfile["project_harmonics"] - 1
        #==========orbital_projection part==========#        
        self.selected_states_index = self.inputfile["selected_states_index"]

    def Bloch_gauge_to_Wannier_gauge(self, Mk: sp.csc_matrix, kpoint: np.ndarray) -> sp.csc_matrix:
        atom_positions_frac = self.structure.get_scaled_positions()
        orbitals_per_atom = [self.element_orbital_dict[atom.symbol] for atom in self.structure]
        orbital_positions_frac = np.repeat(atom_positions_frac, orbitals_per_atom, axis=0)
        phase = 2 * np.pi * (orbital_positions_frac @ kpoint)
        Mk_coo = Mk.tocoo()
        row_idx, col_idx, data = Mk_coo.row, Mk_coo.col, Mk_coo.data
        phase_difference = phase[row_idx] - phase[col_idx]
        phase_factor = np.exp(-1j * phase_difference)
        new_data = data * phase_factor
        return sp.csc_matrix((new_data, (row_idx, col_idx)), shape=Mk.shape)
    
    def get_Hk_and_Sk(self, kmode: Literal["band","dos"], ik: int, kpoint: np.ndarray) -> tuple[sp.csc_matrix, sp.csc_matrix]:
        if kmode == "band":
            root_dir = self.inputfile["matrix_band_dir"]
        elif kmode == "dos":
            root_dir = self.inputfile["matrix_dos_dir"]
        kdir = os.path.join(root_dir, f"k{ik}")
        if self.matrix_source == "OpenMX":
            H_path = os.path.join(kdir, f"k{ik}_Href.npz")
        elif self.matrix_source == "Hot-Ham":
            H_path = os.path.join(kdir, f"k{ik}_H.npz")
        S_path = os.path.join(kdir, f"k{ik}_S.npz")
        Hk = load_csc_matrix(H_path)
        Sk = load_csc_matrix(S_path)
        Hk = Hk + Hk.T.conj() - sp.diags(Hk.diagonal())
        Sk = Sk + Sk.T.conj() - sp.diags(Sk.diagonal())
        Hk = self.Bloch_gauge_to_Wannier_gauge(Hk, kpoint)
        Sk = self.Bloch_gauge_to_Wannier_gauge(Sk, kpoint)
        return Hk, Sk

    def get_primitive_cell(self) -> list[np.ndarray]:
        layer_angles_all = []
        for structure in self.layer_atoms:
            symbols = np.array(structure.get_chemical_symbols())
            elements = set(symbols)
            i, j, S = neighbor_list('ijS', structure, 5.0)
            cell = structure.get_cell()
            dr = structure.positions[j] + np.dot(S, cell) - structure.positions[i]
            d = np.linalg.norm(dr, axis=1)
            if elements == {"C"}:
                d_ref = d[d > 0.5]
                pair_mask = np.ones_like(d, dtype=bool)
            elif elements == {"B", "N"}:
                pair_mask = (((symbols[i] == 'B') & (symbols[j] == 'N')) | ((symbols[i] == 'N') & (symbols[j] == 'B')))
                d_ref = d[pair_mask & (d > 0.5)]
            elif elements == {"Mo", "Te"}:
                pair_mask = (symbols[i] == 'Mo') & (symbols[j] == 'Mo')
                d_ref = d[pair_mask & (d > 1.0)]

            d_sorted = np.sort(d_ref)
            d_first = d_sorted[0]
            mask_dist = (d > d_first * 0.8) & (d < d_first * 1.2)
            mask = pair_mask & mask_dist

            dr = dr[mask].astype(np.float64)
            dr2d = dr[:, :2].copy()
            norms = np.linalg.norm(dr2d, axis=1, keepdims=True)
            valid = norms[:, 0] > 1e-6
            dr2d = dr2d[valid]
            norms = norms[valid]
            dr2d /= norms
            dr2d_folded = np.where(dr2d[:, 0:1] < 0, -dr2d, dr2d)
            edge = (dr2d_folded[:, 0] == 0) & (dr2d_folded[:, 1] < 0)
            dr2d_folded[edge] *= -1
            dirs = KMeans(n_clusters=3, random_state=0, n_init='auto').fit(dr2d_folded).cluster_centers_
            dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
            angles = np.degrees(np.arctan2(dirs[:, 1], dirs[:, 0]))
            angles_full = np.concatenate([angles, angles + 180])
            angles_full = (angles_full + 360) % 360
            angles_full = np.sort(angles_full)
            layer_angles_all.append(angles_full)

        cells = []
        def build_cell(a: float, angle_deg: float) -> np.ndarray:
            a = np.float64(a)
            phi1 = np.radians(angle_deg)
            phi2 = phi1 - np.radians(120.0)
            a1 = np.array([a * np.cos(phi1), a * np.sin(phi1), 0.0], dtype=self.numpy_dtype_float)
            a2 = np.array([a * np.cos(phi2), a * np.sin(phi2), 0.0], dtype=self.numpy_dtype_float)
            a3 = np.array([0.0, 0.0, 40.0], dtype=self.numpy_dtype_float)
            return np.array([a1, a2, a3])

        for idx, (structure, angles) in enumerate(zip(self.layer_atoms, layer_angles_all)):
            elements = set(structure.get_chemical_symbols())
            if elements == {"C"}:
                a = self.avg_bond_lengths[idx] * np.sqrt(3.0)
                ref = angles[0] + 30.0
            elif elements == {"B", "N"}:
                a = self.avg_bond_lengths[idx] * np.sqrt(3.0)
                ref = angles[0] + 30.0
            elif elements == {"Mo", "Te"}:
                a = 3.52
                ref = angles[0]
                #stupid
                if idx == 0:
                    ref += 60
                elif idx == 1:
                    ref = ref
                else:
                    ref += 60 * (idx % 2)
            cell = build_cell(a, ref)
            cells.append(cell)
        return cells

    def get_Gpoint(self, project_target: Literal["K","K'"]) -> np.ndarray:
        cells = self.get_primitive_cell()
        K_frac = np.array([1/3, 1/3, 0], dtype=self.numpy_dtype_float)
        K_points_all = []
        for cell in cells:
            rec = 2 * np.pi * np.linalg.inv(cell).T
            K_cart = K_frac @ rec
            K_points_all.append(K_cart)

        # only support one twist
        method = 2
        # method1, especially for unrelaxed structure, K and K′ are not symmetric about the x-axis.
        if method == 1:
            if self.inputfile["precision"] == "float32":
                K_cart_unique = [k for i, k in enumerate(K_points_all) if not any(np.allclose(k, K_points_all[j], atol=1e-6) for j in range(i))]
            elif self.inputfile["precision"] == "float64":
                K_cart_unique = [k for i, k in enumerate(K_points_all) if not any(np.allclose(k, K_points_all[j], atol=1e-12) for j in range(i))]
        # method2, especially for relaxed structure, K and K′ are symmetric about the x-axis.
        elif method == 2:
            K_cart_unique = []
            positive_found = False
            negative_found = False
            for K_cart in K_points_all:
                if (not positive_found) and K_cart[1] > 0:
                    K_cart_unique.append(K_cart)
                    positive_found = True
                elif (not negative_found) and K_cart[1] < 0:
                    K_cart_unique.append(K_cart)
                    negative_found = True
                if positive_found and negative_found:
                    break

        assert len(K_cart_unique) == 2, "Expected two unique K points, but found {}".format(len(K_cart_unique))
        q1 = K_cart_unique[1] - K_cart_unique[0]
        
        theta = np.radians(120)
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
            [np.sin(theta),  np.cos(theta), 0],
            [0,              0,             1]], dtype=self.numpy_dtype_float)
        q2 = R @ q1
        q3 = R @ q2
        bm1 = q1 + 2*q3
        bm2 = 2*q1 + q3

        hexagon_centers = []
        for n1 in range(-self.project_harmonics, self.project_harmonics+1):
            for n2 in range(-self.project_harmonics, self.project_harmonics+1):
                if max(abs(n1), abs(n2), abs(n1 + n2)) <= self.project_harmonics:
                    R = n1 * bm1 + n2 * bm2
                    hexagon_centers.append(R)
        hexagon_centers = np.array(hexagon_centers)
        hexagon_vertices = np.concatenate((hexagon_centers+q1, hexagon_centers+q2, hexagon_centers+q3), axis=0)
        if self.inputfile["precision"] == "float32":
            _, unique_idx = np.unique(np.round(hexagon_vertices, decimals=6), axis=0, return_index=True)
        elif self.inputfile["precision"] == "float64":
            _, unique_idx = np.unique(np.round(hexagon_vertices, decimals=12), axis=0, return_index=True)
        Q_cart = hexagon_vertices[unique_idx]
        Q_cart = Q_cart[np.argsort(np.linalg.norm(Q_cart, axis=1))]
        
        G_points_all = []
        Q_points_all = []
        for _, K_cart_layer in enumerate(K_points_all):
            if method == 1:
                if np.allclose(K_cart_layer, K_cart_unique[0]):
                    Q_cart_layer = Q_cart
                elif np.allclose(K_cart_layer, K_cart_unique[1]):
                    Q_cart_layer = -Q_cart
                else:
                    raise ValueError("error in matching K_layer to K_unique.")
            if method == 2:
                # may be wrong
                if K_cart_layer[1] > 0:
                    Q_cart_layer = Q_cart
                elif K_cart_layer[1] < 0:
                    Q_cart_layer = -Q_cart
                else:
                    raise ValueError("error in matching K_layer to K_unique.")

            G_cart_layer = K_cart_layer - Q_cart_layer
            G_points_all.append(G_cart_layer)
            Q_points_all.append(Q_cart_layer)
        print("Project target:", project_target)

        if project_target == "K'":
            G_points_all = [-G for G in G_points_all]
            Q_points_all = [-Q for Q in Q_points_all]
            K_points_all = [-K for K in K_points_all]
        G_points = np.concatenate(G_points_all, axis=0)
        Q_points = np.concatenate(Q_points_all, axis=0)
        K_points = np.array(K_points_all)

        self.num_Gpoints_per_layer = len(G_points) // self.num_layers
        print("num G per layer:", self.num_Gpoints_per_layer)

        fig, ax = plt.subplots(1, 2, figsize=(12, 6))
        groups = []
        for i, G in enumerate(G_points_all):
            found = False
            for g in groups:
                if np.allclose(G, g["G"], atol=1e-4):
                    g["indices"].append(i)
                    found = True
                    break
            if not found:
                groups.append({"G": G, "Q": Q_points_all[i], "K": K_points_all[i], "indices": [i]})
        def draw_hexagon(ax, K_cart, color):
            r = np.linalg.norm(K_cart)
            a = np.arctan2(K_cart[1], K_cart[0])
            verts = np.array([[r * np.cos(a + n * np.pi / 3), r * np.sin(a + n * np.pi / 3)] for n in range(7)])
            ax.plot(verts[:, 0], verts[:, 1], color=color, lw=1.2, linestyle='--', alpha=0.6)
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        # ===== 左图：G points =====
        for idx, g in enumerate(groups):
            G_group = g["G"]
            K_cart = g["K"]
            label = f"layer{g['indices']}"
            color = colors[idx % len(colors)]
            offset = (5, 5 if idx % 2 == 0 else -10)
            ax[0].scatter(G_group[:, 0], G_group[:, 1], c=color, s=60, label=label, zorder=3)
            for i, point in enumerate(G_group):
                x, y, _ = point
                ax[0].annotate(f'G{i+1}', (x, y), xytext=offset, textcoords='offset points', c=color, fontsize=8)
            ax[0].quiver(0, 0, K_cart[0], K_cart[1], angles='xy', scale_units='xy', scale=1, color='black', width=0.005)
            ax[0].annotate(f'K_{label}', (K_cart[0], K_cart[1]), xytext=(5, 5), textcoords='offset points', c=color)
            draw_hexagon(ax[0], K_cart, color)
        # ===== 右图：q vectors =====
        for i, v in enumerate([q1, q2, q3], 1):
            ax[1].quiver(0, 0, v[0], v[1], angles='xy', scale_units='xy', scale=1, color='black', width=0.005)
            ax[1].annotate(f'q{i}', (v[0], v[1]), xytext=(5, 5), textcoords='offset points')
        # ===== 右图：Q points=====
        for idx, g in enumerate(groups):
            Q_group = g["Q"]
            label = f"layer{g['indices']}"
            color = colors[idx % len(colors)]
            ax[1].scatter(Q_group[:, 0], Q_group[:, 1], c=color, s=60, label=label)
        # ===== 公共设置 =====
        for a, t in zip(ax, ['G points', 'Q points']):
            a.axhline(0, c='gray', lw=0.5)
            a.axvline(0, c='gray', lw=0.5)
            a.set_aspect('equal')
            a.legend()
            a.set_title(t)
        plt.tight_layout()
        plt.savefig('G_points.png', dpi=150)
        return G_points
    
    def Projector_X(self, G_points: np.ndarray) -> sp.csc_matrix:
        rows_list = []
        cols_list = []
        vals_list = []
        row_offset, col_offset = 0, 0
        for layer_idx in range(self.num_layers):
            atoms_in_layer = self.layer_atoms[layer_idx]
            element_species = atoms_in_layer.get_chemical_symbols()
            if set(element_species) == {"C"}:
                num_atoms_per_unitcell = 2
                num_orbitals_per_unitcell = self.element_orbital_dict["C"]* num_atoms_per_unitcell
            if set(element_species) == {"B", "N"}:
                num_atoms_per_unitcell = 2
                num_orbitals_per_unitcell = self.element_orbital_dict["B"] + self.element_orbital_dict["N"]
            if set(element_species) == {"Mo", "Te"}:
                num_atoms_per_unitcell = 3
                num_orbitals_per_unitcell = self.element_orbital_dict["Mo"] + 2 * self.element_orbital_dict["Te"]
            
            cell_species = [atoms_in_layer[i].symbol for i in range(num_atoms_per_unitcell)]
            # ["C","C"] or ["Mo","Te","Te"]
            orb_per_atom = [self.element_orbital_dict[sp] for sp in cell_species]
            # [13,13] or [13,13,13]
            orb_offset = [sum(orb_per_atom[:i]) for i in range(len(orb_per_atom))]
            # [0,13,26,....]
            num_unitcells_per_layer = len(atoms_in_layer) // num_atoms_per_unitcell
            for G_idx in range(self.num_Gpoints_per_layer):
                G_vec = G_points[layer_idx * self.num_Gpoints_per_layer + G_idx]
                for unitcell_idx in range(num_unitcells_per_layer):
                    for sublat_idx, num_orb in enumerate(orb_per_atom):
                        atom_idx = unitcell_idx * num_atoms_per_unitcell + sublat_idx
                        tau = atoms_in_layer.positions[atom_idx]
                        phase = np.exp(1j * np.dot(G_vec, tau)) / np.sqrt(num_unitcells_per_layer)
                        row = row_offset + unitcell_idx * num_orbitals_per_unitcell + orb_offset[sublat_idx]
                        col = col_offset + G_idx * num_orbitals_per_unitcell + orb_offset[sublat_idx]
                        rows_list.append(np.arange(row, row + num_orb))
                        cols_list.append(np.arange(col, col + num_orb))
                        vals_list.append(np.full(num_orb, phase))
            row_offset += num_unitcells_per_layer * num_orbitals_per_unitcell
            col_offset += self.num_Gpoints_per_layer * num_orbitals_per_unitcell
        self.dim_TAPW = col_offset
        print("Dimension of H_TAPW:", self.dim_TAPW,"*",self.dim_TAPW)
        rows_arr = np.concatenate(rows_list).astype(self.numpy_dtype_int)
        cols_arr = np.concatenate(cols_list).astype(self.numpy_dtype_int)
        vals_arr = np.concatenate(vals_list).astype(self.numpy_dtype_complex)
        X = csc_matrix((vals_arr, (rows_arr, cols_arr)), shape=(self.dim_origin, self.dim_TAPW))
        return X
    
    def get_H_TAPW(self, Hk: sp.csc_matrix, Sk: sp.csc_matrix, X: sp.csc_matrix) -> torch.Tensor:
        H_TAPW = X.conj().T @ Hk @ X
        S_TAPW = X.conj().T @ Sk @ X
        H = torch.from_numpy(H_TAPW.toarray()).to(dtype=self.torch_dtype_complex, device=self.device)
        S = torch.from_numpy(S_TAPW.toarray()).to(dtype=self.torch_dtype_complex, device=self.device)
        eigvals, eigvecs = torch.linalg.eigh(S)
        S_inv_sqrt = eigvecs @ torch.diag((eigvals.to(eigvecs.dtype))**-0.5) @ eigvecs.conj().T
        H_lowdin = S_inv_sqrt @ H @ S_inv_sqrt
        return H_lowdin
    
    def get_H_lowE(self, H_lowdin: torch.Tensor) -> tuple[torch.Tensor, list[np.ndarray]]:
        U0 = torch.zeros((self.dim_TAPW, self.dim_lowE), dtype=self.torch_dtype_complex, device=self.device)
        use_second_order_perturbation = True #keep unchanged unless necessary
        if use_second_order_perturbation:
            U1 = torch.zeros((self.dim_TAPW, self.dim_TAPW - self.dim_lowE), dtype=self.torch_dtype_complex, device=self.device)

        eigvals_all_Q = []
        col_offset1, col_offset2 = 0, 0
        for layer_idx in range(self.num_layers):
            atoms_in_layer = self.layer_atoms[layer_idx]
            element_species = atoms_in_layer.get_chemical_symbols()
            if set(element_species) == {"C"}:
                num_atoms_per_unitcell = 2
                num_orbitals_per_unitcell = self.element_orbital_dict["C"]* num_atoms_per_unitcell
            if set(element_species) == {"B", "N"}:
                num_atoms_per_unitcell = 2
                num_orbitals_per_unitcell = self.element_orbital_dict["B"] + self.element_orbital_dict["N"]
            if set(element_species) == {"Mo", "Te"}:
                num_atoms_per_unitcell = 3
                num_orbitals_per_unitcell = self.element_orbital_dict["Mo"] + 2 * self.element_orbital_dict["Te"]
            chosen_indices = torch.tensor(self.selected_states_index[layer_idx])
            high_mask = torch.ones(num_orbitals_per_unitcell, dtype=torch.bool)
            high_mask[chosen_indices] = False

            for Q_idx in range(self.num_Gpoints_per_layer):
                start_row = layer_idx * self.num_Gpoints_per_layer * num_orbitals_per_unitcell + Q_idx * num_orbitals_per_unitcell
                end_row   = layer_idx * self.num_Gpoints_per_layer * num_orbitals_per_unitcell + (Q_idx + 1) * num_orbitals_per_unitcell
                D_Q = H_lowdin[start_row:end_row, start_row:end_row]
                eigenvalues, eigenvectors = torch.linalg.eigh(D_Q)
                eigvals_all_Q.append(eigenvalues.detach().cpu().numpy())
                u_low_block = eigenvectors[:, chosen_indices]
                start_col_1 = col_offset1
                col_offset1 += len(self.selected_states_index[layer_idx])
                end_col_1 = col_offset1
                U0[start_row:end_row,start_col_1:end_col_1] = u_low_block
                if use_second_order_perturbation:
                    dim_high_per_Q = num_orbitals_per_unitcell - len(self.selected_states_index[layer_idx])
                    u_high_block = eigenvectors[:, high_mask]
                    start_col_2 = col_offset2
                    col_offset2 += dim_high_per_Q
                    end_col_2 = col_offset2
                    U1[start_row:end_row,start_col_2:end_col_2] = u_high_block
        if use_second_order_perturbation:
            h00 = U0.conj().T @ H_lowdin @ U0
            h11 = U1.conj().T @ H_lowdin @ U1
            h01 = U0.conj().T @ H_lowdin @ U1
            #lambda_00, _ = torch.linalg.eigh(h00)
            #lambda_k = lambda_00.mean().to(self.torch_dtype_complex)
            #resolvent = torch.linalg.inv(lambda_k * torch.eye(h11.shape[0], dtype=self.torch_dtype_complex, device=self.device) - h11)
            resolvent = torch.linalg.inv(self.fermi_energy * torch.eye(h11.shape[0], dtype=self.torch_dtype_complex, device=self.device) - h11)
            h_DFT = h00 + h01 @ resolvent @ h01.conj().T
        else:
            h_DFT = U0.conj().T @ H_lowdin @ U0
        return h_DFT, eigvals_all_Q
    
    def run(self, kmode: Literal["band","dos"], kpoints_frac: np.ndarray, project_target: Literal["K","K'"], Hamiltonian_target: Literal["H_TAPW","H_lowE"]):
        Gpoints = self.get_Gpoint(project_target=project_target)
        X = self.Projector_X(Gpoints)
        with h5py.File(f"{Hamiltonian_target}_{kmode}_{project_target}.h5", "w") as h5_file:
            h5_file.attrs["format"] = "upper_triangular_csc"
            if Hamiltonian_target == "H_TAPW":
                for ik, kpoint in enumerate(tqdm(kpoints_frac, desc="Running TAPW projection")):
                    Hk, Sk = self.get_Hk_and_Sk(kmode, ik, kpoint)
                    H_lowdin = self.get_H_TAPW(Hk, Sk, X)
                    H_sparse = sp.triu(sp.csc_matrix(H_lowdin.detach().cpu().numpy()), format="csc")
                    k_group = h5_file.create_group(f"k{ik}")
                    k_group.create_dataset("data", data=H_sparse.data)
                    k_group.create_dataset("indices", data=H_sparse.indices)
                    k_group.create_dataset("indptr", data=H_sparse.indptr)
                    k_group.create_dataset("shape", data=H_sparse.shape)
                    h5_file.flush()

            if Hamiltonian_target == "H_lowE":
                self.dim_lowE = self.num_Gpoints_per_layer * sum(len(states) for states in self.selected_states_index)
                print("Dimension of H_lowE:", self.dim_lowE,"*",self.dim_lowE)
                for ik, kpoint in enumerate(tqdm(kpoints_frac, desc="Running lowE projection")):
                    Hk, Sk = self.get_Hk_and_Sk(kmode, ik, kpoint)
                    H_lowdin = self.get_H_TAPW(Hk, Sk, X)
                    if ik == 0:
                        h_dft, eigvals_all_Q= self.get_H_lowE(H_lowdin)
                        plt.figure(figsize=(8, 4))
                        num_Q = len(eigvals_all_Q)
                        for layer_idx in range(self.num_layers):
                            start_Q = layer_idx * self.num_Gpoints_per_layer
                            end_Q = (layer_idx + 1) * self.num_Gpoints_per_layer if layer_idx < self.num_layers - 1 else num_Q
                            states_idx = self.selected_states_index[layer_idx]
                            for q_idx in range(start_Q, end_Q):
                                ev = eigvals_all_Q[q_idx]
                                plt.scatter(np.full(len(ev), q_idx), ev - self.fermi_energy, s=5)
                                plt.scatter(np.full(len(states_idx), q_idx), ev[states_idx] - self.fermi_energy, s=40, facecolors='none', edgecolors='red', linewidths=1.2)
                        plt.xlabel("q index")
                        plt.ylabel("Energy (eV)")
                        plt.ylim(-8, 8)
                        plt.title(f"Eigenvalues vs q (k={ik})")
                        plt.savefig("eigenvalues_vs_q.png", dpi=150)
                    else:
                        h_dft, _ = self.get_H_lowE(H_lowdin)
                    H_sparse = sp.triu(sp.csc_matrix(h_dft.detach().cpu().numpy()), format="csc")
                    k_group = h5_file.create_group(f"k{ik}")
                    k_group.create_dataset("data", data=H_sparse.data)
                    k_group.create_dataset("indices", data=H_sparse.indices)
                    k_group.create_dataset("indptr", data=H_sparse.indptr)
                    k_group.create_dataset("shape", data=H_sparse.shape)
                    h5_file.flush()

    def _load_ref_band(self) -> np.ndarray:
        ref_data = np.load(self.inputfile["ref_band_file"], allow_pickle=True).item()
        if self.matrix_source == "OpenMX":
            eigenvalues_DFT = ref_data["eigenvalues_ref"]
        elif self.matrix_source == "Hot-Ham":
            eigenvalues_DFT = ref_data["eigenvalues"]
        return eigenvalues_DFT

    def _load_ref_dos(self) -> tuple[np.ndarray, np.ndarray]:
        ref_data = np.load(self.inputfile["ref_dos_file"], allow_pickle=True).item()
        e_grid = ref_data["e"]
        if self.matrix_source == "OpenMX":
            dos_DFT = ref_data.get("dos_ref", ref_data["dos"])
        elif self.matrix_source == "Hot-Ham":
            dos_DFT = ref_data["dos"]
        return e_grid, dos_DFT

    def _diagonalize_hamiltonian_file(self, h5_path: str) -> np.ndarray:
        eigenvalues_all = []
        eigenvectors_all = []
        with h5py.File(h5_path, "r") as h5_file:
            k_keys = sorted(h5_file.keys(), key=lambda key: int(key[1:]))
            for k_key in tqdm(k_keys, desc=f"Diagonalizing {h5_path}"):
                k_group = h5_file[k_key]
                H_upper = sp.csc_matrix((k_group["data"][:], k_group["indices"][:], k_group["indptr"][:]), shape=tuple(k_group["shape"][:]))
                H_sparse = H_upper + H_upper.T.conj() - sp.diags(H_upper.diagonal())
                H = torch.from_numpy(H_sparse.toarray()).to(dtype=self.torch_dtype_complex, device=self.device)
                eigenvalues, eigenvectors = torch.linalg.eigh(H)
                eigenvalues_all.append(eigenvalues.detach().cpu().numpy())
                eigenvectors_all.append(eigenvectors.detach().cpu().numpy())
        eigenvalues_all = np.array(eigenvalues_all)
        eigenvectors_all = np.array(eigenvectors_all)
        eigenvalues_all -= self.fermi_energy
        return eigenvalues_all, eigenvectors_all

    def _plot_band_comparison(self, eigenvalues_DFT: np.ndarray, eigenvalues_K: np.ndarray, eigenvalues_K_prime: np.ndarray, k_distance: np.ndarray, special_k: np.ndarray, title: str, output: str):
        label_map = {"G": r"$\Gamma_{\mathrm{M}}$", "M": r"$\mathrm{M}_{\mathrm{M}}$", "K": r"$\mathrm{K}_{\mathrm{M}}$"}
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        tick_labels = [label_map.get(c, c) for c in self.inputfile["matrix_band_path"]]
        for i, band in enumerate(eigenvalues_DFT.T):
            ax.plot(k_distance, band, color="green", lw=1, ls="-", alpha=0.6, label="DFT" if i == 0 else "")
        for i, band in enumerate(eigenvalues_K.T):
            ax.plot(k_distance, band, color="blue", lw=1.5, ls="--", alpha=0.6, label="K valley" if i == 0 else "")
        for i, band in enumerate(eigenvalues_K_prime.T):
            ax.plot(k_distance, band, color="red", lw=1.5, ls="--", alpha=0.6, label="K' valley" if i == 0 else "")
        for xv in special_k:
            ax.axvline(xv, color="black", lw=0.8, ls="-")
        ax.axhline(0, color="black", lw=0.8, ls="-")
        ax.set_xticks(special_k)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(k_distance[0], k_distance[-1])
        ax.set_ylim(-2, 1)
        ax.set_ylabel("Energy (eV)")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(output, dpi=150)

    def _plot_dos_comparison(self, e_grid: np.ndarray, dos_DFT: np.ndarray, dos_model: np.ndarray, label: str, title: str, output: str):
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(e_grid, dos_DFT, color="green", lw=1.0, ls="-", alpha=0.6, label="DFT")
        ax.plot(e_grid, dos_model, color="blue", lw=1.5, ls="--", alpha=0.6, label=label)
        ax.axvline(0, color="black", lw=0.8, ls="-")
        ax.set_xlim(-4.0, 4.0)
        ax.set_xlabel("Energy (eV)")
        ax.set_ylabel("DOS (arb. units)")
        ax.legend(loc="upper right", fontsize=8)
        fig.suptitle(title)
        plt.tight_layout()
        plt.savefig(output, dpi=150)

    def demo_TAPW_band(self):
        eigenvalues_DFT = self._load_ref_band()
        npoints = eigenvalues_DFT.shape[0]
        kpoints_frac, k_distance, special_k = seek_kpath(self.structure, self.inputfile["matrix_band_path"], self.inputfile["matrix_band_pbc"], npoints)
        self.run(kmode="band", kpoints_frac=kpoints_frac, project_target="K", Hamiltonian_target="H_TAPW")
        self.run(kmode="band", kpoints_frac=kpoints_frac, project_target="K'", Hamiltonian_target="H_TAPW")
        eigenvalues_K, _ = self._diagonalize_hamiltonian_file("H_TAPW_band_K.h5")
        eigenvalues_K_prime, _ = self._diagonalize_hamiltonian_file("H_TAPW_band_K'.h5")
        self._plot_band_comparison(eigenvalues_DFT, eigenvalues_K, eigenvalues_K_prime, k_distance, special_k, f"TAPW band compared with DFT ($E_F$ = {self.fermi_energy:.4f} eV)", "band_TAPW_with_DFT.png")

    def demo_TAPW_dos(self):
        e_grid, dos_DFT = self._load_ref_dos()
        kpoints_frac, kweight = get_dense_grid(self.structure, self.inputfile["matrix_dos_grid"], using_ir_mesh=False)
        self.run(kmode="dos", kpoints_frac=kpoints_frac, project_target="K", Hamiltonian_target="H_TAPW")
        self.run(kmode="dos", kpoints_frac=kpoints_frac, project_target="K'", Hamiltonian_target="H_TAPW")
        eigenvalues_K, _ = self._diagonalize_hamiltonian_file("H_TAPW_dos_K.h5")
        eigenvalues_K_prime, _ = self._diagonalize_hamiltonian_file("H_TAPW_dos_K'.h5")
        dos_TAPW = compute_dos(eigenvalues_K, kweight, e_grid, sigma=0.1) + compute_dos(eigenvalues_K_prime, kweight, e_grid, sigma=0.1)
        self._plot_dos_comparison(e_grid, dos_DFT, dos_TAPW, "TAPW", f"TAPW DOS compared with DFT ($E_F$ = {self.fermi_energy:.4f} eV)", "dos_TAPW_with_DFT.png")

    def demo_lowE_band(self):
        eigenvalues_DFT = self._load_ref_band()
        npoints = eigenvalues_DFT.shape[0]
        kpoints_frac, k_distance, special_k = seek_kpath(self.structure, self.inputfile["matrix_band_path"], self.inputfile["matrix_band_pbc"], npoints)
        self.run(kmode="band", kpoints_frac=kpoints_frac, project_target="K", Hamiltonian_target="H_lowE")
        self.run(kmode="band", kpoints_frac=kpoints_frac, project_target="K'", Hamiltonian_target="H_lowE")
        eigenvalues_K, _ = self._diagonalize_hamiltonian_file("H_lowE_band_K.h5")
        eigenvalues_K_prime, _ = self._diagonalize_hamiltonian_file("H_lowE_band_K'.h5")
        self._plot_band_comparison(eigenvalues_DFT, eigenvalues_K, eigenvalues_K_prime, k_distance, special_k, f"LowE band compared with DFT ($E_F$ = {self.fermi_energy:.4f} eV)", "band_LowE_with_DFT.png")

    def demo_lowE_dos(self):
        e_grid, dos_DFT = self._load_ref_dos()
        kpoints_frac, kweight = get_dense_grid(self.structure, self.inputfile["matrix_dos_grid"], using_ir_mesh=False)
        self.run(kmode="dos", kpoints_frac=kpoints_frac, project_target="K", Hamiltonian_target="H_lowE")
        self.run(kmode="dos", kpoints_frac=kpoints_frac, project_target="K'", Hamiltonian_target="H_lowE")
        eigenvalues_K, _ = self._diagonalize_hamiltonian_file("H_lowE_dos_K.h5")
        eigenvalues_K_prime, _ = self._diagonalize_hamiltonian_file("H_lowE_dos_K'.h5")
        dos_lowE = compute_dos(eigenvalues_K, kweight, e_grid, sigma=0.1) + compute_dos(eigenvalues_K_prime, kweight, e_grid, sigma=0.1)
        self._plot_dos_comparison(e_grid, dos_DFT, dos_lowE, "LowE", f"LowE DOS compared with DFT ($E_F$ = {self.fermi_energy:.4f} eV)", "dos_LowE_with_DFT.png")

    def single_projection(self, project_target, Hamiltonian_target):
        kpoints_frac, k_distance, special_k = seek_kpath(self.structure, self.inputfile["matrix_band_path"], self.inputfile["matrix_band_pbc"], 45)
        self.run(kmode="band", kpoints_frac=kpoints_frac, project_target=project_target, Hamiltonian_target=Hamiltonian_target)
        eigenvalues, _ = self._diagonalize_hamiltonian_file(f"{Hamiltonian_target}_band_{project_target}.h5")
        Hamiltonian_label = {"H_TAPW": "TAPW", "H_lowE": "LowE"}.get(Hamiltonian_target, Hamiltonian_target)

        label_map = {"G": r"$\Gamma_{\mathrm{M}}$", "M": r"$\mathrm{M}_{\mathrm{M}}$", "K": r"$\mathrm{K}_{\mathrm{M}}$"}
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        tick_labels = [label_map.get(c, c) for c in self.inputfile["matrix_band_path"]]
        for band in eigenvalues.T:
            ax.plot(k_distance, band, color="steelblue", lw=1)
        for xv in special_k:
            ax.axvline(xv, color="black", lw=0.8, ls="-")
        ax.axhline(0, color="black", lw=0.8, ls="-")
        ax.set_xticks(special_k)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(k_distance[0], k_distance[-1])
        ax.set_ylim(-2, 1)
        ax.set_ylabel("Energy (eV)")
        fig.suptitle(f"{Hamiltonian_label} {project_target} band ($E_F$ = {self.fermi_energy:.4f} eV)")
        plt.tight_layout()
        plt.savefig(f"band_{Hamiltonian_target}_{project_target}.png", dpi=150)

    def double_projection(self, Hamiltonian_target):
        kpoints_frac, k_distance, special_k = seek_kpath(self.structure, self.inputfile["matrix_band_path"], self.inputfile["matrix_band_pbc"], 45)
        for project_target in ["K", "K'"]:
            self.run(kmode="band", kpoints_frac=kpoints_frac, project_target=project_target, Hamiltonian_target=Hamiltonian_target)

        eigenvalues_K, _ = self._diagonalize_hamiltonian_file(f"{Hamiltonian_target}_band_K.h5")
        eigenvalues_K_prime, _ = self._diagonalize_hamiltonian_file(f"{Hamiltonian_target}_band_K'.h5")
        Hamiltonian_label = {"H_TAPW": "TAPW", "H_lowE": "LowE"}.get(Hamiltonian_target, Hamiltonian_target)

        label_map = {"G": r"$\Gamma_{\mathrm{M}}$", "M": r"$\mathrm{M}_{\mathrm{M}}$", "K": r"$\mathrm{K}_{\mathrm{M}}$"}
        fig, ax = plt.subplots(1, 1, figsize=(6, 8))
        tick_labels = [label_map.get(c, c) for c in self.inputfile["matrix_band_path"]]
        for i, band in enumerate(eigenvalues_K.T):
            ax.plot(k_distance, band, color="blue", lw=1.5, ls="--", alpha=0.6, label="K valley" if i == 0 else "")
        for i, band in enumerate(eigenvalues_K_prime.T):
            ax.plot(k_distance, band, color="red", lw=1.5, ls="--", alpha=0.6, label="K' valley" if i == 0 else "")
        for xv in special_k:
            ax.axvline(xv, color="black", lw=0.8, ls="-")
        ax.axhline(0, color="black", lw=0.8, ls="-")
        ax.set_xticks(special_k)
        ax.set_xticklabels(tick_labels)
        ax.set_xlim(k_distance[0], k_distance[-1])
        ax.set_ylim(-2, 1)
        ax.set_ylabel("Energy (eV)")
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc="upper right", fontsize=8)
        fig.suptitle(f"{Hamiltonian_label} K and K' band ($E_F$ = {self.fermi_energy:.4f} eV)")
        plt.tight_layout()
        plt.savefig(f"band_{Hamiltonian_target}_double_projection.png", dpi=150)


if __name__ == "__main__":
    inputfile = {
        #==========basic para==========#
        "device": "cuda",
        "precision": "float32", #float32 or float64
        "structure": "./model.xyz", #atoms are ordered first by layers then by unitcells, if not relaxed, use rigid.xyz
        "orbitals": {
            "B": ["1s","2s","2p","3p","3d"],
            "C": ["1s","2s","2p","3p","3d"],
            "N": ["1s","2s","2p","3p","3d"],
            "Mo": ["1s","2s","3s","2p","3p","3d","4d"],
            "Te": ["1s","2s","3s","2p","3p","3d","4d","4f"]
        },
        "fermi_energy": -3.8833428010611795,
        "matrix_source": "Hot-Ham", #"OpenMX" or "Hot-Ham"
        #==========only needed for band calculation==========#
        "matrix_band_dir": "./matrix_band",
        "matrix_band_path": "GKMG",
        "matrix_band_pbc": [1, 1, 1],
        #===========only needed for dos calculation==========#
        "matrix_dos_dir": "./matrix_dos",
        "matrix_dos_grid": [10, 10, 1],
        #===========only needed for DFT comparision==========#
        "ref_band_file": "./eigen.npy",
        "ref_dos_file": "./dos.npy",
        #==========valley_projection para==========#
        "project_harmonics": 2, #1 means the nearest 3 hexagons around K, 2 means 12 hexagons
         #==========orbital_projection para==========#
        "selected_states_index": [
            [3,4],
            [3,4],
        ], #choose oribitals in a Q block for projection, 1 line for a layer's choice.
    }
    
    with torch.no_grad():
        lowE = LowEnergyHamiltonian(inputfile)
        #demo
        #lowE.demo_TAPW_band()
        #lowE.demo_TAPW_dos()
        #lowE.demo_lowE_band()
        #lowE.demo_lowE_dos()
        
        #band projection
        #lowE.single_projection(project_target="K", Hamiltonian_target="H_TAPW")
        lowE.double_projection(Hamiltonian_target="H_TAPW")
        