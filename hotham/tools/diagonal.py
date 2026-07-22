import os
import time
import torch
import spglib
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase import Atoms
from ase.io import read
from typing import Literal
from scipy.special import erfc
from collections import Counter

RYDBERG_TO_EV = 13.605693122994

class Electron():
    def __init__(self, inputfile):
        self.para = inputfile
        if self.para["precision"] == "float32":
            self.numpy_int_dtype = np.int32
            self.numpy_float_dtype = np.float32
            self.numpy_complex_dtype = np.complex64
            self.torch_int_dtype = torch.int32
            self.torch_float_dtype = torch.float32
            self.torch_complex_dtype = torch.complex64
        elif self.para["precision"] == "float64":
            self.numpy_int_dtype = np.int64
            self.numpy_float_dtype = np.float64
            self.numpy_complex_dtype = np.complex128
            self.torch_int_dtype = torch.int64
            self.torch_float_dtype = torch.float64
            self.torch_complex_dtype = torch.complex128
        
        self.matrix_HR_pred = {}
        self.matrix_SR = {}
        if self.para["H_ref_csr"] is not None:
            self.matrix_HR_ref = {}

        time_start = time.time()
        self.device = self.para["device"]
        self.atoms = read(self.para["structure"])        
        self.read_csr_HR_and_SR()
        self.kpoints_band, self.k_distance_band, self.special_k_band = self.seek_kpath(self.atoms, self.para["band_kpoints_path"], self.para["band_kpoints_pbc"], self.para["band_kpoints_num"])
        self.kpoints_dos, self.kpoints_weight_dos = self.get_dense_grid(self.atoms, self.para["dos_kpoints_grid"], self.para["dos_kpoints_use_irreducible_k"], using_car=False)
        time_end = time.time()
        print(f"read HR, SR, k, took {time_end - time_start:.2f} seconds")

    def seek_kpath(self, frame: Atoms, path: str, pbc: list, npoints: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        cell = frame.cell
        bandpath = cell.bandpath(path=path, pbc=pbc, npoints=npoints)
        path = bandpath.path
        # fractional coordinates
        kpts = np.asarray(bandpath.kpts, dtype=self.numpy_float_dtype)
        special_points = bandpath.special_points
        special_points = np.asarray([special_points[special_point] for special_point in path], dtype=self.numpy_float_dtype)
        # cartesian coordinates
        kpts_car = kpts @ np.asarray(bandpath.icell, dtype=self.numpy_float_dtype)
        special_points_car = special_points @ np.asarray(bandpath.icell, dtype=self.numpy_float_dtype)
        # distance
        k_distance = np.cumsum(np.linalg.norm((kpts_car[1:] - kpts_car[:-1]), axis=-1), dtype=self.numpy_float_dtype)
        k_distance = np.insert(k_distance, 0, 0).astype(self.numpy_float_dtype, copy=False)
        special_k = np.cumsum(np.linalg.norm((special_points_car[1:] - special_points_car[:-1]), axis=-1), dtype=self.numpy_float_dtype)
        special_k = np.insert(special_k, 0, 0).astype(self.numpy_float_dtype, copy=False)
        return kpts, k_distance, special_k

    def get_dense_grid(self, frame: Atoms, mesh: list, using_ir_mesh: bool, using_car: bool) -> tuple[np.ndarray, np.ndarray]:
        cell = (frame.cell.array, frame.get_scaled_positions(), frame.numbers)
        grid_mapping_table, grid_address = spglib.get_ir_reciprocal_mesh(mesh, cell)
        knum = len(grid_mapping_table)
        if using_ir_mesh:
            ir_counter = Counter(grid_mapping_table)
            kweight = np.array(list(ir_counter.values()), dtype=self.numpy_float_dtype) / knum
            kpoints = grid_address[np.array(list(ir_counter.keys()))] / np.array(mesh, dtype=self.numpy_float_dtype)
        else:
            kweight = np.ones(shape=knum, dtype=self.numpy_float_dtype) / knum
            kpoints = grid_address / np.array(mesh, dtype=self.numpy_float_dtype)
        kpoints = kpoints.astype(self.numpy_float_dtype, copy=False)
        if using_car:
            kpoints = (kpoints @ np.asarray(frame.cell.reciprocal(), dtype=self.numpy_float_dtype)).astype(self.numpy_float_dtype, copy=False)
        return kpoints, kweight

    def read_csr_HR_and_SR(self):
        files = [("HR_pred", self.para["H_pred_csr"]), ("SR", self.para["S_csr"])]
        if self.para["H_ref_csr"] is not None:
            files.append(("HR_ref", self.para["H_ref_csr"]))
        for target, filepath in files:
            with open(filepath, "r") as f:
                next(f)
                dim = int(next(f).split(":")[-1].strip())
                n_cell = int(next(f).split(":")[-1].strip())
                cell_shifts = np.zeros((n_cell, 3), dtype=self.numpy_float_dtype)
                all_cell_idx = []
                all_row = []
                all_col = []
                all_value = []
                for icell in range(n_cell):
                    parts = next(f).strip().split()
                    Rx, Ry, Rz = map(int, parts[:3])
                    nnz = int(parts[3])
                    cell_shifts[icell] = [Rx, Ry, Rz]
                    if nnz > 0:
                        values = np.fromstring(next(f).strip(), sep=' ', dtype=self.numpy_float_dtype)
                        col_ind = np.fromstring(next(f).strip(), sep=' ', dtype=self.numpy_int_dtype)
                        row_ptr = np.fromstring(next(f).strip(), sep=' ', dtype=self.numpy_int_dtype)
                        counts = row_ptr[1:] - row_ptr[:-1]
                        rows = np.repeat(np.arange(dim, dtype=self.numpy_int_dtype), counts)
                        all_cell_idx.append(np.full(nnz, icell, dtype=self.numpy_int_dtype))
                        all_row.append(rows)
                        all_col.append(col_ind)
                        all_value.append(values)
            assert len(all_value) > 0, f"Found nothing in {target} CSR."
            cell_idx = np.concatenate(all_cell_idx)
            row = np.concatenate(all_row)
            col = np.concatenate(all_col)
            value = np.concatenate(all_value)
            if target in ["HR_pred", "HR_ref"]:
                value = (value * RYDBERG_TO_EV).astype(self.numpy_float_dtype, copy=False)
                # Rydberg to eV
            target_dict = self.matrix_HR_pred if target == "HR_pred" else self.matrix_HR_ref if target == "HR_ref" else self.matrix_SR
            target_dict['cell_shifts'] = cell_shifts
            target_dict['cell_idx'] = cell_idx
            target_dict['row'] = row
            target_dict['col'] = col
            target_dict['value'] = value
            target_dict['dim'] = dim
            target_dict['n_cell'] = n_cell

    def get_Hk_and_Sk(self, k: np.ndarray, target: Literal["pred", "ref"]) -> tuple[np.ndarray, np.ndarray]:
        if target == "pred":
            matrix_HR = self.matrix_HR_pred
        elif target == "ref":
            matrix_HR = self.matrix_HR_ref
        Hk = np.zeros((matrix_HR['dim'], matrix_HR['dim']), dtype=self.numpy_complex_dtype)
        Sk = np.zeros((self.matrix_SR['dim'], self.matrix_SR['dim']), dtype=self.numpy_complex_dtype)
        phase_H = np.exp(1j * 2 * np.pi * (matrix_HR['cell_shifts'] @ k)).astype(self.numpy_complex_dtype, copy=False)
        phase_S = np.exp(1j * 2 * np.pi * (self.matrix_SR['cell_shifts'] @ k)).astype(self.numpy_complex_dtype, copy=False)
        np.add.at(Hk, (matrix_HR['row'], matrix_HR['col']), matrix_HR['value'] * phase_H[matrix_HR['cell_idx']])
        np.add.at(Sk, (self.matrix_SR['row'], self.matrix_SR['col']), self.matrix_SR['value'] * phase_S[self.matrix_SR['cell_idx']])
        return Hk, Sk

    def get_fermi_energy(self, eigvals: np.ndarray, kweight: np.ndarray, valence_electrons: int) -> float:
        valence_electrons = self.numpy_float_dtype(valence_electrons)
        sigma = self.numpy_float_dtype(self.para["dos_sigma"])
        elw = self.numpy_float_dtype(eigvals.min() - 2 * sigma)
        eup = self.numpy_float_dtype(eigvals.max() + 2 * sigma)
        for _ in range(100):
            e_fermi = self.numpy_float_dtype(0.5 * (elw + eup))
            n_electron = (0.5 * 2.0 * kweight[:, None] * erfc((eigvals - e_fermi) / sigma)).sum()
            if abs(n_electron - valence_electrons) < 1.0e-10:
                return e_fermi
            if n_electron < valence_electrons:
                elw = e_fermi
            else:
                eup = e_fermi
        raise RuntimeError("Fermi energy did not converge within 100 iterations")

    def get_dos(self, eigvals: np.ndarray, kweight: np.ndarray, efermi: float, energy_range: np.ndarray) -> np.ndarray:
        sigma = self.numpy_float_dtype(self.para["dos_sigma"])
        if eigvals.ndim == 1:
            eigvals = eigvals[None, :]
        shifted_eigvals = eigvals - self.numpy_float_dtype(efermi)
        delta_e = energy_range[:, None, None] - shifted_eigvals[None, :, :]
        gaussian = np.exp(-0.5 * (delta_e / sigma) ** 2)
        prefactor = self.numpy_float_dtype(2.0 / (np.sqrt(2.0 * np.pi) * sigma))
        dos = prefactor * (gaussian * kweight[None, :, None]).sum(axis=(1, 2))
        return dos

    def run(self):
        # 稀疏化存储HR和SR，计算本征值时构造稠密Hk和Sk       
        # 先算dos和fermi，再算band
        eigvals_dos_pred = np.empty((len(self.kpoints_dos), self.matrix_HR_pred["dim"]), dtype=self.numpy_float_dtype)
        for ik, k in enumerate(tqdm(self.kpoints_dos, desc="Diagonalizing H(k)_pred for dos and fermi")):
            Hk, Sk= self.get_Hk_and_Sk(k, target="pred")
            Hk, Sk = torch.as_tensor(Hk, device=self.device, dtype=self.torch_complex_dtype), torch.as_tensor(Sk, device=self.device, dtype=self.torch_complex_dtype)
            L = torch.linalg.cholesky(Sk)
            X = torch.linalg.solve_triangular(L, Hk, upper=False)
            H_prime = torch.linalg.solve_triangular(L, X.transpose(-1, -2).conj(), upper=False).transpose(-1, -2).conj()
            eigvals_dos_pred[ik] = torch.linalg.eigvalsh(H_prime).cpu().numpy()
        num_electrons = sum(self.para["valence_electrons"][atom.symbol] for atom in self.atoms)
        efermi_pred = self.get_fermi_energy(eigvals_dos_pred, self.kpoints_weight_dos, num_electrons)
        dos_energy_range = np.linspace(self.para["dos_energy_range"][0], self.para["dos_energy_range"][1], self.para["dos_energy_range"][2], dtype=self.numpy_float_dtype)
        dos_pred = self.get_dos(eigvals_dos_pred, self.kpoints_weight_dos, efermi_pred, dos_energy_range)

        eigvals_band_pred = np.empty((len(self.kpoints_band), self.matrix_HR_pred["dim"]), dtype=self.numpy_float_dtype)
        for ik, k in enumerate(tqdm(self.kpoints_band, desc="Diagonalizing H(k)_pred for band")):
            Hk, Sk= self.get_Hk_and_Sk(k, target="pred")
            Hk, Sk = torch.as_tensor(Hk, device=self.device, dtype=self.torch_complex_dtype), torch.as_tensor(Sk, device=self.device, dtype=self.torch_complex_dtype)
            L = torch.linalg.cholesky(Sk)
            X = torch.linalg.solve_triangular(L, Hk, upper=False)
            H_prime = torch.linalg.solve_triangular(L, X.transpose(-1, -2).conj(), upper=False).transpose(-1, -2).conj()
            eigvals_band_pred[ik] = torch.linalg.eigvalsh(H_prime).cpu().numpy()
        eigvals_band_pred -= efermi_pred

        if self.para["H_ref_csr"] is not None:
            eigvals_dos_ref = np.empty((len(self.kpoints_dos), self.matrix_HR_ref["dim"]), dtype=self.numpy_float_dtype)
            for ik, k in enumerate(tqdm(self.kpoints_dos, desc="Diagonalizing H(k)_ref for dos and fermi")):
                Hk, Sk= self.get_Hk_and_Sk(k, target="ref")
                Hk, Sk = torch.as_tensor(Hk, device=self.device, dtype=self.torch_complex_dtype), torch.as_tensor(Sk, device=self.device, dtype=self.torch_complex_dtype)
                L = torch.linalg.cholesky(Sk)
                X = torch.linalg.solve_triangular(L, Hk, upper=False)
                H_prime = torch.linalg.solve_triangular(L, X.transpose(-1, -2).conj(), upper=False).transpose(-1, -2).conj()
                eigvals_dos_ref[ik] = torch.linalg.eigvalsh(H_prime).cpu().numpy()
            efermi_ref = self.get_fermi_energy(eigvals_dos_ref, self.kpoints_weight_dos, num_electrons)
            dos_ref = self.get_dos(eigvals_dos_ref, self.kpoints_weight_dos, efermi_ref, dos_energy_range)

            eigvals_band_ref = np.empty((len(self.kpoints_band), self.matrix_HR_ref["dim"]), dtype=self.numpy_float_dtype)
            for ik, k in enumerate(tqdm(self.kpoints_band, desc="Diagonalizing H(k)_ref for band")):
                Hk, Sk= self.get_Hk_and_Sk(k, target="ref")
                Hk, Sk = torch.as_tensor(Hk, device=self.device, dtype=self.torch_complex_dtype), torch.as_tensor(Sk, device=self.device, dtype=self.torch_complex_dtype)
                L = torch.linalg.cholesky(Sk)
                X = torch.linalg.solve_triangular(L, Hk, upper=False)
                H_prime = torch.linalg.solve_triangular(L, X.transpose(-1, -2).conj(), upper=False).transpose(-1, -2).conj()
                eigvals_band_ref[ik] = torch.linalg.eigvalsh(H_prime).cpu().numpy()
            eigvals_band_ref -= efermi_ref
    
        #plot dos and band
        efermi_fmt = ".6f" if self.numpy_float_dtype == np.float32 else ".12f"
        print(f"Fermi energy (pred): {efermi_pred:{efermi_fmt}} eV")
        if self.para["H_ref_csr"] is not None:
            print(f"Fermi energy (ref): {efermi_ref:{efermi_fmt}} eV")
            
        save_dos = {"dos_energy_range": dos_energy_range,"dos": dos_pred,"efermi": efermi_pred,}
        if self.para["H_ref_csr"] is not None:
            save_dos["dos_ref"] = dos_ref
            save_dos["efermi_ref"] = efermi_ref
        np.save("dos.npy", save_dos)

        plt.figure()
        plt.plot(dos_energy_range, dos_pred, c="b", label="pred DOS")
        if self.para["H_ref_csr"] is not None:
            plt.plot(dos_energy_range, dos_ref, c="r", linestyle="--", label="ref DOS")
        plt.axvline(0, color="grey", linestyle="--")
        plt.legend()
        plt.xlabel("Energy (eV)")
        plt.ylabel("DOS")
        plt.title("Density of States")
        plt.savefig("dos.png", dpi=200)
        plt.close()

        save_band = {"eigenvalues": eigvals_band_pred,"k_distance": self.k_distance_band,"path": self.para["band_kpoints_path"],"special_k": self.special_k_band,}
        if self.para["H_ref_csr"] is not None:
            save_band["eigenvalues_ref"] = eigvals_band_ref
        np.save("eigen.npy", save_band)

        plt.figure()
        band_labels = [r"$\Gamma$" if point == "G" else point for point in self.para["band_kpoints_path"]]
        plt.plot(self.k_distance_band, eigvals_band_pred, c="b", label="hotham")
        if self.para["H_ref_csr"] is not None:
            plt.plot(self.k_distance_band, eigvals_band_ref, c="r", linestyle="--", label="openmx")
        plt.xlim(self.k_distance_band.min(), self.k_distance_band.max())
        plt.axhline(y=0, color="grey", linestyle="--")
        plt.xticks(self.special_k_band, band_labels)
        for point in self.special_k_band:
            plt.axvline(x=point, color="grey", linestyle="--", linewidth=0.5)
        plt.ylim(-2, 2)
        plt.ylabel("Energy(eV)")
        plt.savefig("band.png")
        plt.close()

    def extract_band_matrix(self):
        outdir = "./matrix_band"
        os.makedirs(outdir, exist_ok=True)
        for ik, k in enumerate(tqdm(self.kpoints_band, desc="Exporting Mk for Band")):
            k_outdir = os.path.join(outdir, f"k{ik}")
            os.makedirs(k_outdir, exist_ok=True)

            Hk_pred, Sk = self.get_Hk_and_Sk(k, target="pred")
            Hk_pred = sp.triu(sp.csc_matrix(Hk_pred), format="csc")
            Sk = sp.triu(sp.csc_matrix(Sk), format="csc")
            np.savez_compressed(os.path.join(k_outdir, f"k{ik}_H.npz"), data=Hk_pred.data, indices=Hk_pred.indices, indptr=Hk_pred.indptr, shape=Hk_pred.shape)
            np.savez_compressed(os.path.join(k_outdir, f"k{ik}_S.npz"), data=Sk.data, indices=Sk.indices, indptr=Sk.indptr, shape=Sk.shape)
            if self.para["H_ref_csr"] is not None:
                Hk_ref, _ = self.get_Hk_and_Sk(k, target="ref")
                Hk_ref = sp.triu(sp.csc_matrix(Hk_ref), format="csc")
                np.savez_compressed(os.path.join(k_outdir, f"k{ik}_Href.npz"), data=Hk_ref.data, indices=Hk_ref.indices, indptr=Hk_ref.indptr, shape=Hk_ref.shape)

    def extract_dos_matrix(self):
        outdir = "./matrix_dos"
        os.makedirs(outdir, exist_ok=True)

        with open(os.path.join(outdir, "kinfo.txt"), "w") as f:
            f.write("# kx ky kz weight\n")
            for kp, w in zip(self.kpoints_dos, self.kpoints_weight_dos):
                f.write(f"{kp[0]:.8f} {kp[1]:.8f} {kp[2]:.8f} {w:.8f}\n")
            
        for ik, k in enumerate(tqdm(self.kpoints_dos, desc="Exporting Mk for DOS")):
            k_outdir = os.path.join(outdir, f"k{ik}")
            os.makedirs(k_outdir, exist_ok=True)

            Hk_pred, Sk = self.get_Hk_and_Sk(k, target="pred")
            Hk_pred = sp.triu(sp.csc_matrix(Hk_pred), format="csc")
            Sk = sp.triu(sp.csc_matrix(Sk), format="csc")
            np.savez_compressed(os.path.join(k_outdir, f"k{ik}_H.npz"), data=Hk_pred.data, indices=Hk_pred.indices, indptr=Hk_pred.indptr, shape=Hk_pred.shape)
            np.savez_compressed(os.path.join(k_outdir, f"k{ik}_S.npz"), data=Sk.data, indices=Sk.indices, indptr=Sk.indptr, shape=Sk.shape)
            if self.para["H_ref_csr"] is not None:
                Hk_ref, _ = self.get_Hk_and_Sk(k, target="ref")
                Hk_ref = sp.triu(sp.csc_matrix(Hk_ref), format="csc")
                np.savez_compressed(os.path.join(k_outdir, f"k{ik}_Href.npz"), data=Hk_ref.data, indices=Hk_ref.indices, indptr=Hk_ref.indptr, shape=Hk_ref.shape)

if __name__ == "__main__":
    inputfile = {
        "precision": "float64",
        "device": "cuda",
        "structure": "./model.xyz",
        "valence_electrons": {"H": 1, "B": 3, "C": 4, "N": 5, "O": 6, "Mg": 10, "P": 5, "S": 6, "Ti": 12, "Nb": 13, "Mo": 14},
        "H_pred_csr": "./h_pred.csr",
        "H_ref_csr": "./h_ref.csr",
        "S_csr": "./olp.csr",
        # if H_ref is None, will skip calculations for H_ref

        # para band
        "band_kpoints_num":90,
        "band_kpoints_path":"GKMG",
        "band_kpoints_pbc":[True, True, False],
        
        # para dos
        "dos_kpoints_grid":[10,10,1],
        "dos_kpoints_use_irreducible_k": True,
        "dos_sigma": 0.05,
        "dos_energy_range": [-2, 2, 400],
    }

    with torch.no_grad():
        elec = Electron(inputfile)
        # calculate fermi, dos, band
        elec.run()
        # extract upper tri matrix for band/dos
        #elec.extract_band_matrix()
        #elec.extract_dos_matrix()

