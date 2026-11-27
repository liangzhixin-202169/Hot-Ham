import time
import torch
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from ase.io import read

E = 1.602176634e-19 # C
HBAR_EV = 6.582119569e-16 # eV·s
RYDBERG_TO_EV = 13.605693122994
BOHR_TO_ANGSTROM = 0.529177210903
EV_TO_JOULE = 1.602176634e-19

class Optical_Conductivity():
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
        self.device = self.para["device"]
        self.atoms = read(self.para["structure"])

        self.matrix_HR = {}
        self.matrix_SR = {}
        self.matrix_rR = {}
        self.kpoints = np.empty((0, 3), dtype=self.numpy_float_dtype)

    def get_k_points(self):
        #gamma-started full grid
        grid = self.para["grid"]
        axes = [np.arange(n, dtype=self.numpy_float_dtype) / n for n in grid]
        kx, ky, kz = np.meshgrid(*axes, indexing="ij")
        self.kpoints = np.stack([kx.ravel(), ky.ravel(), kz.ravel()], axis=1)

    def read_csr_HR_and_SR(self):
        for target, filepath in [("HR", self.para["HR_csr"]), ("SR", self.para["SR_csr"] )]:
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
            if target == "HR":
                value = (value * RYDBERG_TO_EV).astype(self.numpy_float_dtype, copy=False)
                # Rydberg to eV
            target_dict = self.matrix_HR if target == "HR" else self.matrix_SR
            target_dict['cell_shifts'] = cell_shifts
            target_dict['cell_idx'] = cell_idx
            target_dict['row'] = row
            target_dict['col'] = col
            target_dict['value'] = value
            target_dict['dim'] = dim
            target_dict['n_cell'] = n_cell

    def read_csr_rR(self):
        with open(self.para["rR_csr"], "r") as f:
            next(f)
            dim = int(next(f).split(":")[-1].strip())
            n_cell = int(next(f).split(":")[-1].strip())
            cell_shifts = np.zeros((n_cell, 3), dtype=self.numpy_float_dtype)
            all_cell_idx = [[], [], []]
            all_row = [[], [], []]
            all_col = [[], [], []]
            all_value = [[], [], []]
            for icell in range(n_cell):
                parts = next(f).strip().split()
                Rx, Ry, Rz = map(int, parts[:3])
                cell_shifts[icell] = [Rx, Ry, Rz]
                for idir in range(3):
                    nnz = int(next(f).strip())
                    if nnz > 0:
                        values = np.fromstring(next(f).strip(), sep=' ', dtype=self.numpy_float_dtype)
                        col_ind = np.fromstring(next(f).strip(), sep=' ', dtype=self.numpy_int_dtype)
                        row_ptr = np.fromstring(next(f).strip(), sep=' ', dtype=self.numpy_int_dtype)
                        counts = row_ptr[1:] - row_ptr[:-1]
                        rows = np.repeat(np.arange(dim, dtype=self.numpy_int_dtype), counts)
                        all_cell_idx[idir].append(np.full(nnz, icell, dtype=self.numpy_int_dtype))
                        all_row[idir].append(rows)
                        all_col[idir].append(col_ind)
                        all_value[idir].append(values)
        self.matrix_rR['cell_shifts'] = cell_shifts
        for idir, label in enumerate(['x', 'y', 'z']):
            assert len(all_value[idir]) > 0, f"Found nothing in rR {label} CSR."
            cell_idx = np.concatenate(all_cell_idx[idir])
            row = np.concatenate(all_row[idir])
            col = np.concatenate(all_col[idir])
            value = (np.concatenate(all_value[idir]) * BOHR_TO_ANGSTROM).astype(self.numpy_float_dtype, copy=False)
            # Bohr to Å
            self.matrix_rR[label] = {'cell_idx': cell_idx, 'row': row, 'col': col, 'value': value}
        self.matrix_rR['dim'] = dim
        self.matrix_rR['n_cell'] = n_cell

    def get_Hk_and_Sk(self, k):
        Hk = np.zeros((self.matrix_HR['dim'], self.matrix_HR['dim']), dtype=self.numpy_complex_dtype)
        Sk = np.zeros((self.matrix_SR['dim'], self.matrix_SR['dim']), dtype=self.numpy_complex_dtype)
        phase_H = np.exp(1j * 2 * np.pi * (self.matrix_HR['cell_shifts'] @ k)).astype(self.numpy_complex_dtype, copy=False)
        phase_S = np.exp(1j * 2 * np.pi * (self.matrix_SR['cell_shifts'] @ k)).astype(self.numpy_complex_dtype, copy=False)
        np.add.at(Hk, (self.matrix_HR['row'], self.matrix_HR['col']), self.matrix_HR['value'] * phase_H[self.matrix_HR['cell_idx']])
        np.add.at(Sk, (self.matrix_SR['row'], self.matrix_SR['col']), self.matrix_SR['value'] * phase_S[self.matrix_SR['cell_idx']])
        return Hk, Sk, phase_H, phase_S

    def get_rk_uppertri(self, k):
        phase = np.exp(1j * 2 * np.pi * (self.matrix_rR['cell_shifts'] @ k)).astype(self.numpy_complex_dtype, copy=False)
        rk = []
        for label in ['x', 'y', 'z']:
            data = self.matrix_rR[label]
            value = (data['value'] * phase[data['cell_idx']]).astype(self.numpy_complex_dtype, copy=False)
            rk.append(sp.triu(sp.coo_matrix((value, (data['row'], data['col'])), shape=(self.matrix_rR['dim'], self.matrix_rR['dim'])), format='csr'))
        return rk

    def run(self):
        # 稀疏化存储HR和SR，计算本征值时构造稠密Hk和Sk
        time_start = time.time()
        self.read_csr_HR_and_SR()
        self.read_csr_rR()
        self.get_k_points()
        time_end = time.time()
        print(f"read HR, SR, rR, k, took {time_end - time_start:.2f} seconds")
        # ====================================================================
        #  Greenwood-Kubo 光电导
        #                 2 i * e²*h_bar                (f_nk − f_mk)          v^α_nm(k) v^β_mn(k)
        #  sigma_α(ω) = - ————————————  ·   Σ_k  Σ   · —————————————  · ——————————————————————————————
        #                   Nk  V_cell           n≠m    ε_nk − ε_mk      ω(eV) + i*eta + ε_nk − ε_mk
        #  其中:
        #    f_nk = Fermi-Dirac [0,1] (T=0 only)
        # ====================================================================
        E_F  = torch.as_tensor(self.para["fermi_energy"], device=self.device, dtype=self.torch_float_dtype)
        eta  = torch.as_tensor(self.para["eta"], device=self.device, dtype=self.torch_float_dtype)
        om   = torch.as_tensor(self.para["omega"], device=self.device, dtype=self.torch_float_dtype)
        rec_cell = torch.as_tensor(self.atoms.cell.reciprocal().array, device=self.device, dtype=self.torch_complex_dtype)
        B_inv = torch.linalg.inv(rec_cell)
        omega_grid = torch.arange(om[0], om[1]+self.para["domega"], self.para["domega"], device=self.device, dtype=self.torch_float_dtype)
        omega_c = omega_grid.to(self.torch_complex_dtype)[:, None]
        sigma   = torch.zeros(3, 3, omega_grid.numel(), device=self.device, dtype=self.torch_complex_dtype)

        band_block_size = 64
        occ_block_size = 64
        unocc_block_size = 256
        pair_chunk_size = 10000

        # 每计算一个k点，累加一次sigma，防止显存爆炸
        for k in tqdm(self.kpoints, desc="Diagonalizing H(k)"):
            Hk, Sk, phase_H, phase_S = self.get_Hk_and_Sk(k)
            Hk = torch.as_tensor(Hk, device=self.device, dtype=self.torch_complex_dtype)
            Sk = torch.as_tensor(Sk, device=self.device, dtype=self.torch_complex_dtype)
            L = torch.linalg.cholesky(Sk)
            del Sk
            X = torch.linalg.solve_triangular(L, Hk, upper=False)
            del Hk
            H_prime = torch.linalg.solve_triangular(L, X.transpose(-1, -2).conj(), upper=False).transpose(-1, -2).conj()
            del X
            if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()
            eigvals, eigvecs = torch.linalg.eigh(H_prime)
            del H_prime
            C = torch.linalg.solve_triangular(L.conj().T, eigvecs, upper=True)
            del L, eigvecs
            # ====================================================================
            # 速度矩阵v_nm^α:
            # v_nm^α(k)=1/ħ * [ C_n† (∂H(k)/∂k_α) C_m - ε_n * C_n† (∂S(k)/∂k_α) C_m + i * (ε_n - ε_m) * C_n† r^α(k) C_m]
            # ∂H(k)/∂k_α = i * Σ_R R_α H(R) e^{ik·R}
            # ∂S(k)/∂k_α = i * Σ_R R_α S(R) e^{ik·R}
            # using k_frac, fix k later
            R_H = self.matrix_HR['cell_shifts']
            R_S = self.matrix_SR['cell_shifts']
            dH_value_x = (1j * self.matrix_HR['value'] * R_H[self.matrix_HR['cell_idx'], 0] * phase_H[self.matrix_HR['cell_idx']]).astype(self.numpy_complex_dtype, copy=False)
            dH_value_y = (1j * self.matrix_HR['value'] * R_H[self.matrix_HR['cell_idx'], 1] * phase_H[self.matrix_HR['cell_idx']]).astype(self.numpy_complex_dtype, copy=False)
            dH_value_z = (1j * self.matrix_HR['value'] * R_H[self.matrix_HR['cell_idx'], 2] * phase_H[self.matrix_HR['cell_idx']]).astype(self.numpy_complex_dtype, copy=False)
            dS_value_x = (1j * self.matrix_SR['value'] * R_S[self.matrix_SR['cell_idx'], 0] * phase_S[self.matrix_SR['cell_idx']]).astype(self.numpy_complex_dtype, copy=False)
            dS_value_y = (1j * self.matrix_SR['value'] * R_S[self.matrix_SR['cell_idx'], 1] * phase_S[self.matrix_SR['cell_idx']]).astype(self.numpy_complex_dtype, copy=False)
            dS_value_z = (1j * self.matrix_SR['value'] * R_S[self.matrix_SR['cell_idx'], 2] * phase_S[self.matrix_SR['cell_idx']]).astype(self.numpy_complex_dtype, copy=False)
            dH_dk_x = sp.coo_matrix((dH_value_x, (self.matrix_HR['row'], self.matrix_HR['col'])), shape=(self.matrix_HR['dim'], self.matrix_HR['dim']))
            dH_dk_y = sp.coo_matrix((dH_value_y, (self.matrix_HR['row'], self.matrix_HR['col'])), shape=(self.matrix_HR['dim'], self.matrix_HR['dim']))
            dH_dk_z = sp.coo_matrix((dH_value_z, (self.matrix_HR['row'], self.matrix_HR['col'])), shape=(self.matrix_HR['dim'], self.matrix_HR['dim']))
            dS_dk_x = sp.coo_matrix((dS_value_x, (self.matrix_SR['row'], self.matrix_SR['col'])), shape=(self.matrix_SR['dim'], self.matrix_SR['dim']))
            dS_dk_y = sp.coo_matrix((dS_value_y, (self.matrix_SR['row'], self.matrix_SR['col'])), shape=(self.matrix_SR['dim'], self.matrix_SR['dim']))
            dS_dk_z = sp.coo_matrix((dS_value_z, (self.matrix_SR['row'], self.matrix_SR['col'])), shape=(self.matrix_SR['dim'], self.matrix_SR['dim']))
            del dH_value_x, dH_value_y, dH_value_z, dS_value_x, dS_value_y, dS_value_z, phase_H, phase_S, R_H, R_S

            def scipy_to_torch_sparse(matrix):
                matrix = matrix.tocoo()
                indices = torch.as_tensor(
                    np.vstack((matrix.row, matrix.col)),
                    device=self.device,
                    dtype=torch.long,
                )
                values = torch.as_tensor(matrix.data, device=self.device, dtype=self.torch_complex_dtype)
                return torch.sparse_coo_tensor(indices, values, matrix.shape, device=self.device).coalesce()

            rec_cell_np = rec_cell.detach().cpu().numpy()
            B_inv_np = B_inv.detach().cpu().numpy()
            dS_cart_x = (B_inv_np[0, 0] * dS_dk_x + B_inv_np[0, 1] * dS_dk_y + B_inv_np[0, 2] * dS_dk_z).tocsr()
            dS_cart_y = (B_inv_np[1, 0] * dS_dk_x + B_inv_np[1, 1] * dS_dk_y + B_inv_np[1, 2] * dS_dk_z).tocsr()
            dS_cart_z = (B_inv_np[2, 0] * dS_dk_x + B_inv_np[2, 1] * dS_dk_y + B_inv_np[2, 2] * dS_dk_z).tocsr()
            rk_upper = self.get_rk_uppertri(k)
            rk_x = (sp.triu(rk_upper[0], k=1, format='csr') + sp.tril((rk_upper[0] + 1j * sp.triu(dS_cart_x, format='csr')).getH(), k=0, format='csr')).tocsr()
            rk_y = (sp.triu(rk_upper[1], k=1, format='csr') + sp.tril((rk_upper[1] + 1j * sp.triu(dS_cart_y, format='csr')).getH(), k=0, format='csr')).tocsr()
            rk_z = (sp.triu(rk_upper[2], k=1, format='csr') + sp.tril((rk_upper[2] + 1j * sp.triu(dS_cart_z, format='csr')).getH(), k=0, format='csr')).tocsr()
            rk_cart_x = (rec_cell_np[0, 0] * rk_x + rec_cell_np[0, 1] * rk_y + rec_cell_np[0, 2] * rk_z).tocsr()
            rk_cart_y = (rec_cell_np[1, 0] * rk_x + rec_cell_np[1, 1] * rk_y + rec_cell_np[1, 2] * rk_z).tocsr()
            rk_cart_z = (rec_cell_np[2, 0] * rk_x + rec_cell_np[2, 1] * rk_y + rec_cell_np[2, 2] * rk_z).tocsr()
            del dS_cart_x, dS_cart_y, dS_cart_z, rk_upper, rk_x, rk_y, rk_z, rec_cell_np, B_inv_np

            dH_ops = [scipy_to_torch_sparse(dH_dk_x), scipy_to_torch_sparse(dH_dk_y), scipy_to_torch_sparse(dH_dk_z)]
            dS_ops = [scipy_to_torch_sparse(dS_dk_x), scipy_to_torch_sparse(dS_dk_y), scipy_to_torch_sparse(dS_dk_z)]
            rk_cart_ops = [scipy_to_torch_sparse(rk_cart_x), scipy_to_torch_sparse(rk_cart_y), scipy_to_torch_sparse(rk_cart_z)]
            del dH_dk_x, dH_dk_y, dH_dk_z, dS_dk_x, dS_dk_y, dS_dk_z, rk_cart_x, rk_cart_y, rk_cart_z

            def velocity_block(left_idx, right_idx):
                # 只计算需要的 band block：left_idx × right_idx，不构造完整 v_k[3, dim, dim]
                C_left = C[:, left_idx]
                C_right = C[:, right_idx]
                e_left = eigvals[left_idx]
                e_right = eigvals[right_idx]
                eig_diff_block = e_left[:, None] - e_right[None, :]

                v_crystal = []
                for idir in range(3):
                    # v1 = C† (∂H/∂k) C
                    dH_C_right = torch.sparse.mm(dH_ops[idir], C_right)
                    v1_block = C_left.conj().T @ dH_C_right
                    del dH_C_right
                    # v2 = -ε_n C† (∂S/∂k) C
                    dS_C_right = torch.sparse.mm(dS_ops[idir], C_right)
                    v2_block = -e_left[:, None] * (C_left.conj().T @ dS_C_right)
                    del dS_C_right
                    # v3 = i(ε_n - ε_m) C† r C
                    rk_C_right = torch.sparse.mm(rk_cart_ops[idir], C_right)
                    v3_block = 1j * eig_diff_block * (C_left.conj().T @ rk_C_right)
                    del rk_C_right
                    v_crystal.append(v1_block + v2_block + v3_block)
                    del v1_block, v2_block, v3_block

                v_block = torch.stack([
                    B_inv[0, 0] * v_crystal[0] + B_inv[0, 1] * v_crystal[1] + B_inv[0, 2] * v_crystal[2],
                    B_inv[1, 0] * v_crystal[0] + B_inv[1, 1] * v_crystal[1] + B_inv[1, 2] * v_crystal[2],
                    B_inv[2, 0] * v_crystal[0] + B_inv[2, 1] * v_crystal[1] + B_inv[2, 2] * v_crystal[2],
                ], dim=0)
                del C_left, C_right, e_left, e_right, eig_diff_block, v_crystal
                return v_block

            # 现在的v_k如果除H_BAR，单位是Å/s，为了单精度数值稳定，把H_BAR放最后补除
            # ====================================================================
            # ====================================================================
            # intra_band, 金属项，暂时是近似
            if self.para["Drude_fix"]:
                gamma_D = self.para["Gamma_D"] if self.para["Gamma_D"] is not None else eta
                gamma_F = self.para["Gamma_F"] if self.para["Gamma_F"] is not None else eta
                all_idx = torch.arange(eigvals.numel(), device=self.device)
                v_diag = torch.empty((3, eigvals.numel()), device=self.device, dtype=self.torch_complex_dtype)
                for band_start in range(0, all_idx.numel(), band_block_size):
                    band_idx = all_idx[band_start:band_start + band_block_size]
                    v_band = velocity_block(band_idx, band_idx)
                    v_diag[:, band_idx] = torch.diagonal(v_band, dim1=-2, dim2=-1)
                    del band_idx, v_band
                del all_idx
                fermi_weight = (gamma_F / torch.pi) / ((eigvals - E_F)**2 + gamma_F**2)
                intra_k = (
                    fermi_weight[None, None, :]
                    * v_diag[:, None, :]
                    * v_diag[None, :, :]
                ).sum(dim=-1)
                drude_factor = 1.0 / (gamma_D - 1j * omega_grid.to(self.torch_complex_dtype))
                sigma += 1j * intra_k[:, :, None] * drude_factor[None, None, :]
                del v_diag, fermi_weight, intra_k, drude_factor
            # ====================================================================
            # PYATB构造interband的方法，只遍历 occ → unocc，且有窗口截断设计
            # ε_n − ε_m < 0 (n∈occ, m∈unocc)
            # 参考公式:
            # sigma += (-v^α_nm * v^β_mn) /  ((ε_n − ε_m) + ω + i*eta)      ← 共振在 ω = -Δ (>0, 吸收)
            #          -conj(v^α_nm * v^β_mn) /  (-(ε_n − ε_m) + ω + i*eta) *  ← 共振在 ω =  Δ (<0, 辐射)
            occ_mask = eigvals <= E_F
            occ_idx = torch.where(occ_mask)[0]
            unocc_idx = torch.where(~occ_mask)[0]
            if occ_idx.numel() > 0 and unocc_idx.numel() > 0:
                omega_num = omega_grid.numel()
                max_omega_num = int(omega_num * 1.5)
                max_omega_energy = self.para["omega"][0] + max_omega_num * self.para["domega"]
                for occ_start in range(0, occ_idx.numel(), occ_block_size):
                    occ_blk = occ_idx[occ_start:occ_start + occ_block_size]
                    for unocc_start in range(0, unocc_idx.numel(), unocc_block_size):
                        unocc_blk = unocc_idx[unocc_start:unocc_start + unocc_block_size]
                        delta_block = eigvals[occ_blk][:, None] - eigvals[unocc_blk][None, :]
                        keep = torch.abs(delta_block) <= max_omega_energy
                        if torch.any(keep):
                            v_ou = velocity_block(occ_blk, unocc_blk)
                            v_uo = velocity_block(unocc_blk, occ_blk)
                            n_local, m_local = torch.where(keep)
                            for pair_start in range(0, n_local.numel(), pair_chunk_size):
                                pair_end = min(pair_start + pair_chunk_size, n_local.numel())
                                n_pair = n_local[pair_start:pair_end]
                                m_pair = m_local[pair_start:pair_end]
                                delta = delta_block[n_pair, m_pair]
                                v_nm = v_ou[:, n_pair, m_pair]
                                v_mn = v_uo[:, m_pair, n_pair]
                                vv = v_nm[:, None, :] * v_mn[None, :, :]
                                delta_c = delta.to(self.torch_complex_dtype)[None, :]
                                denom1 = delta_c * (delta_c + omega_c + 1j * eta)
                                denom2 = delta_c * (-delta_c + omega_c + 1j * eta)
                                sigma += vv @ (1.0 / denom1).T
                                sigma += vv.conj() @ (1.0 / denom2).T
                                del n_pair, m_pair, delta, v_nm, v_mn, vv, delta_c, denom1, denom2
                            del v_ou, v_uo, n_local, m_local
                        del delta_block, keep, unocc_blk
                    del occ_blk
                del omega_num, max_omega_num, max_omega_energy
            del occ_mask, occ_idx, unocc_idx, C, eigvals, dH_ops, dS_ops, rk_cart_ops
            if torch.cuda.is_available() and torch.device(self.device).type == "cuda":
                torch.cuda.empty_cache()
        V_A3 = self.atoms.get_volume()
        sigma = - 2j * (E**2) *sigma /  V_A3 / self.kpoints.shape[0] / HBAR_EV
        # 公式中分子HBAR，需要补除计算速度时漏除两次HBAR，等价于分母一个HBAR。 
        # unit = C^2 / (Å*s*eV)
        sigma /= EV_TO_JOULE
        # unit = C^2 / (Å*s*J) = S/Å
        sigma *= 1e10
        # unit = S/m
        
        omega_plot = omega_grid.detach().cpu()
        sigma_plot = sigma.detach().cpu()
        _, axes = plt.subplots(3, 3, figsize=(14, 12))
        labels = ['xx', 'xy', 'xz', 'yx', 'yy', 'yz', 'zx', 'zy', 'zz']
        for idx, (ax, label) in enumerate(zip(axes.flat, labels)):
            i, j = idx // 3, idx % 3
            sigma_ij = sigma_plot[i, j]
            ax.plot(omega_plot, sigma_ij.real, 'b-', linewidth=1.0, label='Re')
            ax.plot(omega_plot, sigma_ij.imag, 'r-', linewidth=1.0, label='Im')
            ax.set_title(rf'$\sigma_{{{label}}}$', fontsize=12)
            ax.set_xlabel(r'$\hbar\omega$ (eV)', fontsize=9)
            ax.set_ylabel('S/m', fontsize=9)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        plt.suptitle('Optical Conductivity Tensor', fontsize=14)
        plt.tight_layout()
        plt.savefig('sigma_tensor.png', dpi=150)
        omega_np = omega_plot.numpy()
        sigma_np = sigma_plot.numpy()
        np.savez("sigma_vs_omega.npz", omega=omega_np, sigma=sigma_np)

if __name__ == "__main__":
    inputfile = {
        "precision": "float32",
        "device": "cuda",
        "structure": "./model.xyz",
        "HR_csr": "./data-HR-sparse_SPIN0.csr",
        "SR_csr": "./data-SR-sparse_SPIN0.csr",
        "rR_csr": "./data-rR-sparse.csr",

        "fermi_energy": 14.524641438576,
        "omega": [0,5],
        "domega": 0.01,
        "eta": 0.1,
        "grid": [2,2,24],
        #True for metal
        "Drude_fix": False,
        #Drude散射率，单位eV，~1/τ，不填默认为tau
        "Gamma_D": None,
        #Drude费米能筛选宽度，单位eV，默认tau
        "Gamma_F": None,
    }
    with torch.no_grad():
        oc = Optical_Conductivity(inputfile)
        oc.run()
