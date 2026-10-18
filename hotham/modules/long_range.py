import torch
from typing import Union
from collections import defaultdict
from e3nn.o3 import Irreps
from .te3nn.to3 import TO3_Irreps


class LongRange(torch.nn.Module):
    def __init__(self, irreps_node: Union[Irreps, TO3_Irreps], para):
        super().__init__()
        self.irreps_node = irreps_node
        self.para = para

        self.N_dl = [1, 1, -1]

        idx_by_l = defaultdict(list)
        offset = 0
        for mul, (l, _, _) in self.irreps_node:
            start, end = offset, offset + mul * (2 * l + 1)
            idx_by_l[l].extend(list(range(start, end)))
            offset = end
        for l, idx in idx_by_l.items():
            self.register_buffer(f"idx_by_{l}", torch.tensor(idx_by_l[l], dtype=para.torch_int_dtype))
        self.l_list = list(sorted(idx_by_l.keys()))

        self.amplitude_1 = torch.tensor(
            [0.2750, 0.1375, 0.0688, 0.0344, 0.0172, 0.0086, 0.0043, 0.0021, 0.0011, 0.0005, 0.0003, 0.0001], dtype=para.torch_float_dtype
        )
        self.shift_1 = torch.tensor([2.8, 5.7, 11.4, 22.7, 45.5, 91.0, 182.0, 364.0, 728.0, 1456.0, 2912.0, 5823.9], dtype=para.torch_float_dtype)
        self.wl = torch.nn.Parameter(self.amplitude_1 * (torch.sqrt(torch.tensor(torch.pi, dtype=para.torch_float_dtype)) ** 3) * self.shift_1**3)
        self.sl = torch.nn.Parameter(-torch.log(2 / self.shift_1))

    def forward(self, x, data):
        device = x.device
        dtype = x.dtype
        n = x.shape[0]

        box = data.lattice.to(dtype)
        cell_inv = torch.linalg.inv(box)
        G = 2 * torch.pi * cell_inv.T

        norms = torch.norm(box, dim=1)
        Nk = [max(1, int(n.item() / self.N_dl[i])) for i, n in enumerate(norms)]
        n1 = torch.arange(-Nk[0], Nk[0] + 1, dtype=self.para.torch_int_dtype, device=device)
        n2 = torch.arange(-Nk[1], Nk[1] + 1, dtype=self.para.torch_int_dtype, device=device)
        n3 = torch.arange(-Nk[2], Nk[2] + 1, dtype=self.para.torch_int_dtype, device=device)

        nvec = torch.stack(torch.meshgrid(n1, n2, n3, indexing="ij"), dim=-1).reshape(-1, 3)
        nvec = nvec.to(G.dtype)
        kvec = nvec @ G

        # Apply k-space cutoff and filter
        k_sq = torch.sum(kvec**2, dim=1)
        mask = k_sq > 0
        kvec = kvec[mask]  # [M, 3]
        k_sq = k_sq[mask]  # [M]
        nvec = nvec[mask]  # [M, 3]

        non_zero = (nvec != 0).to(self.para.torch_int_dtype)
        first_non_zero = torch.argmax(non_zero, dim=1)
        sign = torch.gather(nvec, 1, first_non_zero.unsqueeze(1)).squeeze()
        hemisphere_mask = (sign > 0) | ((nvec == 0).all(dim=1))
        kvec = kvec[hemisphere_mask]
        k_sq = k_sq[hemisphere_mask]
        factors = torch.where((nvec[hemisphere_mask] == 0).all(dim=1), 1.0, 2.0).to(dtype)
        factors = factors / factors.sum()

        # Compute structure factor S(k), Σq*e^(ikr)
        k_dot_r = torch.matmul(data.pos, kvec.T)  # [n, M]
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)

        sigma2_at_kspace = -1 / torch.exp(-2 * self.sl)
        Gaussian_multiplier = (self.wl[None, :] * torch.exp(k_sq.unsqueeze(-1) * sigma2_at_kspace)).sum(dim=1) * factors

        out = torch.zeros_like(x)
        for l in self.l_list:
            idx_by_l = getattr(self, f"idx_by_{l}")
            # sub_x = x[:, idx_by_l].reshape(n, -1, 2 * l + 1)
            sub_x = x[:, idx_by_l]

            sub_x_at_kspace_by_cos = torch.einsum("nm,nk->km", sub_x, cos_k_dot_r)
            sub_x_at_kspace_by_sin = torch.einsum("nm,nk->km", sub_x, sin_k_dot_r)

            sub_x_at_kspace_by_cos = sub_x_at_kspace_by_cos * Gaussian_multiplier[:, None]
            sub_x_at_kspace_by_sin = sub_x_at_kspace_by_sin * Gaussian_multiplier[:, None]

            sub_x_by_cos = torch.einsum("km,nk->nm", sub_x_at_kspace_by_cos, cos_k_dot_r)
            sub_x_by_sin = torch.einsum("km,nk->nm", sub_x_at_kspace_by_sin, sin_k_dot_r)

            # out[:, idx_by_l] = (sub_x_by_cos + sub_x_by_sin).reshape(n, -1)
            out[:, idx_by_l] = sub_x_by_cos + sub_x_by_sin
            pass

        return out
        # if q.dim() == 1:
        #     q = q.unsqueeze(1)
        # # for torchscript compatibility, to avoid dtype mismatch, only use real part
        # cos_k_dot_r = torch.cos(k_dot_r)
        # sin_k_dot_r = torch.sin(k_dot_r)
        # S_k_real = (q.unsqueeze(2) * cos_k_dot_r.unsqueeze(1)).sum(dim=0)
        # S_k_imag = (q.unsqueeze(2) * sin_k_dot_r.unsqueeze(1)).sum(dim=0)
        # S_k_sq = S_k_real**2 + S_k_imag**2  # [M]

        # min_term = -1 / torch.exp(-2 * self.sl)
        # min_term = min_term.view(1, 1, 1, -1)
        # kfac = self.wl.view(1, 1, 1, -1) * torch.exp(k_sq.unsqueeze(-1) * min_term)
        # # print("multiplier:",multiplier.shape,multiplier)c
        # kfac = kfac.sum(dim=-1)
