import torch
import math
from typing import Union
from collections import defaultdict
from e3nn.o3 import Irreps
from .te3nn.to3 import TO3_Irreps


class SpectralChannelGate(torch.nn.Module):
    """Mix equivalent channels and gate their phase/rotation-invariant power."""

    def __init__(self, indices, channels, l, para):
        super().__init__()
        self.register_buffer("indices", torch.tensor(indices, dtype=para.torch_int_dtype), persistent=False)
        self.channels = channels
        self.components = 2 * l + 1
        self.weight = torch.nn.Parameter(torch.eye(channels, dtype=para.torch_float_dtype))
        self.alpha = torch.nn.Parameter(torch.zeros(channels, dtype=para.torch_float_dtype))
        self.beta = torch.nn.Parameter(torch.zeros(channels, dtype=para.torch_float_dtype))

    def forward(self, real, imag):
        # The same real, bias-free matrix acts on both Fourier quadratures.
        shape = (real.shape[0], self.channels, self.components)
        real = torch.einsum("ab,kbm->kam", self.weight, real.reshape(shape))
        imag = torch.einsum("ab,kbm->kam", self.weight, imag.reshape(shape))
        power = (real.square() + imag.square()).mean(dim=-1)
        # Fixed q0=1 in the current feature convention; no new normalization.
        gate = 2 * torch.sigmoid(self.alpha * torch.log1p(power) + self.beta)
        return (real * gate[..., None]).flatten(1), (imag * gate[..., None]).flatten(1)


class LongRange(torch.nn.Module):
    def __init__(self, irreps_node: Union[Irreps, TO3_Irreps], para):
        super().__init__()
        self.irreps_node = irreps_node
        self.para = para

        self.k_cutoff = 2 * math.pi / para.rc

        idx_by_l = defaultdict(list)
        offset = 0
        for mul, (l, _, _) in self.irreps_node:
            start, end = offset, offset + mul * (2 * l + 1)
            idx_by_l[l].extend(list(range(start, end)))
            offset = end
        for l, idx in idx_by_l.items():
            self.register_buffer(f"idx_by_{l}", torch.tensor(idx_by_l[l], dtype=para.torch_int_dtype))
        self.l_list = list(sorted(idx_by_l.keys()))

        # Keep parity and time-reversal types separate, even at the same l.
        groups = {}
        offset = 0
        for mul, ir in self.irreps_node:
            key = (ir.l, ir.p, getattr(ir, "t_p", 1))
            width = mul * (2 * ir[0] + 1)
            if key not in groups:
                groups[key] = [[], 0]
            groups[key][0].extend(range(offset, offset + width))
            groups[key][1] += mul
            offset += width
        self.spectral_groups = torch.nn.ModuleList([
            SpectralChannelGate(indices, channels, ir[0], para)
            for ir, (indices, channels) in groups.items()
        ])

        # Log-uniform initialization including both endpoints; widths remain trainable.
        minimum, maximum, count = para.longrange_range
        self.shift_1 = torch.logspace(math.log10(minimum), math.log10(maximum),
                                     steps=count, dtype=para.torch_float_dtype)
        self.amplitude_1 = 0.2750 * minimum / self.shift_1
        self.wl = torch.nn.Parameter(self.amplitude_1 * (torch.sqrt(torch.tensor(torch.pi, dtype=para.torch_float_dtype)) ** 3) * self.shift_1**3)
        self.sl = torch.nn.Parameter(-torch.log(2 / self.shift_1))

    def forward(self, x, data):
        device = x.device
        dtype = x.dtype
        box = data.lattice.to(device=device, dtype=torch.float64)
        reciprocal = 2 * math.pi * torch.linalg.inv(box).T
        volume = torch.linalg.det(box).abs().to(dtype)
        # n_i = k dot a_i / (2*pi), also for nonorthogonal cells.
        bounds = torch.ceil(torch.linalg.vector_norm(box, dim=1) / self.para.rc).long().tolist()
        axes = [torch.arange(-limit, limit + 1, device=device) for limit in bounds]
        indices = torch.cartesian_prod(*axes)
        i, j, k = indices.unbind(dim=1)
        half = (i > 0) | ((i == 0) & ((j > 0) | ((j == 0) & (k > 0))))
        vectors = indices[half].to(box.dtype) @ reciprocal
        squared = vectors.square().sum(dim=1)
        # Include points on the cutoff sphere up to floating-point roundoff.
        inside = squared <= self.k_cutoff**2 * (1 + 32 * torch.finfo(box.dtype).eps)
        kvec = vectors[inside].to(dtype)
        k_sq = squared[inside].to(dtype)

        k_dot_r = data.pos.to(dtype) @ kvec.T
        cos_k_dot_r = torch.cos(k_dot_r)
        sin_k_dot_r = torch.sin(k_dot_r)

        sigma2_at_kspace = -1 / torch.exp(-2 * self.sl)
        Gaussian_multiplier = (self.wl[None, :] * torch.exp(k_sq.unsqueeze(-1) * sigma2_at_kspace)).sum(dim=1) / volume

        out = torch.zeros_like(x)
        for group in self.spectral_groups:
            sub_x = x[:, group.indices]
            real = torch.einsum("nm,nk->km", sub_x, cos_k_dot_r)
            imag = torch.einsum("nm,nk->km", sub_x, sin_k_dot_r)
            real = real * Gaussian_multiplier[:, None]
            imag = imag * Gaussian_multiplier[:, None]
            real, imag = group(real, imag)
            out[:, group.indices] = 2 * (  # Reconstruct both +k and -k.
                torch.einsum("km,nk->nm", real, cos_k_dot_r)
                + torch.einsum("km,nk->nm", imag, sin_k_dot_r)
            )

        return out
