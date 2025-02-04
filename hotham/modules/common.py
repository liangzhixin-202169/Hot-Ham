from e3nn import o3
from e3nn.util.jit import compile_mode
import torch
from torch_scatter import scatter


class MCST(o3.Irreps):
    """MultiChannelSphericalTensor"""
    def __new__(
        cls,
        lmax,
        p_val,
        channel,
    ):
        return super().__new__(cls, [(channel, (l, p_val * (-1)**l)) for l in range(lmax + 1)])


@compile_mode("script")
class Gate(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps):
        """irreps_in: 
            even_scalar + even_tensor + odd_scalar + dim_odd_tensor
        """
        super().__init__()
        self.irreps_in = irreps_in

        dim_even_scalar, dim_odd_scalar = 0, 0
        dim_even_tensor, dim_odd_tensor = 0, 0
        irreps_gated = o3.Irreps([])
        for irs, s in zip(self.irreps_in, self.irreps_in.slices()):
            if irs.ir.is_scalar():
                self.even_scalar = s
                dim_even_scalar = irs.dim
            elif irs.ir.l == 0 and irs.ir.p == -1:
                self.odd_scalar = s
                dim_odd_scalar = irs.dim
            elif (-1)**irs.ir.l == irs.ir.p:
                dim_even_tensor += irs.dim
                irreps_gated += o3.Irreps([irs])
            elif (-1)**irs.ir.l == -irs.ir.p:
                dim_odd_tensor += irs.dim
                irreps_gated += o3.Irreps([irs])

        self.even_tensor = slice(dim_even_scalar, dim_even_scalar+dim_even_tensor)
        if dim_odd_scalar == 0:
            self.odd_scalar = None
            self.odd_tensor = None
            assert dim_odd_tensor == 0
        else:
            self.odd_tensor = slice(dim_even_scalar+dim_even_tensor+dim_odd_scalar, dim_even_scalar+dim_even_tensor+dim_odd_scalar+dim_odd_tensor)
        self.irreps_gated = irreps_gated

        irreps_scalars = o3.Irreps(f"{dim_even_scalar}x0e")
        irreps_gates = o3.Irreps(f"{irreps_gated.num_irreps}x0e")
        self.gates_lin = o3.Linear(irreps_scalars, (irreps_scalars+irreps_gates).simplify())
        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        self.mul_slices = [slice(0, dim_even_tensor)]
        if dim_odd_scalar != 0:
            self.mul_slices.append(slice(dim_even_tensor, dim_even_tensor+dim_odd_tensor))
        a = 1

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_in})"

    def forward(self, x):
        even_scalars = x[..., self.even_scalar]
        even_tensor = x[..., self.even_tensor]

        gates = self.gates_lin(even_scalars)
        size_es = even_scalars.shape[-1]
        even_scalars = torch.nn.functional.silu(gates[..., :size_es])

        if self.odd_tensor is not None:
            odd_tensor = x[..., self.odd_tensor]
            odd_scalars = x[..., self.odd_scalar]
            gated = self.mul(torch.cat([even_tensor, odd_tensor], dim=-1), torch.sigmoid(gates[..., size_es:]))
            odd_scalars = torch.tanh(odd_scalars)
            x = torch.cat([even_scalars, gated[..., self.mul_slices[0]], odd_scalars, gated[..., self.mul_slices[1]]], dim=-1)
        else:
            gated = self.mul(even_tensor, torch.sigmoid(gates[..., size_es:]))
            x = torch.cat([even_scalars, gated], dim=-1)
        return x


class E3LayerNormal(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, eps=1e-05, elementwise_affine=True):
        super().__init__()
        self.irreps_in = irreps_in
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(irreps_in.num_irreps))
            self.bias = torch.nn.Parameter(torch.empty(irreps_in.num_irreps))
            affine_slices = []
            s_start = 0
            for mul, _ in irreps_in:
                affine_slices.append(slice(s_start, s_start+mul))
                s_start += mul
            self.affine_slices = affine_slices
        else:
            self.register_parameter('weight', None)
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
            torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        if batch == None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        out = []
        for (mul, ir), ir_slice, affine_slice in zip(self.irreps_in, self.irreps_in.slices(), self.affine_slices):
            f = x[..., ir_slice].reshape(-1, mul, ir.dim)
            mean = scatter(f, batch, dim=0, reduce="mean").mean(dim=1, keepdim=True)
            f = f-mean[batch]
            if ir.is_scalar():
                norm = scatter(f.abs().pow(2), batch, dim=0, reduce="mean").mean(dim=(1, 2), keepdim=True)
                f = f/(norm.sqrt()[batch]+self.eps)

            if self.elementwise_affine:
                w = self.weight[affine_slice]
                if ir.is_scalar():
                    b = self.bias[affine_slice]
                    f = f*w[None, :, None]+b[None, :, None]
                else:
                    f = f*w[None, :, None]

            out.append(f.reshape(-1, mul * ir.dim))

        out = torch.cat(out, dim=-1)
        return out


class MyScatter(torch.nn.Module):
    def __init__(self, fix_average: bool = False, N_average=None):
        super().__init__()
        self.fix_average = fix_average
        self.register_buffer("N_average", torch.tensor(-1.0, dtype=torch.float32))
        if fix_average:
            self.N_average = N_average

    def forward(self, src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int):
        if self.fix_average:
            out = scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum")/self.N_average
        else:
            out = scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")
        return out
