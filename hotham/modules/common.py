from e3nn import o3
from e3nn.util.jit import compile_mode
import torch
from torch_scatter import scatter
import re
from ase.atom import atomic_numbers
from .te3nn.to3 import TO3_Irreps, TO3_Linear, TO3_ElementwiseTensorProduct


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
            atomtype: list(map(lambda x: self.AMSymbol_to_AM[x], amsymbollist)) for atomtype, amsymbollist in self.AtomType_AMSymbolList.items()
        }
        self.AtomSymbol_to_AMList = {self.AtomType_to_AtomSymbol[atomtype]: amlist for atomtype, amlist in self.AtomType_AMList.items()}
        self.AtomSymbol_to_OrbitlSum = {self.AtomType_to_AtomSymbol[atomtype]: obsum for atomtype, obsum in enumerate(self.AtomType_OrbitalSum)}


class MCST(o3.Irreps):
    """MultiChannelSphericalTensor"""

    def __new__(
        cls,
        lmax,
        p_val,
        channel,
    ):
        return super().__new__(cls, [(channel, (l, p_val * (-1) ** l)) for l in range(lmax + 1)])


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
            elif (-1) ** irs.ir.l == irs.ir.p:
                dim_even_tensor += irs.dim
                irreps_gated += o3.Irreps([irs])
            elif (-1) ** irs.ir.l == -irs.ir.p:
                dim_odd_tensor += irs.dim
                irreps_gated += o3.Irreps([irs])

        self.even_tensor = slice(dim_even_scalar, dim_even_scalar + dim_even_tensor)
        if dim_odd_scalar == 0:
            self.odd_scalar = None
            self.odd_tensor = None
            assert dim_odd_tensor == 0
        else:
            self.odd_tensor = slice(
                dim_even_scalar + dim_even_tensor + dim_odd_scalar, dim_even_scalar + dim_even_tensor + dim_odd_scalar + dim_odd_tensor
            )
        self.irreps_gated = irreps_gated

        irreps_scalars = o3.Irreps(f"{dim_even_scalar}x0e")
        irreps_gates = o3.Irreps(f"{irreps_gated.num_irreps}x0e")
        self.gates_lin = o3.Linear(irreps_scalars, (irreps_scalars + irreps_gates).simplify())
        self.mul = o3.ElementwiseTensorProduct(irreps_gated, irreps_gates)
        self.mul_slices = [slice(0, dim_even_tensor)]
        if dim_odd_scalar != 0:
            self.mul_slices.append(slice(dim_even_tensor, dim_even_tensor + dim_odd_tensor))
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


@compile_mode("script")
class TO3Gate(torch.nn.Module):
    def __init__(self, irreps_in: TO3_Irreps):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = self.irreps_in

        dim_even_scalar = 0
        irreps_gated = TO3_Irreps([])
        mask_even_scalar = torch.zeros(self.irreps_in.dim, dtype=torch.bool)
        mask_pseudo_scalar = torch.zeros(self.irreps_in.dim, dtype=torch.bool)
        mask_tensor = torch.zeros(self.irreps_in.dim, dtype=torch.bool)
        for irs, s in zip(self.irreps_in, self.irreps_in.slices()):
            if irs.ir.is_scalar():
                mask_even_scalar[s] = True
                dim_even_scalar += irs.dim
            elif irs.ir.l == 0:
                mask_pseudo_scalar[s] = True
            else:
                mask_tensor[s] = True
                irreps_gated += TO3_Irreps([irs])
        if irreps_gated.num_irreps > 0 and dim_even_scalar == 0:
            raise ValueError("TO3Gate requires at least one 0ee scalar to gate non-scalar irreps.")
        self.has_even_scalar = bool(mask_even_scalar.any().item())
        self.has_pseudo_scalar = bool(mask_pseudo_scalar.any().item())
        self.has_tensor = bool(mask_tensor.any().item())
        self.register_buffer("mask_even_scalar", mask_even_scalar)
        self.register_buffer("mask_pseudo_scalar", mask_pseudo_scalar)
        self.register_buffer("mask_tensor", mask_tensor)

        if self.has_even_scalar:
            irreps_scalars = TO3_Irreps(f"{dim_even_scalar}x0ee")
            irreps_gates = TO3_Irreps(f"{irreps_gated.num_irreps}x0ee")
            self.gates_lin = TO3_Linear(irreps_scalars, (irreps_scalars + irreps_gates).simplify())
        else:
            self.gates_lin = None
        if self.has_tensor:
            self.mul = TO3_ElementwiseTensorProduct(irreps_gated, irreps_gates)
        else:
            self.mul = None

    def __repr__(self):
        return f"{self.__class__.__name__} ({self.irreps_in} -> {self.irreps_out})"

    def forward(self, x):
        out = x.new_zeros(x.shape[:-1] + (self.irreps_out.dim,))

        gates = None
        if self.has_even_scalar:
            even_scalars = x[..., self.mask_even_scalar]
            even_scalars_plus_gates = self.gates_lin(even_scalars)
            size_es = even_scalars.shape[-1]
            out[..., self.mask_even_scalar] = torch.nn.functional.silu(even_scalars_plus_gates[..., :size_es])
            gates = even_scalars_plus_gates[..., size_es:]

        if self.has_pseudo_scalar:
            pseudo_scalars = x[..., self.mask_pseudo_scalar]
            out[..., self.mask_pseudo_scalar] = torch.tanh(pseudo_scalars)

        if self.has_tensor:
            assert gates is not None, "Gates must be computed from even scalars before applying to tensors."
            tensors = x[..., self.mask_tensor]
            gated = self.mul(tensors, torch.sigmoid(gates))
            out[..., self.mask_tensor] = gated

        return out


class E3LayerNormal(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, eps=1e-05, elementwise_affine=True, floatdtype=torch.float32):
        super().__init__()
        self.irreps_in = irreps_in
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        if elementwise_affine:
            self.weight = torch.nn.Parameter(torch.empty(irreps_in.num_irreps, dtype=floatdtype))
            self.bias = torch.nn.Parameter(torch.empty(irreps_in.num_irreps, dtype=floatdtype))
            affine_slices = []
            s_start = 0
            for mul, _ in irreps_in:
                affine_slices.append(slice(s_start, s_start + mul))
                s_start += mul
            self.affine_slices = affine_slices
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

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
            f = f - mean[batch]
            if ir.is_scalar():
                norm = scatter(f.abs().pow(2), batch, dim=0, reduce="mean").mean(dim=(1, 2), keepdim=True)
                f = f / (norm.sqrt()[batch] + self.eps)

            if self.elementwise_affine:
                w = self.weight[affine_slice]
                if ir.is_scalar():
                    b = self.bias[affine_slice]
                    f = f * w[None, :, None] + b[None, :, None]
                else:
                    f = f * w[None, :, None]

            out.append(f.reshape(-1, mul * ir.dim))

        out = torch.cat(out, dim=-1)
        return out


class TO3LayerNormal(torch.nn.Module):
    def __init__(
        self,
        irreps_in: TO3_Irreps,
        eps=1e-05,
        floatdtype=torch.float32,
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.eps = eps

        self.weight = torch.nn.Parameter(torch.empty(irreps_in.num_irreps, dtype=floatdtype))
        self.bias = torch.nn.Parameter(torch.empty(irreps_in.num_irreps, dtype=floatdtype))
        affine_slices = []
        s_start = 0
        for mul, _ in irreps_in:
            affine_slices.append((s_start, s_start + mul))
            s_start += mul
        self.affine_slices = affine_slices

        self.reset_parameters()

    def reset_parameters(self) -> None:
        torch.nn.init.ones_(self.weight)
        torch.nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, batch: torch.Tensor = None):
        if batch is None:
            batch = torch.zeros(x.shape[0], dtype=torch.long, device=x.device)

        out = []
        for (mul, ir), ir_slice, affine_slice in zip(self.irreps_in, self.irreps_in.slices(), self.affine_slices):
            f = x[..., ir_slice].reshape(-1, mul, ir.dim)
            mean = scatter(f, batch, dim=0, reduce="mean").mean(dim=1, keepdim=True)
            f = f - mean[batch]
            # if ir.is_scalar():
            #     norm = scatter(f.abs().pow(2), batch, dim=0, reduce="mean").mean(dim=(1, 2), keepdim=True)
            #     f = f / (norm.sqrt()[batch] + self.eps)
            norm = scatter(f.abs().pow(2), batch, dim=0, reduce="mean").mean(dim=(1, 2), keepdim=True)
            f = f / (norm + self.eps).sqrt()[batch]
            # f = f / (norm.sqrt()[batch] + self.eps)
            # f = f / norm.sqrt()[batch]

            w = self.weight[affine_slice[0] : affine_slice[1]]
            if ir.is_scalar():
                b = self.bias[affine_slice[0] : affine_slice[1]]
                f = f * w[None, :, None] + b[None, :, None]
            else:
                f = f * w[None, :, None]

            out.append(f.reshape(-1, mul * ir.dim))

        out = torch.cat(out, dim=-1)
        return out


class MyScatter(torch.nn.Module):
    # def __init__(self, fix_average: bool = False, N_average=None):
    def __init__(self, N_average=None):
        super().__init__()
        # self.fix_average = fix_average
        self.register_buffer("N_average", torch.tensor(-1.0, dtype=N_average.dtype if isinstance(N_average, torch.Tensor) else torch.float32))
        # if fix_average:
        #     self.N_average = N_average
        self.N_average = N_average

    def forward(self, src: torch.Tensor, index: torch.Tensor, dim: int, dim_size: int):
        # if self.fix_average:
        if self.N_average > 0:
            out = scatter(src, index, dim=dim, dim_size=dim_size, reduce="sum") / self.N_average
        else:
            out = scatter(src, index, dim=dim, dim_size=dim_size, reduce="mean")
        return out


def is_valid_path(irs1: TO3_Irreps, irs2: TO3_Irreps, irs3: TO3_Irreps):
    ir1ir2_set = set()
    for _, ir1 in irs1:
        for _, ir2 in irs2:
            ir1ir2_set = ir1ir2_set | set([ir for ir in ir1 * ir2])
    ir3_set = set([ir for _, ir in irs3])
    return ir3_set.issubset(ir1ir2_set)


def Maximum_matching_irreps(irs1: TO3_Irreps, irs2: TO3_Irreps, irs3: TO3_Irreps):
    ir1ir2_set = set()
    for _, ir1 in irs1:
        for _, ir2 in irs2:
            ir1ir2_set = ir1ir2_set | set([ir for ir in ir1 * ir2])
    out_irs = TO3_Irreps(None)
    missing_irs = TO3_Irreps(None)
    for mul, ir in irs3:
        if ir in ir1ir2_set:
            out_irs = out_irs + TO3_Irreps([(mul, ir)])
        else:
            missing_irs = missing_irs + TO3_Irreps([(mul, ir)])
    return out_irs, missing_irs


def assert_cg_rank_check(ir1, ir3, intdtype=torch.int64):
    l1, p1 = ir1.l, ir1.p
    l3, p3 = ir3.l, ir3.p
    l_min = min([l1, l3])
    # idx = torch.tensor(list(range(1, l_min + 1)), dtype=torch.long)
    idx = torch.tensor(list(range(l_min + 1)), dtype=intdtype)
    p2 = p1 * p3
    cg_weights = []
    path_num = 0
    for l2 in range(abs(l1 - l3), l1 + l3 + 1):
        if (-1) ** l2 != p2:
            continue
        path_num += 1
        cg_l1l2l3 = o3.wigner_3j(l1, l2, l3, dtype=torch.float64)[:, l2, :]
        l_tot = l1 + l2 + l3
        if l_tot % 2 == 0:
            cg_diag = cg_l1l2l3[l1 + idx, l3 + idx]
        else:
            # beacuse cg_diag is always zero when m_o==m_i==0, we exclude this component
            assert len(idx) > 1
            cg_diag = cg_l1l2l3[l1 + idx[1:], l3 - idx[1:]]
        cg_weights.append(cg_diag)
    cg_weights = torch.stack(cg_weights, dim=1)
    rank = torch.linalg.matrix_rank(cg_weights).item()
    # assert path_num >= rank
    assert cg_weights.shape[0] == rank
