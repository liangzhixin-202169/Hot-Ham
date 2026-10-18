import e3nn
from e3nn import o3
import torch
from torch.nn import Linear, init
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data
import math
from typing import Optional, Union
from .common import MCST, Gate, TO3Gate, MyScatter, is_valid_path, Maximum_matching_irreps, assert_cg_rank_check
from .te3nn.to3 import TO3_Irreps, TO3_Linear, TO3_FullyConnectedTensorProduct, TO3_ElementwiseTensorProduct
from .output_linear import OutputLinear


class EdgeFeatureInitialize(torch.nn.Module):
    def __init__(
        self,
        irreps_node,
        irreps_edge,
        irreps_sh,
        mag_length_emb_to3irreps: TO3_Irreps,
        mag_sh_emb_to3irreps: TO3_Irreps,
        local_mag_env=None,
        para=None,
        **kwargs,
    ):
        super().__init__()
        if isinstance(irreps_node, e3nn.o3.Irreps):
            irreps_node = TO3_Irreps([(mul, (l, p, 1)) for mul, (l, p) in irreps_node])
        if isinstance(irreps_edge, e3nn.o3.Irreps):
            irreps_edge = TO3_Irreps([(mul, (l, p, 1)) for mul, (l, p) in irreps_edge])
        if isinstance(irreps_sh, e3nn.o3.Irreps):
            irreps_sh = TO3_Irreps([(mul, (l, p, 1)) for mul, (l, p) in irreps_sh])
        self.irreps_node = irreps_node
        self.irreps_edge = irreps_edge
        self.irreps_sh = irreps_sh
        self.mag_length_emb_to3irreps = mag_length_emb_to3irreps
        self.mag_sh_emb_to3irreps = mag_sh_emb_to3irreps
        self.local_mag_env = local_mag_env
        self.para = para
        channel = irreps_node[0][0]

        # init local mag env
        embedding_dim = self.para.basis_size
        mag_lenght_irreps = TO3_Irreps([(embedding_dim, (0, 1, 1))])
        # irreps_in = mag_lenght_irreps + mag_lenght_irreps + self.mag_sh_emb_to3irreps + self.mag_sh_emb_to3irreps + self.irreps_sh
        if self.local_mag_env is None:
            irreps_in = self.mag_sh_emb_to3irreps + self.mag_sh_emb_to3irreps + self.irreps_sh
        else:
            irreps_in = self.local_mag_env + self.local_mag_env + self.irreps_sh
        irreps_out = TO3_Irreps([(channel, (l, p, t_p)) for _, (l, p, t_p) in irreps_in.sort()[0].simplify()])
        self.lin1 = OutputLinear(irreps_in=irreps_in, irreps_out=irreps_out, shared_weights=False, intdtype=para.torch_int_dtype)
        # self.weights_for_lin1 = e3nn.nn.FullyConnectedNet([embedding_dim, 64, self.lin1.weight_numel], torch.nn.functional.silu)
        self.weights_for_lin1 = e3nn.nn.FullyConnectedNet([embedding_dim * 3, 64, self.lin1.weight_numel], torch.nn.functional.silu)
        self.scatter = MyScatter(self.para.N_average)

        # pairwise feature
        irreps_in = self.irreps_edge + self.irreps_node + self.irreps_node + self.lin1.irreps_out + self.lin1.irreps_out
        irreps_out = TO3_Irreps([(channel, (l, p, t_p)) for _, (l, p, t_p) in irreps_in.sort()[0].simplify()])
        self.lin2 = OutputLinear(irreps_in=irreps_in, irreps_out=irreps_out, shared_weights=True, intdtype=para.torch_int_dtype)

    def forward(self, f_node, f_edge, edge_sh, length_emb, mag_length_emb, mag_sh_emb, data: Data, local_mag_env=None):
        edge_src, edge_dst = data.edge_index_hop

        # gather local env info
        if self.local_mag_env is None:
            edge_info = torch.cat([mag_sh_emb[edge_src], mag_sh_emb[edge_dst], edge_sh], dim=-1)
        else:
            edge_info = torch.cat([local_mag_env[edge_src], local_mag_env[edge_dst], edge_sh], dim=-1)
        weights_for_lin1 = self.weights_for_lin1(torch.cat([mag_length_emb[edge_src], mag_length_emb[edge_dst], length_emb], dim=-1))
        edge_info = self.lin1(edge_info, weights_for_lin1)
        local_mag_env_by_scatter = self.scatter(edge_info, edge_dst, dim=0, dim_size=data.num_nodes)

        # initialize edge feature
        f_edge = self.lin2(
            torch.cat([f_edge, f_node[edge_src], f_node[edge_dst], local_mag_env_by_scatter[edge_src], local_mag_env_by_scatter[edge_dst]], dim=-1),
        )
        return f_edge


def lf_parity(p: int):
    if p == -1:
        return 1
    elif p == 1:
        return 0


class TO2_0_linear(torch.nn.Module):
    def __init__(self, irreps_in: TO3_Irreps, irreps_out: TO3_Irreps, embedding_dim: int = None, using_bias=True, shared_weights=True, intdtype=torch.int64, floatdtype=torch.float32):
        super().__init__()
        assert isinstance(irreps_in, TO3_Irreps)
        assert isinstance(irreps_out, TO3_Irreps)
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.using_bias = using_bias
        self.shared_weights = shared_weights

        self.c_in = irreps_in.num_irreps
        self.c_out = irreps_out.num_irreps

        self.lmax = max([self.irreps_in.lmax, self.irreps_out.lmax])
        self.irreps_sh = TO3_Irreps.spherical_harmonics(lmax=self.lmax * 2, p=-1, t_p=1)

        # define weight_mask
        weight_mask = torch.zeros((self.irreps_in.num_irreps, self.irreps_out.num_irreps), dtype=torch.bool)
        offset_row = 0
        for mul1, irs1 in self.irreps_in:
            offset_col = 0
            for mul2, irs2 in self.irreps_out:

                # check is tensor product path allowed
                is_path_allowed = is_valid_path(TO3_Irreps([(mul1, irs1)]), self.irreps_sh, TO3_Irreps([(mul2, irs2)]))

                # check is cg==0 when m_o==m_i==0 && m_f==0
                is_cg_zero = False
                l_tot = irs1.l + irs2.l + lf_parity(irs1.p * irs2.p)
                if l_tot % 2 == 1:
                    is_cg_zero = True

                # exclude improper path or path with zero cg weight
                if (not is_path_allowed) or is_cg_zero:
                    weight_mask[offset_row : offset_row + mul1, offset_col : offset_col + mul2] = True

                offset_col += mul2
            offset_row += mul1
        self.register_buffer("weight_mask", weight_mask)  # (in_dim, out_dim)

        # define weights
        if self.shared_weights:
            self.weight = torch.nn.Parameter(torch.empty(weight_mask.shape, dtype=floatdtype))
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = e3nn.nn.FullyConnectedNet([embedding_dim, 64, self.c_in * self.c_out], torch.nn.functional.silu)

        # define scalar_idx in output for bias
        if self.using_bias:
            scalar_idx = []
            offset = 0
            for mul2, irs2 in self.irreps_out:
                start, end = offset, offset + mul2
                if irs2.is_scalar():
                    scalar_idx.extend(list(range(start, end)))
                offset = end

            self.bias = torch.nn.Parameter(torch.empty(len(scalar_idx), dtype=floatdtype))
            init.uniform_(self.bias, -1.0 / math.sqrt(self.c_in), 1.0 / math.sqrt(self.c_in))
            self.register_buffer("bias_idx", torch.tensor(scalar_idx, dtype=intdtype))

        # define input's m0 component idx
        mask_tmp = torch.zeros((self.irreps_in.dim,), dtype=torch.bool)
        for (mul, (l, p, t_p)), s in zip(self.irreps_in, self.irreps_in.slices()):
            mask_tmp[s][torch.arange(mul, dtype=intdtype) * (2 * l + 1) + l] = True
        input_m0_idx = torch.arange(self.irreps_in.dim, dtype=intdtype)[mask_tmp]
        self.register_buffer("input_m_idx", input_m0_idx)

        # define output's m0 component idx
        mask_tmp = torch.zeros((self.irreps_out.dim,), dtype=torch.bool)
        for (mul, (l, p, t_p)), s in zip(self.irreps_out, self.irreps_out.slices()):
            mask_tmp[s][torch.arange(mul, dtype=intdtype) * (2 * l + 1) + l] = True
        output_m0_idx = torch.arange(self.irreps_out.dim, dtype=intdtype)[mask_tmp]
        self.register_buffer("output_m_idx", output_m0_idx)

    def forward(self, x, length_emb: Optional[torch.Tensor] = None):
        if self.shared_weights:
            weight = self.weight
            weight = weight.masked_fill(self.weight_mask, 0.0)
            out = torch.einsum("ji,zj->zi", weight, x)
            if self.using_bias:
                bias = self.bias
                out[:, self.bias_idx] = out[:, self.bias_idx] + bias

        else:
            weight = self.weight(length_emb).reshape(-1, self.c_in, self.c_out)
            weight = weight.masked_fill(self.weight_mask.unsqueeze(0), 0.0)
            out = torch.einsum("zji,zj->zi", weight, x)
            if self.using_bias:
                bias = self.bias
                out[:, self.bias_idx] = out[:, self.bias_idx] + bias

        return out


class TO2_m_linear(torch.nn.Module):
    def __init__(self, irreps_in: TO3_Irreps, irreps_out: TO3_Irreps, m: int, embedding_dim: int = None, shared_weights=True, intdtype=torch.int64, floatdtype=torch.float32):
        super().__init__()
        assert isinstance(irreps_in, TO3_Irreps)
        assert isinstance(irreps_out, TO3_Irreps)
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.target_m = m
        self.shared_weights = shared_weights

        self.c_in = irreps_in.num_irreps
        self.c_out = irreps_out.num_irreps

        self.lmax = max([self.irreps_in.lmax, self.irreps_out.lmax])
        self.irreps_sh = TO3_Irreps.spherical_harmonics(lmax=self.lmax * 2, p=-1, t_p=1)
        irreps_sh_even = TO3_Irreps([irs for irs in self.irreps_sh if irs.ir.p == 1])
        irreps_sh_odd = TO3_Irreps([irs for irs in self.irreps_sh if irs.ir.p == -1])

        # To determine cg's format
        # (1) calculate legal p_f by irs1.p and irs2.p
        # (2) determine allowed l_f' parity by p_f: p_f==(-1)**l_f
        # (3) judge is path irs1 * (l_f,p_f) -> irs2 allowed?
        # (4.1) if path is not allowed, cg weight is zero
        # (4.2) else if (l_o+l_i)+l_f == even, cg[:,0,:] is diagonal, y_mo = sigma_f(w_f*x_mo)
        # (4.3) else if (l_o+l_i)+l_f == odd, cg[:,0,:] is antidiagonal, y_mo = sigma_f(w_f*x_-mo)
        self.reduced_irreps_in = TO3_Irreps([irs for irs in self.irreps_in if irs.ir.l >= self.target_m])
        self.reduced_irreps_out = TO3_Irreps([irs for irs in self.irreps_out if irs.ir.l >= self.target_m])
        cg_idx = torch.zeros((self.reduced_irreps_in.num_irreps, self.reduced_irreps_out.num_irreps), dtype=intdtype)
        offset_row = 0
        for mul1, irs1 in self.reduced_irreps_in:
            offset_col = 0
            for mul2, irs2 in self.reduced_irreps_out:
                lf_p = lf_parity(irs1.p * irs2.p)
                if lf_p % 2 == 0:
                    passible_irreps_sh = irreps_sh_even
                else:
                    passible_irreps_sh = irreps_sh_odd
                is_path_allowed = is_valid_path(TO3_Irreps([(mul1, irs1)]), passible_irreps_sh, TO3_Irreps([(mul2, irs2)]))
                if is_path_allowed:
                    l_tot = irs1.l + irs2.l + lf_p
                    if l_tot % 2 == 0:
                        cg_idx[offset_row : offset_row + mul1, offset_col : offset_col + mul2] = 1
                    else:
                        cg_idx[offset_row : offset_row + mul1, offset_col : offset_col + mul2] = 2
                    assert_cg_rank_check(ir1=irs1, ir3=irs2, intdtype=intdtype)

                offset_col += mul2
            offset_row += mul1
        cg_zero = torch.zeros((2, 2), dtype=floatdtype)
        cg_even = torch.eye(n=2, dtype=floatdtype)
        cg_odd = torch.tensor([[0, 1], [-1, 0]], dtype=floatdtype)
        cgs = torch.stack([cg_zero, cg_even, cg_odd], dim=0)
        self.register_buffer("cg_idx", cg_idx)
        self.register_buffer("cgs", cgs)

        # define weights
        self.c_in = self.reduced_irreps_in.num_irreps
        self.c_out = self.reduced_irreps_out.num_irreps
        self.shared_weights = shared_weights
        if shared_weights:
            self.weight = torch.nn.Parameter(torch.empty((self.c_out, self.c_in), dtype=floatdtype))
            init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        else:
            self.weight = e3nn.nn.FullyConnectedNet([embedding_dim, 64, self.c_in * self.c_out], torch.nn.functional.silu)

        # define input's m and -m idx
        mask_tmp = torch.zeros((self.irreps_in.dim,), dtype=torch.bool)
        for (mul, (l, _, _)), s in zip(self.irreps_in, self.irreps_in.slices()):
            if l < self.target_m:
                continue
            mask_tmp[s][torch.arange(mul, dtype=intdtype) * (2 * l + 1) + l + self.target_m] = True
            mask_tmp[s][torch.arange(mul, dtype=intdtype) * (2 * l + 1) + l - self.target_m] = True
        input_m_idx = torch.arange(self.irreps_in.dim, dtype=intdtype)[mask_tmp]
        self.register_buffer("input_m_idx", input_m_idx)

        # define output's m and -m idx
        mask_tmp = torch.zeros((self.irreps_out.dim,), dtype=torch.bool)
        for (mul, (l, _, _)), s in zip(self.irreps_out, self.irreps_out.slices()):
            if l < self.target_m:
                continue
            mask_tmp[s][torch.arange(mul, dtype=intdtype) * (2 * l + 1) + l + self.target_m] = True
            mask_tmp[s][torch.arange(mul, dtype=intdtype) * (2 * l + 1) + l - self.target_m] = True
        output_m_idx = torch.arange(self.irreps_out.dim, dtype=intdtype)[mask_tmp]
        self.register_buffer("output_m_idx", output_m_idx)

    def forward(self, x: torch.Tensor, length_emb: Optional[torch.Tensor] = None):
        # x: (n,i,m_i)
        # weight: (i,j)
        # (n,i,m_i) * (i,j,m_i,m_o) -> (n,j,m_o)
        n, _, _ = x.shape

        if self.shared_weights:
            weight = self.weight
            out = torch.einsum("ijab,ji,zia->zjb", self.cgs[self.cg_idx], weight, x)

        return out


class TO2Linear(torch.nn.Module):
    def __init__(
        self, irreps_in: TO3_Irreps, irreps_out: TO3_Irreps, embedding_dim: int, shared_weights: bool, edge_include_sc, irreps_sh: TO3_Irreps = None, intdtype=torch.int64, floatdtype=torch.float32
    ):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_sh = irreps_sh
        self.irreps_out = irreps_out
        self.edge_include_sc = edge_include_sc
        self.intdtype = intdtype
        self.shared_weights = shared_weights
        self.m_max = min(self.irreps_in.lmax, self.irreps_out.lmax)

        # Create TO2 linear layers for each m value
        self.to2_m_fcs = torch.nn.ModuleList()
        for m in range(self.m_max + 1):
            if m == 0:
                to2_m_fc = TO2_0_linear(
                    irreps_in=self.irreps_in, irreps_out=self.irreps_out, embedding_dim=embedding_dim, shared_weights=shared_weights, intdtype=intdtype, floatdtype=floatdtype
                )
            else:
                to2_m_fc = TO2_m_linear(
                    irreps_in=self.irreps_in, irreps_out=self.irreps_out, m=m, embedding_dim=embedding_dim, shared_weights=shared_weights, intdtype=intdtype, floatdtype=floatdtype
                )
            self.to2_m_fcs.append(to2_m_fc)

        # Create slices for weights if shared_weights is True
        if shared_weights:
            weights_slices = [0]
            for m in range(self.m_max + 1):
                s_lenght = self.to2_m_fcs[m].c_out
                weights_slices.append(weights_slices[-1] + s_lenght)
            self.weights_slices = weights_slices
            self.weight_numel = self.weights_slices[-1]

        # Define idex for input' l component
        for l_target in range(self.irreps_in.lmax + 1):
            idx = []
            for (mul, (l, _, _)), s in zip(self.irreps_in, self.irreps_in.slices()):
                if l != l_target:
                    continue
                idx.extend(list(range(s.start, s.stop)))
            self.register_buffer(f"input_{l_target}_idx", torch.tensor(idx, dtype=intdtype))

        # Define idex for output' l component
        for l_target in range(self.irreps_out.lmax + 1):
            idx = []
            for (mul, (l, _, _)), s in zip(self.irreps_out, self.irreps_out.slices()):
                if l != l_target:
                    continue
                idx.extend(list(range(s.start, s.stop)))
            self.register_buffer(f"output_{l_target}_idx", torch.tensor(idx, dtype=intdtype))

    def forward(self, x, length_emb, wigner_D, edge_sh=None, mask_edge=None, mask_sc=None, weight=None):
        batch_size = x.shape[0]
        x_new = torch.zeros_like(x)

        # l==0 component is invariant under rotation
        input_0_idx = getattr(self, f"input_{0}_idx")
        x_new[..., input_0_idx] = x[..., input_0_idx]
        for l in range(1, self.irreps_in.lmax + 1):
            D_l = wigner_D[l]
            idx = getattr(self, f"input_{l}_idx")
            x_new[..., idx] = torch.einsum("zcj,zji->zci", x[..., idx].reshape(batch_size, -1, 2 * l + 1), D_l).reshape(batch_size, -1)

        out = x.new_zeros((batch_size, self.irreps_out.dim))
        if self.edge_include_sc and self.shared_weights:
            sub_batch_size = int(mask_edge.sum().item())
            index_0 = torch.arange(out.shape[0], dtype=self.intdtype, device=out.device).reshape(-1, 1)[mask_edge]
            # index_1 = torch.arange(out.shape[1], device=out.device).reshape(1, -1)
            _x_new = x_new[mask_edge]
            _weight = weight[mask_edge]
            for m in range(self.m_max + 1):
                n, _ = _x_new.shape
                _w = _weight[:, self.weights_slices[m] : self.weights_slices[m + 1]]
                to2_lin = self.to2_m_fcs[m]
                input_m_idx = to2_lin.input_m_idx
                output_m_idx = to2_lin.output_m_idx
                if m == 0:
                    _x = _x_new[:, input_m_idx]
                    _y = to2_lin(_x)
                    _y = _y * _w
                    out[index_0, output_m_idx] = _y
                elif m > 0:
                    _x = _x_new[:, input_m_idx].reshape(n, -1, 2)
                    _y = to2_lin(_x)
                    _y = _y * (_w.unsqueeze(-1))
                    out[index_0, output_m_idx] = _y.reshape(sub_batch_size, -1)
                # for m in range(self.m_max + 1):
                #     _w = _weight[:, self.weights_slices[m] : self.weights_slices[m + 1]]
                #     if m == 0:
                #         _x = _x_new[:, self.mask_m_in[m]]
                #         _y = self.to2_m_fcs[m](_x) * _w
                #         out[index_0, index_1[:, self.mask_m_out[m]]] = _y

                #     elif m > 0:
                #         _x = _x_new[:, self.mask_m_in[m]].reshape(sub_batch_size, self.to2_m_fcs[m].c_in, 2)
                #         _y = self.to2_m_fcs[m](_x) * (_w.unsqueeze(-1))
                #         out[index_0, index_1[:, self.mask_m_out[m]]] = _y.reshape(sub_batch_size, -1)

        elif self.edge_include_sc and (not self.shared_weights):
            raise NotImplementedError("Not implemented yet for edge_include_sc=True and shared_weights=False")
            # for m in range(self.m_max + 1):
            #     if m == 0:
            #         _x = x_new[..., self.mask_m_in[m]]
            #         out[index_0, index_1[:, self.mask_m_out[m]]] = self.to2_m_fcs[m](x=_x[mask_edge], length_emb=length_emb[mask_edge])

            #     elif m > 0:
            #         _x = x_new[..., self.mask_m_in[m]]
            #         out[index_0, index_1[:, self.mask_m_out[m]]] = self.to2_m_fcs[m](x=_x[mask_edge], length_emb=length_emb[mask_edge])

        elif (not self.edge_include_sc) and self.shared_weights:
            raise NotImplementedError("Not implemented yet for edge_include_sc=False and shared_weights=True")
            # for m in range(self.m_max + 1):
            #     w = weight[..., self.weights_slices[m] : self.weights_slices[m + 1]]
            #     if m == 0:
            #         _x = x_new[..., self.mask_m_in[m]]
            #         out[index_0, index_1[:, self.mask_m_out[m]]] = self.to2_m_fcs[m](_x) * w

            #     elif m > 0:
            #         _x = x_new[..., self.mask_m_in[m]]
            #         out[index_0, index_1[:, self.mask_m_out[m]]] = self.to2_m_fcs[m](_x) * w

        elif (not self.edge_include_sc) and (not self.shared_weights):
            raise NotImplementedError("Not implemented yet for edge_include_sc=False and shared_weights=False")
            # for m in range(self.m_max + 1):
            #     if m == 0:
            #         _x = x_new[..., self.mask_m_in[m]]
            #         out[index_0, index_1[:, self.mask_m_out[m]]] = self.to2_m_fcs[m](x=_x, length_emb=length_emb)

            #     elif m > 0:
            #         _x = x_new[..., self.mask_m_in[m]]
            #         out[index_0, index_1[:, self.mask_m_out[m]]] = self.to2_m_fcs[m](x=_x, length_emb=length_emb)

        # l==0 component is invariant under rotation
        for l in range(1, self.irreps_out.lmax + 1):
            D_l = wigner_D[l]
            idx = getattr(self, f"output_{l}_idx")
            out[..., idx] = torch.einsum("zcj,zij->zci", out[..., idx].reshape(batch_size, -1, 2 * l + 1), D_l).reshape(batch_size, -1)

        return out


class TO2_wrap(torch.nn.Module):
    def __init__(
        self,
        irreps_in: TO3_Irreps,
        irreps_out: TO3_Irreps,
        embedding_dim: int,
        shared_weights=True,
        edge_include_sc=True,
        irreps_sh: TO3_Irreps = None,
        intdtype=torch.int64,
        floatdtype=torch.float32,
    ):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.shared_weights = shared_weights
        self.edge_include_sc = edge_include_sc
        self.lmax = self.irreps_in.lmax
        self.irreps_sh = irreps_sh

        self.to2_lin = TO2Linear(
            irreps_in=irreps_in,
            irreps_out=irreps_out,
            embedding_dim=embedding_dim,
            shared_weights=shared_weights,
            edge_include_sc=self.edge_include_sc,
            intdtype=intdtype, floatdtype=floatdtype,
        )

        if self.edge_include_sc:
            self.sc_conv = TO3_FullyConnectedTensorProduct(self.irreps_in, "1x0ee", self.irreps_out, shared_weights=self.shared_weights)
            if self.shared_weights:
                self.weight_multi = TO3_ElementwiseTensorProduct(
                    self.sc_conv.to3_irreps_out, TO3_Irreps([(self.sc_conv.irreps_out.num_irreps, (0, 1, 1))])
                )
                self.fc = e3nn.nn.FullyConnectedNet([embedding_dim, 64, self.weight_multi.irreps_out.num_irreps], torch.nn.functional.silu)
            else:
                self.fc = e3nn.nn.FullyConnectedNet([embedding_dim, 64, self.sc_conv.weight_numel], torch.nn.functional.silu)

        if self.shared_weights:
            self.weights = e3nn.nn.FullyConnectedNet([embedding_dim, 64, self.to2_lin.weight_numel], torch.nn.functional.silu)

    def forward(
        self, x: torch.Tensor, edge_sh: torch.Tensor, length_emb: torch.Tensor, wigner_D: list, mask_edge: torch.Tensor, mask_sc: torch.Tensor
    ):
        weights = self.weights(length_emb)
        out = self.to2_lin(x=x, edge_sh=edge_sh, length_emb=length_emb, wigner_D=wigner_D, mask_edge=mask_edge, mask_sc=mask_sc, weight=weights)

        # sh_sc will be projected to 0e only: sh=Constant
        if self.edge_include_sc:
            if self.shared_weights:
                w = self.fc(length_emb[mask_sc])
                _x = self.sc_conv(x[mask_sc], x.new_tensor([[0.28209479177387814]]))
                out[mask_sc] += self.weight_multi(_x, w)
            else:
                w = self.fc(length_emb[mask_sc])
                out[mask_sc] += self.sc_conv(x[mask_sc], x.new_tensor([[0.28209479177387814]]), w)
        return out


class GeneralLocalConv(torch.nn.Module):
    def __init__(
        self,
        irreps_node: Union[o3.Irreps, TO3_Irreps],
        irreps_edge: Union[o3.Irreps, TO3_Irreps],
        irreps_sh: TO3_Irreps,
        irreps_out: TO3_Irreps,
        num_type: int,
        basis_size=12,
        hidden_neurons=64,
        local_mag_env=None,
        para=None,
        # mag_length_emb_to3irreps=None,
        # mag_sh_emb_to3irreps=None,
        **kwargs,
    ):
        super().__init__()

        # default to be time invariant
        if isinstance(irreps_node, o3.Irreps):
            irreps_node = TO3_Irreps([(mul, (l, p, 1)) for mul, (l, p) in irreps_node])
        if isinstance(irreps_edge, o3.Irreps):
            irreps_edge = TO3_Irreps([(mul, (l, p, 1)) for mul, (l, p) in irreps_edge])
        self.irreps_node = irreps_node
        self.irreps_edge = irreps_edge
        self.irreps_sh = irreps_sh
        self.local_mag_env = local_mag_env
        self.target_irreps_out = irreps_out
        self.irreps_out = irreps_out
        self.basis_size = basis_size
        self.nn_split_scale = para.nn_split_scale
        self.scatter = MyScatter(para.N_average)
        self.missing_irreps = TO3_Irreps(None)

        assert "mag_length_emb_to3irreps" in kwargs and "mag_sh_emb_to3irreps" in kwargs
        self.mag_length_emb_to3irreps = kwargs["mag_length_emb_to3irreps"]
        self.mag_sh_emb_to3irreps = kwargs["mag_sh_emb_to3irreps"]

        # lin1(f_node)|            |scatter->lin2_node
        #             |->tp->gate->|
        # lin1(f_edge)|            |->lin2_edge
        self.EdgeFeatureInitialize = EdgeFeatureInitialize(
            irreps_node=self.irreps_node,
            irreps_edge=self.irreps_edge,
            irreps_sh=self.irreps_sh,
            mag_length_emb_to3irreps=self.mag_length_emb_to3irreps,
            mag_sh_emb_to3irreps=self.mag_sh_emb_to3irreps,
            local_mag_env=self.local_mag_env,
            para=para,
        )
        irreps_in = self.EdgeFeatureInitialize.lin2.irreps_out
        self.local_irreps_out = self.irreps_out
        if not is_valid_path(irreps_in, self.irreps_sh, self.irreps_out):
            self.local_irreps_out, self.missing_irreps = Maximum_matching_irreps(irreps_in, self.irreps_sh, self.irreps_out)

        self.to2tp = TO2_wrap(
            irreps_in=irreps_in,
            irreps_out=self.local_irreps_out,
            embedding_dim=self.basis_size,
            shared_weights=True,
            edge_include_sc=para.edge_include_sc,
            irreps_sh=self.irreps_sh,
            intdtype=para.torch_int_dtype, floatdtype=para.torch_float_dtype,
        )
        self.gate = TO3Gate(irreps_in=self.to2tp.irreps_out)

        self.lin2_node = TO3_Linear(irreps_in=self.to2tp.irreps_out, irreps_out=self.local_irreps_out)
        self.lin2_edge = TO3_Linear(irreps_in=self.to2tp.irreps_out, irreps_out=self.local_irreps_out)

        # self_connection
        # self.sc_node = TO3_FullyConnectedTensorProduct(self.irreps_node, TO3_Irreps(f"{num_type}x0ee"), self.local_irreps_out)
        # self.sc_edge = TO3_FullyConnectedTensorProduct(self.irreps_node, TO3_Irreps(f"{2*num_type+basis_size}x0ee"), self.local_irreps_out)
        self.sc_node = TO3_Linear(self.irreps_node, self.local_irreps_out)
        self.sc_edge = TO3_Linear(self.irreps_edge, self.local_irreps_out)

        assert self.sc_node.irreps_out == self.lin2_node.irreps_out

    def forward(self, f_node, f_edge, edge_sh, node_emb, length_emb, data: Data, mag_length_emb=None, mag_sh_emb=None, local_mag_env=None):
        edge_src, edge_dst = data.edge_index_hop

        # self-connection
        # sc_node = self.sc_node(f_node, node_emb)
        # sc_edge = self.sc_edge(f_edge, torch.cat([node_emb[edge_src], node_emb[edge_dst], length_emb], dim=-1))
        sc_node = self.sc_node(f_node)
        sc_edge = self.sc_edge(f_edge)

        # lin1(f_node)|               |scatter->lin2_node
        #             |->so2tp->gate->|
        # lin1(f_edge)|               |->lin2_edge
        f_cat = self.EdgeFeatureInitialize(
            f_node=f_node,
            f_edge=f_edge,
            edge_sh=edge_sh,
            length_emb=length_emb,
            mag_length_emb=mag_length_emb,
            mag_sh_emb=mag_sh_emb,
            local_mag_env=local_mag_env,
            data=data,
        )
        if self.nn_split_scale > 1:
            sub_batch_size = math.ceil(data.num_edges / self.nn_split_scale)
            f_edge = [[] for _ in range(self.nn_split_scale)]
            for index_sub_batch in range(self.nn_split_scale):
                if index_sub_batch == self.nn_split_scale - 1:
                    start, end = (sub_batch_size * index_sub_batch, data.num_edges)
                else:
                    start, end = (sub_batch_size * index_sub_batch, sub_batch_size * (index_sub_batch + 1))
                sub_f_edge = self.to2tp(
                    x=f_cat[start:end],
                    edge_sh=edge_sh[start:end],
                    length_emb=length_emb[start:end],
                    wigner_D=[D[start:end] for D in data.wigner_D],
                    mask_edge=data.mask_edge[start:end],
                    mask_sc=data.mask_sc[start:end],
                )
                f_edge[index_sub_batch] = sub_f_edge
            f_edge = torch.cat(f_edge, dim=0)
        else:
            f_edge = self.to2tp(
                x=f_cat, edge_sh=edge_sh, length_emb=length_emb, wigner_D=data.wigner_D, mask_edge=data.mask_edge, mask_sc=data.mask_sc
            )

        f_edge = self.gate(f_edge)
        f_node = self.scatter(f_edge, edge_dst, dim=0, dim_size=data.num_nodes)
        f_node = self.lin2_node(f_node)
        f_edge = self.lin2_edge(f_edge)

        # add self-connection
        f_node = f_node + sc_node
        f_edge = f_edge + sc_edge

        return f_node, f_edge


if __name__ == "__main__":
    pass
