import e3nn
from e3nn import o3
from e3nn.nn import FullyConnectedNet
import torch
from torch.nn import Linear, init
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.data import Data
import math
from typing import Optional
from .common import MCST, Gate, MyScatter
from .EdgeSelfInteraction import EdgeSelfInteraction


class SO2_0_linear(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int, embedding_dim: int = None, using_bias=True, shared_weights=True):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.shared_weights = shared_weights
        self.using_bias = using_bias
        if shared_weights:
            self.linear = Linear(c_in, c_out, bias=using_bias)
        else:
            self.weight = FullyConnectedNet([embedding_dim, 64, c_in*c_out], torch.nn.functional.silu)
            if self.using_bias:
                self.num_bias = c_in*c_out
                self.bias = torch.nn.Parameter(torch.empty(c_out))
                init.uniform_(self.bias, -1./math.sqrt(c_out), 1./math.sqrt(c_out))

    def forward(self, x, emb: Optional[torch.Tensor] = None):
        if self.shared_weights:
            out = self.linear(x)
        else:
            weight = self.weight(emb).reshape(-1, self.c_out, self.c_in)
            if not self.using_bias:
                out = torch.einsum("zij,zj->zi", weight, x)
            else:
                out = torch.einsum("zij,zj->zi", weight, x)+self.bias
        return out


class SO2_m_linear(torch.nn.Module):
    def __init__(self, c_in: int, c_out: int, embedding_dim: int, shared_weights=True):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.shared_weights = shared_weights
        if self.shared_weights:
            self.linear = Linear(c_in, 2*c_out, bias=False)
        else:
            self.weight = FullyConnectedNet([embedding_dim, 64, c_in*(2*c_out)], torch.nn.functional.silu)

    def forward(self, x: torch.Tensor, emb: Optional[torch.Tensor] = None):
        # _x's index: [z, [-mo, mo], c_in]
        if self.shared_weights:
            y = self.linear(x)
        else:
            weight = self.weight(emb).reshape(emb.shape[0], 2*self.c_out, self.c_in)
            y = torch.einsum("zij,zmj->zmi", weight, x)

        y_neg_mo = y[:, [0], :self.c_out]+y[:, [1], self.c_out:]
        y_pos_mo = y[:, [1], :self.c_out]-y[:, [0], self.c_out:]

        return torch.cat([y_neg_mo, y_pos_mo], dim=-2)


class SO2Linear(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_sh: o3.Irreps, irreps_out: o3.Irreps, embedding_dim: int, shared_weights: bool, edge_include_sc):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_sh = irreps_sh
        self.irreps_out = irreps_out
        self.edge_include_sc = edge_include_sc
        self.shared_weights = shared_weights
        self.m_max = min(self.irreps_in.lmax, self.irreps_out.lmax)

        mask_m_in = torch.zeros((self.irreps_in.lmax+1, self.irreps_in.dim), dtype=torch.bool)
        mask_m_out = torch.zeros((self.irreps_out.lmax+1, self.irreps_out.dim), dtype=torch.bool)

        for (mul, (l, _)), s in zip(self.irreps_in, self.irreps_in.slices()):
            for m in range(l+1):
                if m == 0:
                    mask_m_in[m, s][torch.arange(mul)*(2*l+1)+l] = True
                else:
                    mask_m_in[m, s][torch.arange(mul)*(2*l+1)+l+m] = True
                    mask_m_in[m, s][torch.arange(mul)*(2*l+1)+l-m] = True
        self.register_buffer("mask_m_in", mask_m_in)

        l_in = torch.tensor([l for mul, (l, p) in self.irreps_in]).unsqueeze(1)
        l_sh = torch.tensor([l for mul, (l, p) in self.irreps_sh]).unsqueeze(0)
        p_in = torch.tensor([p for mul, (l, p) in self.irreps_in]).unsqueeze(1)
        p_sh = torch.tensor([p for mul, (l, p) in self.irreps_sh]).unsqueeze(0)
        l_up = (l_in+l_sh)
        l_down = (l_in-l_sh).abs()

        for (mul, (l, p)), s in zip(self.irreps_out, self.irreps_out.slices()):
            if (((l_up >= l)*(l_down <= l))[((p_in*p_sh) == p)]).sum() == 0:
                continue
            for m in range(l+1):
                if m == 0:
                    mask_m_out[m, s][torch.arange(mul)*(2*l+1)+l] = True
                else:
                    mask_m_out[m, s][torch.arange(mul)*(2*l+1)+l+m] = True
                    mask_m_out[m, s][torch.arange(mul)*(2*l+1)+l-m] = True
        self.register_buffer("mask_m_out", mask_m_out)

        self.fc_m0 = SO2_0_linear(mask_m_in[0].sum().item(), mask_m_out[0].sum().item(), embedding_dim=embedding_dim, using_bias=True, shared_weights=shared_weights)
        self.so2_m_fc = torch.nn.ModuleList()
        for m in range(1, self.m_max+1):
            c_in = int(mask_m_in[m].sum().div(2).item())
            c_out = int(mask_m_out[m].sum().div(2).item())
            self.so2_m_fc.append(SO2_m_linear(c_in, c_out, embedding_dim=embedding_dim, shared_weights=shared_weights))

        if shared_weights:
            weights_slices = [slice(0, self.fc_m0.c_out)]
            for m in range(1, self.m_max+1):
                s_start = weights_slices[-1].stop
                s_lenght = self.so2_m_fc[m-1].c_out
                weights_slices.append(slice(s_start, s_start+s_lenght))
            self.weights_slices = weights_slices
            self.weight_numel = self.weights_slices[-1].stop

    def forward(self, x, sh, emb, wigner_D, mask_edge=None, mask_sc=None, weight=None):
        batch_size = x.shape[0]

        x_new = torch.zeros_like(x)
        for (mul, (l, _)), s in zip(self.irreps_in, self.irreps_in.slices()):
            D_l = wigner_D[l]
            x_new[..., s] = torch.einsum("zcj,zji->zci", x[..., s].reshape(x.shape[0], mul, 2*l+1), D_l).reshape(batch_size, mul*(2*l+1))

        # check if Y_lm==0 iff m==0
        # mask_sh = torch.zeros(sh.shape[-1], dtype=torch.bool)
        # irreps_sh_complete = o3.Irreps.spherical_harmonics(lmax=self.irreps_sh.lmax, p=-1)
        # for (mul, ir), s in zip(irreps_sh_complete, irreps_sh_complete.slices()):
        #     if ir in self.irreps_sh:
        #         mask_sh[s] = True
        # sh_new = sh[..., mask_sh]
        # for (mul, (l, _)), s in zip(self.irreps_sh, self.irreps_sh.slices()):
        #     wigner_D = data.wigner_D[l]
        #     sh_new[..., s] = torch.einsum("zcj,zji->zci", sh_new[..., s].reshape(x.shape[0], mul, 2*l+1), wigner_D).reshape(batch_size, mul*(2*l+1))

        out = x.new_zeros((batch_size, self.irreps_out.dim))
        if self.edge_include_sc and self.shared_weights:
            sub_batch_size = mask_edge.sum()
            index_0 = torch.arange(out.shape[0], device=out.device).reshape(-1, 1)[mask_edge]
            index_1 = torch.arange(out.shape[1], device=out.device).reshape(1, -1)
            _x_new = x_new[mask_edge]
            _weight = weight[mask_edge]
            for m in range(self.m_max+1):
                _w = _weight[:, self.weights_slices[m]]
                if m == 0:
                    out[index_0, index_1[:, self.mask_m_out[m]]] += self.fc_m0(_x_new[:, self.mask_m_in[m]])*_w

                elif m > 0:
                    _x = _x_new[:, self.mask_m_in[m]].reshape(sub_batch_size, self.so2_m_fc[m-1].c_in, 2).transpose(dim0=-1, dim1=-2)
                    out[index_0, index_1[:, self.mask_m_out[m]]] += (self.so2_m_fc[m-1](_x)*(_w.unsqueeze(1))).reshape(sub_batch_size, -1)

        elif self.edge_include_sc and (not self.shared_weights):
            for m in range(self.m_max+1):
                if m == 0:
                    _x = x_new[..., self.mask_m_in[m]]
                    out[mask_edge][..., self.mask_m_out[m]] += self.fc_m0(x=_x[mask_edge], emb=emb[mask_edge])

                elif m > 0:
                    _x = x_new[..., self.mask_m_in[m]]
                    out[mask_edge][..., self.mask_m_out[m]] += self.so2_m_fc[m-1](x=_x[mask_edge], emb=emb[mask_edge])

        elif (not self.edge_include_sc) and self.shared_weights:
            for m in range(self.m_max+1):
                w = weight[..., self.weights_slices[m]]
                if m == 0:
                    _x = x_new[..., self.mask_m_in[m]]
                    out[..., self.mask_m_out[m]] += self.fc_m0(_x)*w

                elif m > 0:
                    _x = x_new[..., self.mask_m_in[m]]
                    out[..., self.mask_m_out[m]] += self.o2_m_fc[m-1](_x)*w

        elif (not self.edge_include_sc) and (not self.shared_weights):
            for m in range(self.m_max+1):
                if m == 0:
                    _x = x_new[..., self.mask_m_in[m]]
                    out[..., self.mask_m_out[m]] += self.fc_m0(x=_x, emb=emb)

                elif m > 0:
                    _x = x_new[..., self.mask_m_in[m]]
                    out[..., self.mask_m_out[m]] += self.o2_m_fc[m-1](x=_x, emb=emb)

        for (mul, (l, _)), s in zip(self.irreps_out, self.irreps_out.slices()):
            D_l = wigner_D[l]
            out[..., s] = torch.einsum("zcj,zij->zci", out[..., s].reshape(out.shape[0], mul, 2*l+1), D_l).reshape(batch_size, mul*(2*l+1))

        return out


class SO2_wrap(torch.nn.Module):
    def __init__(self, irreps_in: o3.Irreps, irreps_sh: o3.Irreps, irreps_out: o3.Irreps, embedding_dim: int, shared_weights=True, edge_include_sc=True):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.shared_weights = shared_weights
        self.edge_include_sc = edge_include_sc
        self.lmax = self.irreps_in.lmax
        self.irreps_sh = o3.Irreps.spherical_harmonics(lmax=self.lmax)

        self.so2_lin = SO2Linear(irreps_in, irreps_sh, irreps_out, embedding_dim, shared_weights, self.edge_include_sc)

        if self.edge_include_sc:
            self.sc_conv = o3.FullyConnectedTensorProduct(self.irreps_in, "1x0e", self.irreps_out, shared_weights=self.shared_weights)
            if self.shared_weights:
                self.weight_multi = e3nn.o3.ElementwiseTensorProduct(self.sc_conv.irreps_out, o3.Irreps([(self.sc_conv.irreps_out.num_irreps, (0, 1))]))
                self.fc = e3nn.nn.FullyConnectedNet([embedding_dim, 64, self.weight_multi.irreps_out.num_irreps], torch.nn.functional.silu)
            else:
                self.fc = e3nn.nn.FullyConnectedNet([embedding_dim, 64, self.sc_conv.weight_numel], torch.nn.functional.silu)

        if self.shared_weights:
            self.weights = e3nn.nn.FullyConnectedNet([embedding_dim, 64, self.so2_lin.weight_numel], torch.nn.functional.silu)

    def forward(self, x: torch.Tensor, sh: torch.Tensor, length_emb: torch.Tensor, wigner_D: list, mask_edge: torch.Tensor, mask_sc: torch.Tensor):
        weights = self.weights(length_emb)
        out = self.so2_lin(x=x, sh=sh, emb=length_emb, wigner_D=wigner_D, mask_edge=mask_edge, mask_sc=mask_sc, weight=weights)

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


class SO2OutputLayer(torch.nn.Module):
    def __init__(self, irreps_node: o3.Irreps, irreps_out: o3.Irreps,  num_type: int, basis_size=12, hidden_neurons=64, para=None, is_final_layer=True, **kwargs):
        super().__init__()

        self.irreps_node = irreps_node
        self.irreps_out = irreps_out
        self.basis_size = basis_size
        self.split_stru = para.split_stru
        self.scatter = MyScatter(para.fix_average, para.N_average)
        self.irreps_sh = MCST(self.irreps_node.lmax, 1, 1)
        self.para = para
        self.is_final_layer = is_final_layer

        # self_connection
        self.sc_node = o3.FullyConnectedTensorProduct(self.irreps_node, o3.Irreps(f"{num_type}x0e"), self.irreps_out)
        self.sc_edge = o3.FullyConnectedTensorProduct(self.irreps_node, o3.Irreps(f"{2*num_type+basis_size}x0e"), self.irreps_out)

        # lin1(f_node)|               |scatter->lin2_node
        #             |->so2tp->gate->|
        # lin1(f_edge)|               |->lin2_edge
        self.lin1_node = o3.Linear(irreps_in=self.irreps_node, irreps_out=self.irreps_node)
        self.lin1_edge = o3.Linear(irreps_in=self.irreps_node, irreps_out=self.irreps_node)
        irreps_in = o3.Irreps([(mul*3, (l, p)) for mul, (l, p) in self.irreps_node])

        self.so2tp = SO2_wrap(irreps_in, self.irreps_sh, self.irreps_out, self.basis_size, shared_weights=True, edge_include_sc=para.edge_include_sc)

        if self.para.edgeselfinteraction:
            self.edgeselfinteraction = EdgeSelfInteraction(v_max=2,
                                                           irreps_in=self.irreps_node,
                                                           irreps_out=self.irreps_node,
                                                           num_type=num_type,
                                                           basis_size=self.basis_size,
                                                           split_stru=self.para.split_stru,
                                                           para=self.para)

        self.gate = Gate(irreps_in=self.so2tp.irreps_out)

        self.lin2_node = o3.Linear(irreps_in=self.so2tp.irreps_out, irreps_out=self.irreps_out)
        self.lin2_edge = o3.Linear(irreps_in=self.so2tp.irreps_out, irreps_out=self.irreps_out)

        assert self.sc_node.irreps_out == self.lin2_node.irreps_out

    def forward(self, f_node, f_edge, sh, node_emb, length_emb, data: Data):
        edge_src, edge_dst = data.edge_index_hop

        # self-connection
        sc_node = self.sc_node(f_node, node_emb)
        sc_edge = self.sc_edge(f_edge, torch.cat([node_emb[edge_src], node_emb[edge_dst], length_emb], dim=-1))

        # lin1(f_node)|               |scatter->lin2_node
        #             |->so2tp->gate->|
        # lin1(f_edge)|               |->lin2_edge
        f_node = self.lin1_node(f_node)
        f_edge = self.lin1_edge(f_edge)

        f_cat = torch.cat([torch.cat([f_node[:, s][edge_src], f_node[:, s][edge_dst], f_edge[:, s]], dim=-1) for s in self.irreps_node.slices()], dim=-1)
        if self.split_stru > 1:
            sub_batch_size = math.ceil(data.num_edges/self.split_stru)
            f_edge = [[] for _ in range(self.split_stru)]
            for index_sub_batch in range(self.split_stru):
                if index_sub_batch == self.split_stru-1:
                    batch_slice = slice(sub_batch_size*index_sub_batch, data.num_edges)
                else:
                    batch_slice = slice(sub_batch_size*index_sub_batch, sub_batch_size*(index_sub_batch+1))
                sub_f_edge = self.so2tp(x=f_cat[batch_slice],
                                        sh=sh[batch_slice],
                                        length_emb=length_emb[batch_slice],
                                        wigner_D=[D[batch_slice] for D in data.wigner_D],
                                        mask_edge=data.mask_edge[batch_slice],
                                        mask_sc=data.mask_sc[batch_slice])
                f_edge[index_sub_batch] = sub_f_edge
            f_edge = torch.cat(f_edge, dim=0)
        else:
            f_edge = self.so2tp(x=f_cat, sh=sh, length_emb=length_emb, wigner_D=data.wigner_D, mask_edge=data.mask_edge, mask_sc=data.mask_sc)

        if self.para.edgeselfinteraction and (not self.is_final_layer):
            f_node, f_edge = self.edgeselfinteraction(f_node, f_edge,  node_emb, length_emb, data)

        f_edge = self.gate(f_edge)
        f_node = self.scatter(f_edge, edge_dst, dim=0, dim_size=data.num_nodes)
        f_node = self.lin2_node(f_node)
        f_edge = self.lin2_edge(f_edge)

        # add self-connection
        f_node = f_node+sc_node
        f_edge = f_edge+sc_edge

        return f_node, f_edge


if __name__ == '__main__':
    pass
