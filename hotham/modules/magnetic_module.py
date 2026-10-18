import torch
import e3nn
from torch_geometric.data import Data
from typing import List
from .te3nn.to3 import TO3_Irreps, TO3_Linear
from .common import TO3LayerNormal, MyScatter
from .embedding import NodeEmbedding, EdgeEmbedding
from .general_local_convolution import TO2_wrap, GeneralLocalConv
from .self_interaction import Self_interaction
from .output_linear import OutputLinear
from .long_range import LongRange


class MagEmbedding(torch.nn.Module):
    def __init__(self, lmax, basis_size, mag_cutoff, para=None, **kwargs):
        super().__init__()
        self.lmax = lmax
        self.basis_size = basis_size
        self.mag_cutoff = mag_cutoff
        self.para = para

        # self.mag_sh_o3irreps = e3nn.o3.Irreps.spherical_harmonics(lmax=self.lmax, p=1)
        self.mag_sh_emb_to3irreps = TO3_Irreps.spherical_harmonics(lmax=lmax, p=1, t_p=-1)
        self.mag_sh_emb = e3nn.o3.SphericalHarmonics(irreps_out=e3nn.o3.Irreps.spherical_harmonics(self.lmax), normalize=True)

        self.mag_length_emb_to3irreps = TO3_Irreps([(basis_size, (0, 1, 1))])
        self.mag_length_emb = EdgeEmbedding(basis_size, self.mag_cutoff)

    def forward(self, magmom: torch.Tensor):
        magmom_vec = torch.stack(
            [
                torch.sin(magmom[:, 2] * torch.pi / 180) * torch.sin(magmom[:, 3] * torch.pi / 180),
                torch.cos(magmom[:, 2] * torch.pi / 180),
                torch.sin(magmom[:, 2] * torch.pi / 180) * torch.cos(magmom[:, 3] * torch.pi / 180),
            ],
            dim=-1,
        )
        mag_sh = self.mag_sh_emb(magmom_vec) * magmom[:, 0:1]
        mag_length = torch.clamp(
            magmom[:, 1],
            min=self.mag_cutoff[0],
            max=self.mag_cutoff[1],
        )
        mag_length_emb = self.mag_length_emb(mag_length) * magmom[:, 0:1]
        return mag_sh, mag_length_emb


class LocalMAgEnv(torch.nn.Module):
    def __init__(self, irreps_node, irreps_out=None, para=None, **kwargs):
        super().__init__()
        self.irreps_node = irreps_node
        self.irreps_out = None
        self.para = para

        if self.irreps_out is None:
            irreps_out = TO3_Irreps(None)
            # lmax = self.para.L_max + 1
            lmax = 1
            irreps_sh = TO3_Irreps.spherical_harmonics(lmax, p=-1, t_p=1)
            for mul1, ir1 in irreps_sh:
                for mul2, ir2 in irreps_node:
                    irs3 = TO3_Irreps(list(ir1 * ir2))
                    irreps_out += irs3
            irreps_out = irreps_out.sort()[0].simplify()
            irreps_out = TO3_Irreps([irs for irs in irreps_out if irs.ir.l <= lmax])
            self.irreps_out = irreps_out

        self.to2tp = TO2_wrap(
            irreps_in=self.irreps_node,
            irreps_out=self.irreps_out,
            embedding_dim=self.para.basis_size,
            shared_weights=True,
            edge_include_sc=para.edge_include_sc,
            irreps_sh=None,
            intdtype=self.para.torch_int_dtype,
            floatdtype=self.para.torch_float_dtype,
        )

        self.lin_by_mag_lenght = OutputLinear(irreps_in=self.irreps_out, irreps_out=self.irreps_out, shared_weights=False, intdtype=self.para.torch_int_dtype)
        self.weight_for_out_by_mag_length = e3nn.nn.FullyConnectedNet(
            [self.para.basis_size, 64, self.lin_by_mag_lenght.weight_numel], torch.nn.functional.silu
        )

        self.scatter = MyScatter(self.para.N_average)

    def forward(self, x, length_emb, mag_length_emb, data: Data):
        edge_src, edge_dst = data.edge_index_hop
        out = self.to2tp(x=x[edge_src], edge_sh=None, length_emb=length_emb, wigner_D=data.wigner_D, mask_edge=data.mask_edge, mask_sc=data.mask_sc)

        weight_for_out_by_mag_length = self.weight_for_out_by_mag_length(mag_length_emb[edge_src])
        out = self.lin_by_mag_lenght(out, weight_for_out_by_mag_length)

        out = self.scatter(out, edge_dst, dim=0, dim_size=data.num_nodes)

        return out


class Self_Interaction_by_Edge(torch.nn.Module):
    def __init__(self, irreps_node, irreps_edge, irreps_localmagenv, channel, para):
        super().__init__()
        self.irreps_node = irreps_node
        self.irreps_edge = irreps_edge
        self.irreps_localmagenv = irreps_localmagenv
        self.channel = channel
        self.para = para
        self.basis_size = self.para.basis_size

        self.irreps_cat = self.irreps_edge + (self.irreps_node + self.irreps_node) + (self.irreps_localmagenv + self.irreps_localmagenv)
        irreps_cat_sort = self.irreps_cat.sort()[0].simplify()
        irreps_out = TO3_Irreps(None)
        for mul1, ir1 in irreps_cat_sort:
            for mul2, ir2 in irreps_cat_sort:
                p = ir1.p * ir2.p
                t_p = ir1.t_p * ir2.t_p
                l3_max = ir1.l + ir2.l
                l_list = [l3 for l3 in range(abs(ir1.l - ir2.l), min([l3_max, self.para.L_max + 1]) + 1) if (ir1.l + ir2.l + l3) % 2 == 0]
                irreps_out += TO3_Irreps([(1, (l3, p, t_p)) for l3 in l_list])
        irreps_out = irreps_out.sort()[0].simplify()
        irreps_out = TO3_Irreps([(self.channel, ir) for _, ir in irreps_out])
        self.irreps_out = irreps_out

        self.lin_prev = OutputLinear(irreps_in=self.irreps_cat, irreps_out=self.irreps_out, intdtype=self.para.torch_int_dtype)
        self.self_interaction = Self_interaction(
            irreps_in=self.irreps_out, irreps_out=self.irreps_out, channel=int(self.channel / 4), symmetrized=True, para=para
        )
        self.lin_post = OutputLinear(irreps_in=self.irreps_out, irreps_out=self.irreps_out, shared_weights=False, intdtype=self.para.torch_int_dtype)
        self.weight_for_lin_post = e3nn.nn.FullyConnectedNet([self.basis_size * 3, 64, self.lin_post.weight_numel], torch.nn.functional.silu)

    def forward(self, f_node, f_edge, local_mag_env, length_emb, mag_length_emb, data):
        edge_src, edge_dst = data.edge_index_hop
        f_cat = torch.cat([f_edge, f_node[edge_src], f_node[edge_dst], local_mag_env[edge_src], local_mag_env[edge_dst]], dim=-1)
        f_cat = self.lin_prev(f_cat)
        f_edge = self.self_interaction(f_cat)

        weight_for_lin = self.weight_for_lin_post(torch.cat([length_emb, mag_length_emb[edge_src], mag_length_emb[edge_dst]], dim=-1))
        f_edge = self.lin_post(x=f_edge, weights=weight_for_lin)

        return f_edge


class MagMessagePassing(torch.nn.Module):
    def __init__(self, irreps_node, irreps_mids: List[TO3_Irreps], irreps_sh, para=None, **kwargs):
        super().__init__()
        if isinstance(irreps_node, e3nn.o3.Irreps):
            irreps_node = TO3_Irreps([(mul, (l, p, 1)) for mul, (l, p) in irreps_node])
        if isinstance(irreps_sh, e3nn.o3.Irreps):
            irreps_sh = TO3_Irreps([(mul, (l, p, 1)) for mul, (l, p) in irreps_sh])
        self.irreps_node = irreps_node
        self.irreps_edge = irreps_node
        self.irreps_mids = irreps_mids
        self.irreps_sh = irreps_sh
        self.para = para

        self.lmax = max([self.irreps_node.lmax] + [i.lmax for i in self.irreps_mids])
        self.mag_emb = MagEmbedding(lmax=self.lmax, basis_size=para.basis_size, mag_cutoff=para.mag_cutoff, para=para)

        self.local_mag_env = LocalMAgEnv(irreps_node=self.mag_emb.mag_sh_emb_to3irreps, para=self.para)

        Conv_layers = torch.nn.ModuleList([])
        if self.para.using_layernorm3:
            LayerNorm_node, LayerNorm_edge = torch.nn.ModuleList([]), torch.nn.ModuleList([])
        if self.para.self_interaction:
            SelfInteractions = torch.nn.ModuleList([])
        if self.para.self_interaction_by_edge:
            SelfInteractions_by_edge = torch.nn.ModuleList([])
        if self.para.using_longrange:
            Longranges = torch.nn.ModuleList([])
        for layer_index in range(len(self.irreps_mids)):
            if layer_index == 0:
                irreps_edge_prev = self.irreps_edge
                if self.para.self_interaction_by_edge:
                    selfinteraction_by_edge = Self_Interaction_by_Edge(
                        irreps_node=irreps_node,
                        irreps_edge=irreps_node,
                        irreps_localmagenv=self.local_mag_env.irreps_out,
                        channel=self.para["mid_channel"][layer_index],
                        para=self.para,
                    )
                    SelfInteractions_by_edge.append(selfinteraction_by_edge)
                    irreps_edge_prev = selfinteraction_by_edge.irreps_out
                generallocalconv = GeneralLocalConv(
                    irreps_node=self.irreps_node,
                    irreps_edge=irreps_edge_prev,
                    irreps_sh=self.irreps_sh,
                    irreps_out=self.irreps_mids[layer_index],
                    num_type=self.para.num_types,
                    basis_size=self.para.basis_size,
                    para=self.para,
                    mag_length_emb_to3irreps=self.mag_emb.mag_length_emb_to3irreps,
                    mag_sh_emb_to3irreps=self.mag_emb.mag_sh_emb_to3irreps,
                    local_mag_env=self.local_mag_env.irreps_out,
                )
            else:
                irreps_edge_prev = Conv_layers[-1].local_irreps_out
                if self.para.self_interaction_by_edge:
                    selfinteraction_by_edge = Self_Interaction_by_Edge(
                        irreps_node=Conv_layers[-1].local_irreps_out,
                        irreps_edge=irreps_edge_prev,
                        irreps_localmagenv=self.local_mag_env.irreps_out,
                        channel=self.para["mid_channel"][layer_index],
                        para=self.para,
                    )
                    SelfInteractions_by_edge.append(selfinteraction_by_edge)
                    irreps_edge_prev = selfinteraction_by_edge.irreps_out
                generallocalconv = GeneralLocalConv(
                    irreps_node=Conv_layers[-1].local_irreps_out,
                    irreps_edge=irreps_edge_prev,
                    irreps_sh=self.irreps_sh,
                    irreps_out=self.irreps_mids[layer_index],
                    num_type=self.para.num_types,
                    basis_size=self.para.basis_size,
                    para=self.para,
                    mag_length_emb_to3irreps=self.mag_emb.mag_length_emb_to3irreps,
                    mag_sh_emb_to3irreps=self.mag_emb.mag_sh_emb_to3irreps,
                    local_mag_env=self.local_mag_env.irreps_out,
                )
            Conv_layers.append(generallocalconv)

            if self.para.using_layernorm3:
                LayerNorm_node.append(TO3LayerNormal(irreps_in=Conv_layers[layer_index].local_irreps_out, floatdtype=para.torch_float_dtype))
                LayerNorm_edge.append(TO3LayerNormal(irreps_in=Conv_layers[layer_index].local_irreps_out, floatdtype=para.torch_float_dtype))

            if self.para.self_interaction:
                selfinteraction = Self_interaction(
                    irreps_in=Conv_layers[layer_index].local_irreps_out,
                    irreps_out=Conv_layers[layer_index].local_irreps_out,
                    channel=self.para["self_interaction_channel"][layer_index],
                    para=self.para,
                )
                SelfInteractions.append(selfinteraction)

            if self.para.using_longrange:
                longrange = LongRange(irreps_node=Conv_layers[layer_index].local_irreps_out, para=self.para)
                Longranges.append(longrange)

        assert (
            self.irreps_mids[-1] == Conv_layers[-1].local_irreps_out
        ), f"irreps_mids[-1] {self.irreps_mids[-1]} != Conv_layers[-1].local_irreps_out {Conv_layers[-1].local_irreps_out}\nmissing irreps: {Conv_layers[-1].missing_irreps}"
        self.Conv_layers = Conv_layers
        if self.para.using_layernorm3:
            self.LayerNorm_node = LayerNorm_node
            self.LayerNorm_edge = LayerNorm_edge
        if self.para.self_interaction:
            self.SelfInteractions = SelfInteractions
        if self.para.self_interaction_by_edge:
            self.SelfInteractions_by_edge = SelfInteractions_by_edge
        if self.para.using_longrange:
            self.Longranges = Longranges

        self.node_out = Conv_layers[-1].irreps_out
        self.edge_out = Conv_layers[-1].irreps_out

    def forward(self, f_node, f_edge, edge_sh, node_emb, length_emb, data: Data):
        # AtomType = data.AtomType
        # edge_src, edge_dst = data.edge_index_hop

        mag_sh_emb, mag_length_emb = self.mag_emb(data.magmom)
        local_mag_env = self.local_mag_env(x=mag_sh_emb, length_emb=length_emb, mag_length_emb=mag_length_emb, data=data)

        for layer_index in range(len(self.irreps_mids)):
            if self.para.self_interaction_by_edge:
                f_edge = self.SelfInteractions_by_edge[layer_index](
                    f_node=f_node, f_edge=f_edge, local_mag_env=local_mag_env, length_emb=length_emb, mag_length_emb=mag_length_emb, data=data
                )
            f_node, f_edge = self.Conv_layers[layer_index](
                f_node=f_node,
                f_edge=f_edge,
                edge_sh=edge_sh,
                node_emb=node_emb,
                length_emb=length_emb,
                data=data,
                mag_length_emb=mag_length_emb,
                mag_sh_emb=mag_sh_emb,
                local_mag_env=local_mag_env,
            )
            if self.para.using_layernorm3:
                f_node = self.LayerNorm_node[layer_index](f_node)
                f_edge = self.LayerNorm_edge[layer_index](f_edge)
            if self.para.self_interaction:
                f_node = self.SelfInteractions[layer_index](f_node)
            if self.para.using_longrange:
                f_node_lr = self.Longranges[layer_index](f_node, data)
                f_node = (f_node + f_node_lr) / (2**0.5)

        return f_node, f_edge
