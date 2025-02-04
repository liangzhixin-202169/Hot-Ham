import math
import numpy as np
import torch
from torch.fft import fft, ifft, fft2, ifft2
from torch_geometric.data import Data
# from torch_scatter import scatter
from typing import List, Union
import os
import e3nn
from e3nn import o3
from .common import MCST, Gate, E3LayerNormal, MyScatter

__all__ = ["GauntTensorProduct", "GauntConvelution"]


class GauntTensorProduct(torch.nn.Module):
    """general Gaunt tensor product (sphere-grid)"""

    def __init__(self,
                 irreps_in1: Union[o3.Irreps, List[o3.Irreps]],
                 irreps_in2: Union[o3.Irreps, None],
                 irreps_out: Union[o3.Irreps, List[o3.Irreps]],
                 shared_weights=True,
                 edge_include_sc=True,
                 **kwargs):
        super().__init__()

        # check irreps_in1
        if isinstance(irreps_in1, o3.Irreps):
            irreps_in1 = [irreps_in1]
        elif isinstance(irreps_in1, list) and irreps_in1[1] == o3.Irreps([]):
            irreps_in1.pop()
        assert len(irreps_in1) <= 2
        lmax = irreps_in1[0].lmax
        input1_channels = [irs[0].mul for irs in irreps_in1]
        for c, irs, p_val in zip(input1_channels, irreps_in1, [1, -1]):
            assert irs == MCST(lmax, p_val, c)

        # check irreps_in2
        if irreps_in2 is None:
            irreps_in2 = o3.Irreps.spherical_harmonics(lmax=lmax)
        else:
            assert (irreps_in2 == MCST(lmax, 1, sum(input1_channels))) or (irreps_in2 == MCST(lmax, 1, 1))
        self.sh_channel = irreps_in2[0].mul

        # check irreps_out
        if isinstance(irreps_out, o3.Irreps):
            irreps_out = [irreps_out]
        elif isinstance(irreps_out, list) and irreps_out[1] == o3.Irreps([]):
            irreps_out.pop()
        assert len(irreps_out) <= 2
        output_channels = [irs[0].mul for irs in irreps_out]
        for c, irs, p_val in zip(output_channels, irreps_out, [1, -1]):
            assert irs == MCST(lmax, p_val, c)

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.shared_weights = shared_weights

        self.lmax = lmax
        self.input1_channels = input1_channels
        self.c1_size = sum(input1_channels)
        self.c3_size = sum(input1_channels)
        self.output_channels = output_channels

        # slice according to parity, irreps_in1's channel and irreps_out's channel
        parity_slice, channel_in_slice, channel_out_slice = [], [], []
        pin0_size, cin0_size, cout0_size = 0, 0, 0
        for index in range(len(irreps_in1)):
            pin_size = irreps_in1[index].dim
            parity_slice.append(slice(pin0_size, pin0_size+pin_size))
            pin0_size += pin_size

            cin_size = self.input1_channels[index]
            channel_in_slice.append(slice(cin0_size, cin0_size+cin_size))
            cin0_size += cin_size

            cout_size = self.output_channels[index]
            channel_out_slice.append(slice(cout0_size, cout0_size+cout_size))
            cout0_size += cout_size
        self.parity_slice = parity_slice
        self.channel_in_slice = channel_in_slice
        self.channel_out_slice = channel_out_slice

        # share weights if shared_weights, else use weights from embedding
        # w(l1,l2,l3,c1,c2) is decomposed into w(l1)*w(l2)*w(l3)*w(l3,c1,c2)
        if self.shared_weights:
            self.w1 = torch.nn.Parameter(torch.randn(self.c1_size, self.lmax + 1))
            self.w2 = torch.nn.Parameter(torch.randn(self.lmax + 1))
            self.w3 = torch.nn.Parameter(torch.randn(self.c12_size, self.lmax + 1))
            wc = []
            for index in range(len(irreps_in1)):
                wc.append(torch.nn.Parameter(torch.randn(self.lmax + 1, self.input1_channels[index], self.output_channels[index])))
            self.wc = torch.nn.ParameterList(wc)

        else:
            self.w1_shape = (self.c1_size, self.lmax + 1)
            self.w2_shape = (self.sh_channel, self.lmax + 1)
            wc_shape = []
            for index in range(len(irreps_in1)):
                wc_shape.append((self.lmax + 1, self.input1_channels[index], self.output_channels[index]))
            self.wc_shape = wc_shape

            self.w1_slice = slice(0, math.prod(self.w1_shape))
            self.w2_slice = slice(self.w1_slice.stop, self.w1_slice.stop+math.prod(self.w2_shape))
            wc_slices = [slice(self.w2_slice.stop, self.w2_slice.stop+math.prod(self.wc_shape[0]))]
            if len(self.wc_shape) == 2:
                wc_slices.append(slice(wc_slices[0].stop, wc_slices[0].stop+math.prod(self.wc_shape[1])))
            self.wc_slices = wc_slices

            self.weight_numel = wc_slices[-1].stop

        res_beta = 4*(self.lmax+1) if self.lmax % 2 == 1 else 4*self.lmax
        res = (res_beta, res_beta-1)
        self.tos2grid = o3.ToS2Grid(lmax=self.lmax, res=res)
        self.froms2grid = o3.FromS2Grid(res=res, lmax=self.lmax)

    def forward(self, feature: torch.Tensor, sh: torch.Tensor, weight=None, data=None, mask_edge=None, mask_sc=None):
        feature = self.preprocess(feature)

        if self.shared_weights:
            weight1 = torch.cat([self.w1[:, [l]].expand(-1, 2*l+1) for l in range(self.lmax+1)], dim=-1)
            weight2 = torch.cat([self.w2[l].expand(2*l+1) for l in range(self.lmax+1)], dim=-1)
            weightc = self.wc
        else:
            weight1 = weight[..., self.w1_slice].reshape((-1,)+self.w1_shape)
            weight1 = torch.cat([weight1[..., [l]].expand(-1, -1, 2*l+1) for l in range(self.lmax+1)], dim=-1)
            weight2 = weight[..., self.w2_slice].reshape((-1,)+self.w2_shape)
            weight2 = torch.cat([weight2[..., [l]].expand(-1, -1, 2*l+1) for l in range(self.lmax+1)], dim=-1)
            weightc = [weight[..., s].reshape((-1,)+self.wc_shape[i]) for i, s in enumerate(self.wc_slices)]

        image = torch.einsum("bci,bci->bci", feature, weight1)
        x1 = self.tos2grid(image)

        if len(sh.shape) == 2:
            weight2 = weight2.squeeze(1)
        kernel = torch.einsum("bi,bi->bi", sh, weight2)
        x2 = self.tos2grid(kernel)

        x3 = x1*x2[:, None, :, :]

        feature = self.froms2grid(x3)

        feature_mixed_channel = []
        for index, s in enumerate(self.channel_in_slice):
            f = feature[:, s, :]
            new_shape = (f.shape[0], -1)
            f = torch.cat([torch.einsum("bci,bca->bai", f[..., l**2:(l+1)**2], weightc[index][:, l, ...]).reshape(new_shape) for l in range(self.lmax+1)], dim=-1)
            feature_mixed_channel.append(f)

        feature = torch.cat(feature_mixed_channel, dim=-1)

        return feature.to(sh.dtype)

    def preprocess(self, feature):
        new_feature = []
        for index, s1 in enumerate(self.parity_slice):
            f = feature[:, s1]
            new_shape = (f.shape[0], self.input1_channels[index], -1)
            irs_in1 = self.irreps_in1[index]
            f = torch.cat([f[:, s2].reshape(new_shape) for s2 in irs_in1.slices()], dim=-1)
            new_feature.append(f)
        new_feature = torch.cat(new_feature, dim=-2)
        return new_feature

    def read_precompute(self, lmax):
        npcomplex = np.complex64
        precompute_path = os.path.join(os.path.dirname(__file__), f"../utilities/precompute_{lmax}.npy")
        assert os.path.exists(precompute_path)
        precompute = np.load(precompute_path, allow_pickle=True).item()
        iml = torch.from_numpy(precompute['ilm'])
        ciuv = torch.from_numpy(precompute['Ciuv'].astype(npcomplex))
        cuvi = torch.from_numpy(precompute['Cuvi'].astype(npcomplex))
        fciuv = torch.from_numpy(precompute['FCiuv'].astype(npcomplex))
        rcuvi = torch.from_numpy(precompute['RCuvi'].astype(npcomplex))
        return iml, ciuv, cuvi, fciuv, rcuvi


class GauntTensorProduct_LCT(torch.nn.Module):
    """TODO: implement Gaunt tensor product (sphere-grid) with local coordinate transformation"""

    def __init__(self,
                 irreps_in1: Union[o3.Irreps, List[o3.Irreps]],
                 irreps_in2: Union[o3.Irreps, None],
                 irreps_out: Union[o3.Irreps, List[o3.Irreps]],
                 shared_weights=True,
                 edge_include_sc=True,
                 **kwargs):
        super().__init__()

        # check irreps_in1
        if isinstance(irreps_in1, o3.Irreps):
            irreps_in1 = [irreps_in1]
        elif isinstance(irreps_in1, list) and irreps_in1[1] == o3.Irreps([]):
            irreps_in1.pop()
        assert len(irreps_in1) <= 2
        lmax = irreps_in1[0].lmax
        input1_channels = [irs[0].mul for irs in irreps_in1]
        for c, irs, p_val in zip(input1_channels, irreps_in1, [1, -1]):
            assert irs == MCST(lmax, p_val, c)

        # check irreps_in2
        if irreps_in2 is None:
            irreps_in2 = o3.Irreps.spherical_harmonics(lmax=lmax)
        else:
            assert (irreps_in2 == MCST(lmax, 1, sum(input1_channels))) or (irreps_in2 == MCST(lmax, 1, 1))
        self.sh_channel = irreps_in2[0].mul

        # check irreps_out
        if isinstance(irreps_out, o3.Irreps):
            irreps_out = [irreps_out]
        elif isinstance(irreps_out, list) and irreps_out[1] == o3.Irreps([]):
            irreps_out.pop()
        assert len(irreps_out) <= 2
        output_channels = [irs[0].mul for irs in irreps_out]
        for c, irs, p_val in zip(output_channels, irreps_out, [1, -1]):
            assert irs == MCST(lmax, p_val, c)

        self.irreps_in1 = irreps_in1
        self.irreps_in2 = irreps_in2
        self.irreps_out = irreps_out
        self.shared_weights = shared_weights
        self.edge_include_sc = edge_include_sc

        self.lmax = lmax
        self.input1_channels = input1_channels
        self.c1_size = sum(input1_channels)
        self.c3_size = sum(input1_channels)
        self.output_channels = output_channels

        # slice according to parity, irreps_in1's channel and irreps_out's channel
        parity_in_slice, parity_out_slice, channel_in_slice, channel_out_slice = [], [], [], []
        pin0_size, pout0_size, cin0_size, cout0_size = 0, 0, 0, 0
        for index in range(len(irreps_in1)):
            pin_size = irreps_in1[index].dim
            parity_in_slice.append(slice(pin0_size, pin0_size+pin_size))
            pin0_size += pin_size

            pout_size = irreps_out[index].dim
            parity_out_slice.append(slice(pout0_size, pout0_size+pout_size))
            pout_size += pout_size

            cin_size = self.input1_channels[index]
            channel_in_slice.append(slice(cin0_size, cin0_size+cin_size))
            cin0_size += cin_size

            cout_size = self.output_channels[index]
            channel_out_slice.append(slice(cout0_size, cout0_size+cout_size))
            cout0_size += cout_size
        self.parity_in_slice = parity_in_slice
        self.parity_out_slice = parity_out_slice
        self.channel_in_slice = channel_in_slice
        self.channel_out_slice = channel_out_slice

        # share weights if shared_weights, else use weights from embedding
        # w(l1,l2,l3,c1,c2) is decomposed into w(l1)*w(l2)*w(l3)*w(l3,c1,c2)
        if self.shared_weights:
            self.w1 = torch.nn.Parameter(torch.randn(self.c1_size, self.lmax + 1))
            self.w2 = torch.nn.Parameter(torch.randn(self.lmax + 1))
            self.w3 = torch.nn.Parameter(torch.randn(self.c12_size, self.lmax + 1))
            wc = []
            for index in range(len(irreps_in1)):
                wc.append(torch.nn.Parameter(torch.randn(self.lmax + 1, self.input1_channels[index], self.output_channels[index])))
            self.wc = torch.nn.ParameterList(wc)

        else:
            self.w1_shape = (self.c1_size, self.lmax + 1)
            self.w2_shape = (self.sh_channel, self.lmax + 1)
            # self.w3_shape = (self.c3_size, self.lmax + 1)
            wc_shape = []
            for index in range(len(irreps_in1)):
                wc_shape.append((self.lmax + 1, self.input1_channels[index], self.output_channels[index]))
            self.wc_shape = wc_shape

            self.w1_slice = slice(0, math.prod(self.w1_shape))
            self.w2_slice = slice(self.w1_slice.stop, self.w1_slice.stop+math.prod(self.w2_shape))
            # self.w3_slice = slice(self.w2_slice.stop, self.w2_slice.stop+math.prod(self.w3_shape))
            # wc_slices = [slice(self.w3_slice.stop, self.w3_slice.stop+math.prod(self.wc_shape[0]))]
            wc_slices = [slice(self.w2_slice.stop, self.w2_slice.stop+math.prod(self.wc_shape[0]))]
            if len(self.wc_shape) == 2:
                wc_slices.append(slice(wc_slices[0].stop, wc_slices[0].stop+math.prod(self.wc_shape[1])))
            self.wc_slices = wc_slices

            self.weight_numel = wc_slices[-1].stop

        # read coefficients that from sh to fourier(Ciuv) and that from fourier back to sh(Cuvi)
        _, ciuv, _, rciuv, _, hciuv, _ = self.read_precompute(self.lmax)
        _, _, _, _, _, _, rcuvi_2l = self.read_precompute(self.lmax*2)
        self.register_buffer("Ciuv", ciuv)
        self.register_buffer("RCiuv", rciuv)
        self.register_buffer("HCiuv", hciuv)
        self.register_buffer("RCuvi_2l", rcuvi_2l)

        # spherical along z direction, the edge with length=0(sc) will be projected to scalar
        sh_z_edge = torch.tensor([((2*l+1)/(4*torch.pi))**0.5 for l in range(self.lmax+1)]).to(self.RCiuv.dtype)
        sh_z_sc = torch.zeros_like(sh_z_edge)
        sh_z_sc[0] = sh_z_edge[0]
        self.register_buffer("sh_z_edge", sh_z_edge)
        self.register_buffer("sh_z_sc", sh_z_sc)

    def forward(self, feature: torch.Tensor, sh: torch.Tensor, weight=None, wigner_D=None, mask_edge=None, mask_sc=None):
        feature = self.preprocess(feature, wigner_D)

        if self.shared_weights:
            weight1 = torch.cat([self.w1[:, [l]].expand(-1, 2*l+1) for l in range(self.lmax+1)], dim=-1)
            weight2 = torch.cat([self.w2[l].expand(2*l+1) for l in range(self.lmax+1)], dim=-1)
            weight3 = torch.cat([self.w3[:, [l]].expand(-1, 2*l+1) for l in range(self.lmax+1)], dim=-1)
            weightc = self.wc
        else:
            weight1 = weight[..., self.w1_slice].reshape((-1,)+self.w1_shape)
            weight1 = torch.cat([weight1[..., [l]].expand(-1, -1, 2*l+1) for l in range(self.lmax+1)], dim=-1)
            weight2 = weight[..., self.w2_slice]
            weightc = [weight[..., s].reshape((-1,)+self.wc_shape[i]) for i, s in enumerate(self.wc_slices)]

        image = torch.einsum("bci,bci,iuv->bcuv", feature.to(self.HCiuv.dtype), weight1.to(self.Ciuv.dtype), self.HCiuv)
        if self.edge_include_sc:
            kernel = self.RCiuv.new_zeros((weight2.shape[0], self.RCiuv.shape[1]))
            kernel[mask_edge] = torch.einsum("i,bi,iu->bu", self.sh_z_edge, weight2[mask_edge].to(self.RCiuv.dtype), self.RCiuv)
            kernel[mask_sc] = torch.einsum("i,bi,iu->bu", self.sh_z_sc, weight2[mask_sc].to(self.RCiuv.dtype), self.RCiuv)
        else:
            kernel = torch.einsum("i,bi,iu->bu", self.sh_z_edge, weight2.to(self.RCiuv.dtype), self.RCiuv)

        image_fft = fft(image, n=4*self.lmax+1, dim=-2)
        kernel_fft = fft(kernel, n=4*self.lmax+1, dim=-1)
        IK_fft = torch.einsum("bcuv,bu->bcuv", image_fft, kernel_fft)
        IK_ifft = ifft(IK_fft, n=4*self.lmax+1, dim=-2)

        feature = torch.einsum("bcuv,iuv->bci", IK_ifft[..., :2*self.lmax+1, :], self.RCuvi_2l[:(self.lmax+1)**2, :, self.lmax:2*self.lmax+1]).real

        out = feature.new_zeros((feature.shape[0], sum(self.output_channels)*(self.lmax+1)**2))
        # for index, sp_in, sp_out, sc_in, sc_out in zip([0, 1], self.parity_in_slice, self.parity_out_slice, self.channel_in_slice, self.channel_out_slice):
        for index, sp_out, sc_in in zip([0, 1], self.parity_out_slice, self.channel_in_slice):
            irs_out = self.irreps_out[index]
            for l, s2 in zip(range(0, self.lmax+1), irs_out.slices()):
                out[:, sp_out][:, s2] = torch.einsum("bij,bcj,bca->bai", wigner_D[l], feature[:, sc_in, l**2:(l+1)**2], weightc[index][:, l, ...]).reshape(feature.shape[0], -1)

        return out

    def preprocess(self, feature, wigner_D):
        new_feature = feature.new_zeros((feature.shape[0], sum(self.input1_channels), (self.lmax+1)**2))
        for index, sp, sc1 in zip([0, 1], self.parity_in_slice, self.channel_in_slice):
            irs_in1 = self.irreps_in1[index]
            for l, s2 in zip(range(self.lmax+1), irs_in1.slices()):
                new_feature[:, sc1, l**2:(l+1)**2] = torch.einsum("bji,bcj->bci", wigner_D[l], feature[:, sp][:, s2].reshape(feature.shape[0], self.input1_channels[index], 2*l+1))

        return new_feature

    def read_precompute(self, lmax):
        npcomplex = np.complex64
        precompute_path = os.path.join(os.path.dirname(__file__), f"../utilities/precompute_{lmax}.npy")
        assert os.path.exists(precompute_path)
        precompute = np.load(precompute_path, allow_pickle=True).item()
        iml = torch.from_numpy(precompute['ilm'])
        ciuv = torch.from_numpy(precompute['Ciuv'].astype(npcomplex))
        cuvi = torch.from_numpy(precompute['Cuvi'].astype(npcomplex))
        rciuv = torch.from_numpy(precompute['RCiuv'].astype(npcomplex))
        fciuv = torch.from_numpy(precompute['FCiuv'].astype(npcomplex))
        hciuv = torch.from_numpy(precompute['HCiuv'].astype(npcomplex))
        rcuvi = torch.from_numpy(precompute['RCuvi'].astype(npcomplex))
        return iml, ciuv, cuvi, rciuv, fciuv, hciuv, rcuvi


class GauntConvelution(torch.nn.Module):
    def __init__(self, irreps_node: o3.Irreps, irreps_out: o3.Irreps,  num_type: int, basis_size=12, sh_channel=None, hidden_neurons=64, split_stru=0, para=None):
        super().__init__()
        self.irreps_node = irreps_node
        self.irreps_out = irreps_out
        self.basis_size = basis_size
        self.lmax = self.irreps_node.lmax
        self.split_stru = split_stru
        self.using_layernorm2 = para.using_layernorm2
        # self.fix_average = para.fix_average
        self.scatter = MyScatter(para.fix_average, para.N_average)

        ##############################################################################################################################################
        # Split irreps_in and irreps_out according to even and parity, such that irreps == "Cex0e+Cex1o+Cex2e+..." + "Cox0o+Cox1e+Cox2o+...".
        # irreps_in is concatenated from two irreps_nodes and one irreps_edge, where irreps_node and irreps_edge are the same form, and is then rearranged.
        ##############################################################################################################################################

        irreps_in1 = o3.Irreps([(mul*3, (l, p)) for mul, (l, p) in self.irreps_node if (-1)**l == p])
        irreps_in2 = o3.Irreps([(mul*3, (l, p)) for mul, (l, p) in self.irreps_node if (-1)**l == -p])
        irreps_in = [irreps_in1, irreps_in2]

        if sh_channel is None:
            sh_channel = sum([irs[0].mul for irs in irreps_in])
        self.irreps_sh = MCST(self.lmax, p_val=1, channel=sh_channel)

        irreps_out1 = o3.Irreps([(mul, (l, p)) for mul, (l, p) in self.irreps_out if (-1)**l == p])
        irreps_out2 = o3.Irreps([(mul, (l, p)) for mul, (l, p) in self.irreps_out if (-1)**l == -p])
        irreps_out = [irreps_out1, irreps_out2]

        ##############################################################################################################################################
        # Define net module
        ##############################################################################################################################################

        # self-connection
        self.sc_node = o3.FullyConnectedTensorProduct(self.irreps_node, o3.Irreps(f"{num_type}x0e"), self.irreps_out)
        self.sc_edge = o3.FullyConnectedTensorProduct(self.irreps_node, o3.Irreps(f"{2*num_type+basis_size}x0e"), self.irreps_out)

        # lin1(f_node)->e3tp->gate->scatter->lin2_node
        #                        \->lin2_edge
        self.lin1_node = o3.Linear(irreps_in=self.irreps_node, irreps_out=self.irreps_node)
        self.lin1_edge = o3.Linear(irreps_in=self.irreps_node, irreps_out=self.irreps_node)

        self.gtp = GauntTensorProduct(irreps_in1=irreps_in.copy(),
                                      irreps_in2=self.irreps_sh,
                                      irreps_out=irreps_out.copy(),
                                      shared_weights=False,
                                      edge_include_sc=para.edge_include_sc)
        self.weight_gtp = e3nn.nn.FullyConnectedNet([self.basis_size,
                                                     hidden_neurons,
                                                     self.gtp.weight_numel],
                                                    torch.nn.functional.silu)

        if self.using_layernorm2:
            self.layernorm = E3LayerNormal(self.irreps_out)

        self.gate = Gate(irreps_in=self.irreps_out)
        self.lin2_node = o3.Linear(irreps_in=self.irreps_out, irreps_out=self.irreps_out)
        self.lin2_edge = o3.Linear(irreps_in=self.irreps_out, irreps_out=self.irreps_out)

    def forward(self, f_node, f_edge, sh, node_emb, length_emb, data: Data):
        edge_src, edge_dst = data.edge_index_hop

        # self-connection
        sc_node = self.sc_node(f_node, node_emb)
        sc_edge = self.sc_edge(f_edge, torch.cat([node_emb[edge_src], node_emb[edge_dst], length_emb], dim=-1))

        # lin1(f_node)|              |->scatter->lin2_node
        #             |->e3tp->gate->|
        # lin1(f_edge)|              |->lin2_edge
        f_node = self.lin1_node(f_node)
        f_edge = self.lin1_edge(f_edge)

        f_cat = torch.cat([torch.cat([f_node[:, s][edge_src], f_node[:, s][edge_dst], f_edge[:, s]], dim=-1) for s in self.irreps_node.slices()], dim=-1)
        weight = self.weight_gtp(length_emb)

        if self.split_stru > 1:
            sub_batch_size = math.ceil(data.num_edges/self.split_stru)
            f_edge = [[] for _ in range(self.split_stru)]
            for index_sub_batch in range(self.split_stru):
                if index_sub_batch == self.split_stru-1:
                    batch_slice = slice(sub_batch_size*index_sub_batch, data.num_edges)
                else:
                    batch_slice = slice(sub_batch_size*index_sub_batch, sub_batch_size*(index_sub_batch+1))
                sub_f_edge = self.gtp(f_cat[batch_slice], sh[batch_slice], weight[batch_slice], [D[batch_slice] for D in data.wigner_D], data.mask_edge[batch_slice], data.mask_sc[batch_slice])
                f_edge[index_sub_batch] = sub_f_edge
            f_edge = torch.cat(f_edge, dim=0)
        else:
            f_edge = self.gtp(f_cat, sh, weight, data.wigner_D, data.mask_edge, data.mask_sc)

        if self.using_layernorm2:
            f_edge = self.layernorm(f_edge)

        f_edge = self.gate(f_edge)
        f_node = self.scatter(f_edge, edge_dst, dim=0, dim_size=data.num_nodes)

        f_node = self.lin2_node(f_node)
        f_edge = self.lin2_edge(f_edge)

        # add self-connection
        f_node = f_node+sc_node
        f_edge = f_edge+sc_edge

        return f_node, f_edge
