import math
import numpy as np
import torch
from torch.fft import fft, ifft, fft2, ifft2
import e3nn
from e3nn import o3
from typing import List, Union
import os
from torch_geometric.data import Data
from modules.common import MCST, E3LayerNormal


class GauntTwoBody(torch.nn.Module):
    def __init__(self,
                 irreps_in: Union[o3.Irreps, List[o3.Irreps]],
                 irreps_out: Union[o3.Irreps, List[o3.Irreps]],
                 **kwargs):
        super().__init__()

        # check irreps_in
        if isinstance(irreps_in, o3.Irreps):
            irreps_in = [irreps_in]
        elif isinstance(irreps_in, list) and irreps_in[1] == o3.Irreps([]):
            irreps_in.pop()
        assert len(irreps_in) <= 2
        lmax = irreps_in[0].lmax
        input_channels = [irs[0].mul for irs in irreps_in]
        for c, irs, p_val in zip(input_channels, irreps_in, [1, -1]):
            assert irs == MCST(lmax, p_val, c)

        # check irreps_out
        if isinstance(irreps_out, o3.Irreps):
            irreps_out = [irreps_out]
        elif isinstance(irreps_out, list) and irreps_out[1] == o3.Irreps([]):
            irreps_out.pop()
        assert len(irreps_out) <= 2
        output_channels = [irs[0].mul for irs in irreps_out]
        for c, irs, p_val in zip(output_channels, irreps_out, [1, -1]):
            assert irs == MCST(lmax, p_val, c)

        self.v_max = 2
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.lmax = lmax
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.c1_size = sum(input_channels)
        self.c3_size = sum(input_channels)

        # slice according to parity, irreps_in's channel and irreps_out's channel
        parity_slice, channel_in_slice, channel_out_slice, channel_mid_slice = [], [], [], []
        pin0_size, cin0_size, cout0_size, cmid0_size = 0, 0, 0, 0
        for index in range(len(irreps_in)):
            pin_size = irreps_in[index].dim
            parity_slice.append(slice(pin0_size, pin0_size+pin_size))
            pin0_size += pin_size

            cin_size = self.input_channels[index]
            channel_in_slice.append(slice(cin0_size, cin0_size+cin_size))
            cin0_size += cin_size

            cout_size = self.output_channels[index]
            channel_out_slice.append(slice(cout0_size, cout0_size+cout_size))
            cout0_size += cout_size

            cmid_size = self.input_channels[index]*self.v_max
            channel_mid_slice.append(slice(cmid0_size, cmid_size))
            cmid0_size += cmid_size

        self.parity_slice = parity_slice
        self.channel_in_slice = channel_in_slice
        self.channel_out_slice = channel_out_slice
        self.channel_mid_slice = channel_mid_slice

        # weights' shape
        w1_shape, wc_shape = [], []
        for index in range(len(irreps_in)):
            w1_shape.append((self.lmax + 1, self.input_channels[index], self.input_channels[index]*self.v_max))
            wc_shape.append((self.lmax + 1, self.input_channels[index]*(self.v_max-1), self.output_channels[index]))
        self.w1_shape = w1_shape
        self.wc_shape = wc_shape

        # weights' slice
        w1_slices = [slice(0, math.prod(self.w1_shape[0]))]
        if len(self.w1_shape) == 2:
            w1_slices.append(slice(w1_slices[0].stop, w1_slices[0].stop+math.prod(self.w1_shape[0])))
        self.w1_slices = w1_slices
        wc_slices = [slice(self.w1_slices[-1].stop, self.w1_slices[-1].stop+math.prod(self.wc_shape[0]))]
        if len(self.wc_shape) == 2:
            wc_slices.append(slice(wc_slices[0].stop, wc_slices[0].stop+math.prod(self.wc_shape[1])))
        self.wc_slices = wc_slices
        self.weight_numel = wc_slices[-1].stop

        # read coefficients that from sh to fourier(Ciuv) and that from fourier back to sh(Cuvi)
        _, ciuv, _, _, _ = self.read_precompute(self.lmax)
        iml, _, _, _, rcuvi_2l = self.read_precompute(self.lmax*2)
        self.register_buffer("iml", iml)
        self.register_buffer("Ciuv", ciuv)
        self.register_buffer("RCuvi_2l", rcuvi_2l)

    def forward(self, feature: torch.Tensor, weight=None, data=None, mask_edge=None, mask_sc=None):
        weight1 = [weight[..., s].reshape((-1,)+self.w1_shape[i]) for i, s in enumerate(self.w1_slices)]
        weightc = [weight[..., s].reshape((-1,)+self.wc_shape[i]) for i, s in enumerate(self.wc_slices)]

        feature = self.preprocess(feature, weight1)
        dtype_feature = feature.dtype
        image = torch.einsum("bci,iuv->bcuv", feature.to(self.Ciuv.dtype), self.Ciuv)

        image_shape = image.shape
        image_shape = (image_shape[0], self.v_max, int(image_shape[1]/self.v_max), image_shape[-2], image_shape[-1])
        image = image.reshape(image_shape)

        size = [2*self.v_max*self.lmax+1, 2*self.v_max*self.lmax+1]
        cut0 = (self.v_max-2)*self.lmax
        cut1 = self.v_max*self.lmax+1
        image_fft = fft2(image, s=size)
        B2_fft = image_fft[:, 0]*image_fft[:, 1]
        B2 = ifft2(B2_fft, s=size)[..., cut0:cut1, cut0:cut1]

        feature = torch.einsum("bcuv,iuv->bci", B2, self.RCuvi_2l[:(self.lmax+1)**2]).real
        feature = self.postprcess(feature, weightc)

        return feature.to(dtype_feature)

    def preprocess(self, feature, weight1):
        new_feature = []
        for index, s1 in enumerate(self.parity_slice):
            f = feature[:, s1]
            new_shape = (f.shape[0], self.input_channels[index], -1)
            irs_in1 = self.irreps_in[index]
            f = torch.cat([torch.einsum("bci,bca->bai", f[:, s2].reshape(new_shape), weight1[index][:, i, :]) for i, s2 in enumerate(irs_in1.slices())], dim=-1)
            new_feature.append(f)
        new_feature = torch.cat(new_feature, dim=-2)
        return new_feature

    def postprcess(self, feature, weightc):
        feature_mixed_channel = []
        for index, s in enumerate(self.channel_mid_slice):
            f = feature[:, s, :]
            new_shape = (f.shape[0], -1)
            f = torch.cat([torch.einsum("bci,bca->bai", f[..., l**2:(l+1)**2], weightc[index][:, l, ...]).reshape(new_shape) for l in range(self.lmax+1)], dim=-1)
            feature_mixed_channel.append(f)
        feature = torch.cat(feature_mixed_channel, dim=-1)
        return feature

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


class EdgeSelfInteraction(torch.nn.Module):
    def __init__(self,
                 v_max,
                 irreps_in: Union[o3.Irreps, List[o3.Irreps]],
                 irreps_out: Union[o3.Irreps, List[o3.Irreps]],
                 num_type: int,
                 basis_size=32,
                 hidden_neurons=64,
                 split_stru=0,
                 para=None,
                 **kwargs):
        super().__init__()

        self.v_max = v_max
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.split_stru = split_stru

        self.gmb = GauntTwoBody(v_max=self.v_max, irreps_in=self.irreps_in, irreps_out=self.irreps_out)
        self.weight_gmb = e3nn.nn.FullyConnectedNet([2*num_type+basis_size,
                                                    hidden_neurons,
                                                    self.gmb.weight_numel],
                                                    torch.nn.functional.silu)
        self.self_connection_edge = o3.FullyConnectedTensorProduct(self.irreps_in,
                                                                   o3.Irreps(f"{2*num_type+basis_size}x0e"),
                                                                   self.irreps_out)
        # self.layernorm = E3LayerNormal(irreps_in=self.irreps_in)

    def forward(self, f_node, f_edge=None, node_emb=None, length_emb=None, data: Data = None):
        edge_src, edge_dst = data.edge_index_hop
        emb = torch.cat([node_emb[edge_src], node_emb[edge_dst], length_emb], dim=-1)
        weight = self.weight_gmb(emb)
        self_connection = self.self_connection_edge(f_edge, emb)

        if self.split_stru > 1:
            sub_batch_size = math.ceil(data.num_edges/self.split_stru)
            f_edge = [[] for _ in range(self.split_stru)]
            for index_sub_batch in range(self.split_stru):
                if index_sub_batch == self.split_stru-1:
                    batch_slice = slice(sub_batch_size*index_sub_batch, data.num_edges)
                else:
                    batch_slice = slice(sub_batch_size*index_sub_batch, sub_batch_size*(index_sub_batch+1))
                sub_f_edge = self.gmb(f_edge[batch_slice], weight[batch_slice])
                f_edge[index_sub_batch] = sub_f_edge
            f_edge = torch.cat(f_edge, dim=0)
        else:
            f_edge = self.gmb(f_edge, weight)

        f_edge = self_connection+f_edge
        # f_edge = self.layernorm(f_edge)
        return f_node, f_edge


if __name__ == "__main__":
    torch.manual_seed(2025)
    lmax = 8
    irreps_in = MCST(lmax, 1, 3)
    irreps_out = MCST(lmax, 1, 3)

    gtb = GauntTwoBody(irreps_in, irreps_out)

    n = 2
    weight = torch.rand(size=(n, gtb.weight_numel))
    R_cpu = o3.rand_matrix(1).squeeze()
    D = gtb.irreps_out[0].D_from_matrix(R_cpu)
    input_random = irreps_in.randn(n, -1)
    input_random_rotated = torch.einsum("ij,zj->zi", D, input_random)

    f = gtb(input_random, weight)
    f_rotated = gtb(input_random_rotated, weight)

    f_tmp = torch.einsum("ij,zj->zi", D, f)
    delta = (f_tmp-f_rotated).abs().max()
    max_value = max([f_tmp.abs().max(), f_rotated.abs().max()])
    ratio = delta/max_value
    print(delta, ratio)

    pass
