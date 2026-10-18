import os
import torch
from torch_geometric.data import Data
from typing import Union
import e3nn
from e3nn import o3
from .parameters import Parameters
from ..modules.embedding import NodeEmbedding, EdgeEmbedding
from ..modules.so2_convolution import SO2OutputLayer
from ..modules.general_local_convolution import GeneralLocalConv
from ..modules.common import MCST, E3LayerNormal, MyScatter
from ..modules.output_linear import OutputLinear
from ..modules.magnetic_module import MagMessagePassing
from ..modules.te3nn.to3 import TO3_Irrep, TO3_Irreps, TO3_Linear
from .basemodel import BaseModel


class FeatureInitialize(torch.nn.Module):
    def __init__(self, irreps_in: Union[o3.Irreps, TO3_Irreps], node_emb_out_dim, edge_emb_out_dim, para: Union[dict, Parameters], **kwargs):
        super().__init__()
        self.first_channel = para.first_channel
        self.irreps_in = irreps_in
        if isinstance(irreps_in, o3.Irreps):
            irreps_class = o3.Irreps
            linear_class = o3.Linear
        elif isinstance(irreps_in, TO3_Irreps):
            irreps_class = TO3_Irreps
            linear_class = TO3_Linear
        self.irreps_out = irreps_class([(self.first_channel, ir) for l, ir in irreps_in])
        self.linear = linear_class(self.irreps_in, self.irreps_out, shared_weights=False)
        self.weight_func = e3nn.nn.FullyConnectedNet(
            [2 * node_emb_out_dim + edge_emb_out_dim, 64, self.linear.weight_numel], torch.nn.functional.silu
        )
        self.scatter = MyScatter(para.N_average)

    def forward(self, node_emb: torch.Tensor, length_emb: torch.Tensor, edge_sh_emb: torch.Tensor, data: Data):
        edge_src, edge_dst = data.edge_index_hop
        scalar_info_cat = weight_f = torch.cat([node_emb[edge_src], node_emb[edge_dst], length_emb], dim=-1)
        weights = self.weight_func(scalar_info_cat)
        f_edge = self.linear(edge_sh_emb, weights)
        f_node = self.scatter(f_edge, edge_dst, dim=0, dim_size=data.num_nodes)
        return f_node, f_edge


class Model(BaseModel):
    def __init__(self, para: Union[dict, Parameters], **kwargs):
        super().__init__(para, **kwargs)

        self.lmax = para.L_max
        self.basis_size = self.para.basis_size
        # self.Ce = para.Ce
        # self.Co = para.Co
        # assert len(self.Ce) == len(self.Co)

        # self.layer_irreps = [MCST(self.lmax, 1, self.Ce[i]) + MCST(self.lmax, -1, self.Co[i]) for i in range(len(self.Ce))]
        # self.layer_irreps = [ir.remove_zero_multiplicities() for ir in self.layer_irreps]
        # # restrict the last layer irreps
        # if para.restrict_last_out:
        #     self.layer_irreps[-1] = o3.Irreps([irs for irs in self.layer_irreps[-1] if irs[1].l <= self.edge_irreps_output.lmax])

        # # Define scatter function
        # self.scatter = MyScatter(para.N_average)

        # Embedding
        self.node_emb = NodeEmbedding(self.num_atomtype)
        self.edge_emb = EdgeEmbedding(self.basis_size, self.para.rc, basis=self.para.basis)
        self.edge_sh_emb = o3.SphericalHarmonics(irreps_out=o3.Irreps.spherical_harmonics(self.lmax), normalize=True)
        self.irreps_sh = self.edge_sh_emb.irreps_out

        # Feature initialize
        self.FeatureInitialize = FeatureInitialize(
            irreps_in=self.edge_sh_emb.irreps_out, node_emb_out_dim=self.node_emb.out_dim, edge_emb_out_dim=self.edge_emb.out_dim, para=para
        )

        self.magnetism = para.magnetism
        if self.magnetism:
            if para.get("mid_channel") is None:
                raise ValueError("Missing required input parameter 'mid_channel' for magnetic LCAO version 2")
            mmp_mid_to3_irreps_list = []
            for i in range(len(para["mid_channel"])):
                channel = para["mid_channel"][i]
                mmp_mid_irreps = MCST(self.lmax + 1, 1, channel) + MCST(self.lmax + 1, -1, channel)
                t_ps = [1 for _ in range(len(mmp_mid_irreps))]
                mmp_mid_irreps += mmp_mid_irreps
                t_ps += [-t_p for t_p in t_ps]
                mmp_mid_to3_irreps = TO3_Irreps([(mul, (l, p, t_p)) for (mul, (l, p)), t_p in zip(mmp_mid_irreps, t_ps)])
                mmp_mid_to3_irreps_reduced = TO3_Irreps(None)
                if para["reduced_to3"]:
                    # for mul, (l, p, t_p) in mmp_mid_to3_irreps:
                    #     if TO3_Irrep((l, p, t_p)) in self.edge_to3_irreps_output:
                    #         mmp_mid_to3_irreps_reduced += TO3_Irreps([(mul, (l, p, t_p))])
                    # mmp_mid_to3_irreps_list.append(mmp_mid_to3_irreps_reduced)
                    mmp_mid_to3_irreps_list.append(self.edge_to3_irreps_output.sort()[0].simplify())
                else:
                    mmp_mid_to3_irreps_list.append(mmp_mid_to3_irreps)

            self.messagepassing = MagMessagePassing(
                irreps_node=self.FeatureInitialize.irreps_out, irreps_mids=mmp_mid_to3_irreps_list, irreps_sh=self.irreps_sh, para=para
            )

        else:
            pass

        irreps_prev = self.messagepassing.node_out
        if not self.spin:
            irreps_out = self.edge_irreps_output
            if para.using_rearrangement_linear:
                self.node_endlinear = OutputLinear(irreps_prev, irreps_out, intdtype=self.para.torch_int_dtype)
                self.edge_endlinear = OutputLinear(irreps_prev, irreps_out, intdtype=self.para.torch_int_dtype)
            else:
                self.node_endlinear = o3.Linear(irreps_prev, irreps_out)
                self.edge_endlinear = o3.Linear(irreps_prev, irreps_out)
        else:
            irreps_out = self.edge_to3_irreps_output
            irreps_out_real = self.edge_to3_irreps_output_real
            irreps_out_imag = self.edge_to3_irreps_output_imag
            self.node_endlinear_real = OutputLinear(irreps_prev, irreps_out_real, intdtype=self.para.torch_int_dtype)
            self.edge_endlinear_real = OutputLinear(irreps_prev, irreps_out_real, intdtype=self.para.torch_int_dtype)
            self.node_endlinear_imag = OutputLinear(irreps_prev, irreps_out_imag, intdtype=self.para.torch_int_dtype)
            self.edge_endlinear_imag = OutputLinear(irreps_prev, irreps_out_imag, intdtype=self.para.torch_int_dtype)
        # if para.using_rearrangement_linear:
        #     self.node_endlinear = OutputLinear(irreps_prev, irreps_out)
        #     self.edge_endlinear = OutputLinear(irreps_prev, irreps_out)
        # else:
        #     self.node_endlinear = o3.Linear(irreps_prev, irreps_out)
        #     self.edge_endlinear = o3.Linear(irreps_prev, irreps_out)

    def _forward_for_equivariance_check(self, data: Data):
        AtomType = data.AtomType
        edge_length = data.d_hop
        edge_vector = data.D_hop[:, [1, 2, 0]]

        node_emb = self.node_emb(AtomType)
        length_emb = self.edge_emb(edge_length)
        edge_sh = self.edge_sh_emb(edge_vector)

        f_node, f_edge = self.FeatureInitialize(node_emb=node_emb, length_emb=length_emb, edge_sh_emb=edge_sh, data=data)
        f_node, f_edge = self.messagepassing(f_node=f_node, f_edge=f_edge, edge_sh=edge_sh, node_emb=node_emb, length_emb=length_emb, data=data)
        if not self.spin:
            f_node = self.node_endlinear(f_node)
            f_edge = self.edge_endlinear(f_edge)
        else:
            f_node_real = self.node_endlinear_real(f_node)
            f_node_imag = self.node_endlinear_imag(f_node)
            f_edge_real = self.edge_endlinear_real(f_edge)
            f_edge_imag = self.edge_endlinear_imag(f_edge)
            f_node = f_node_real + 1.0j * f_node_imag
            f_edge = f_edge_real + 1.0j * f_edge_imag
        return f_node, f_edge

    def _get_jd_for_equivariance_check(self, dtype: torch.dtype, device: torch.device):
        if not hasattr(self, "_Jd_equivariance_check"):
            import hotham

            self._Jd_equivariance_check = torch.load(os.path.join(hotham.__path__[0], "utilities/Jd.pt"))
        return [J.to(dtype=dtype, device=device) for J in self._Jd_equivariance_check]

    def _z_rot_mat_for_equivariance_check(self, angle: torch.Tensor, l: int):
        shape, device, dtype = angle.shape, angle.device, angle.dtype
        M = angle.new_zeros((*shape, 2 * l + 1, 2 * l + 1))
        inds = torch.arange(0, 2 * l + 1, 1, dtype=self.para.torch_int_dtype, device=device)
        reversed_inds = torch.arange(2 * l, -1, -1, dtype=self.para.torch_int_dtype, device=device)
        frequencies = torch.arange(l, -l - 1, -1, dtype=dtype, device=device)
        M[..., inds, reversed_inds] = torch.sin(frequencies * angle[..., None])
        M[..., inds, inds] = torch.cos(frequencies * angle[..., None])
        return M

    def _wigner_D_for_equivariance_check(self, l: int, alpha: torch.Tensor, beta: torch.Tensor, gamma: torch.Tensor, Jd):
        alpha, beta, gamma = torch.broadcast_tensors(alpha, beta, gamma)
        J = Jd[l]
        Xa = self._z_rot_mat_for_equivariance_check(alpha, l)
        Xb = self._z_rot_mat_for_equivariance_check(beta, l)
        Xc = self._z_rot_mat_for_equivariance_check(gamma, l)
        return Xa @ J @ Xb @ J @ Xc

    def _get_wigner_Ds_for_equivariance_check(self, lmax: int, edge_vec: torch.Tensor, mask_sc=None):
        Jd = self._get_jd_for_equivariance_check(edge_vec.dtype, edge_vec.device)
        alpha, beta = o3.xyz_to_angles(edge_vec)
        wigner_D = []
        for l in range(lmax + 1):
            D = self._wigner_D_for_equivariance_check(l, alpha, beta, torch.zeros_like(alpha), Jd)
            if mask_sc is not None:
                D[mask_sc] = torch.eye(2 * l + 1, dtype=D.dtype, device=D.device).unsqueeze(0)
            wigner_D.append(D)
        return wigner_D

    def _transform_magmom_for_equivariance_check(self, magmom: torch.Tensor, matrix: torch.Tensor = None, time_reversal: bool = False):
        magmom_transformed = magmom.clone()
        magmom_vec = torch.stack(
            [
                torch.sin(magmom[:, 2] * torch.pi / 180) * torch.sin(magmom[:, 3] * torch.pi / 180),
                torch.cos(magmom[:, 2] * torch.pi / 180),
                torch.sin(magmom[:, 2] * torch.pi / 180) * torch.cos(magmom[:, 3] * torch.pi / 180),
            ],
            dim=-1,
        )
        if matrix is not None:
            matrix = matrix.to(dtype=magmom_vec.dtype, device=magmom_vec.device)
            det = torch.det(matrix).sign()
            magmom_vec = det * torch.einsum("ij,zj->zi", matrix, magmom_vec)
        if time_reversal:
            magmom_vec = -magmom_vec
        magmom_vec = magmom_vec / magmom_vec.norm(dim=-1, keepdim=True).clamp_min(1.0e-12)
        magmom_transformed[:, 2] = torch.acos(magmom_vec[:, 1].clamp(-1.0, 1.0)) * 180 / torch.pi
        magmom_transformed[:, 3] = torch.atan2(magmom_vec[:, 0], magmom_vec[:, 2]) * 180 / torch.pi % 360.0
        return magmom_transformed

    def _transform_data_for_equivariance_check(self, data: Data, matrix: torch.Tensor = None, time_reversal: bool = False):
        data_transformed = data.clone()
        if matrix is not None:
            edge_vector = data.D_hop[:, [1, 2, 0]]
            matrix = matrix.to(dtype=edge_vector.dtype, device=edge_vector.device)
            edge_vector_transformed = torch.einsum("ij,zj->zi", matrix, edge_vector)
            data_transformed.D_hop = edge_vector_transformed[:, [2, 0, 1]]
            if hasattr(data, "pos"):
                pos = data.pos.to(dtype=matrix.dtype, device=matrix.device)
                pos_transformed = torch.einsum("ij,zj->zi", matrix, pos[:, [1, 2, 0]])
                data_transformed.pos = pos_transformed[:, [2, 0, 1]].to(dtype=data.pos.dtype, device=data.pos.device)
            if hasattr(data, "lattice"):
                lattice = data.lattice.to(dtype=matrix.dtype, device=matrix.device)
                lattice_transformed = torch.einsum("ij,zj->zi", matrix, lattice[:, [1, 2, 0]])
                data_transformed.lattice = lattice_transformed[:, [2, 0, 1]].to(dtype=data.lattice.dtype, device=data.lattice.device)
            if hasattr(data, "wigner_D"):
                mask_sc = data.mask_sc if self.para.edge_include_sc and hasattr(data, "mask_sc") else None
                data_transformed.wigner_D = self._get_wigner_Ds_for_equivariance_check(
                    len(data.wigner_D) - 1, edge_vector_transformed, mask_sc=mask_sc
                )
        if hasattr(data, "magmom") and (matrix is not None or time_reversal):
            data_transformed.magmom = self._transform_magmom_for_equivariance_check(data.magmom, matrix=matrix, time_reversal=time_reversal)
        return data_transformed

    def _time_reversal_D_for_equivariance_check(self, irreps, dtype, device):
        if not isinstance(irreps, TO3_Irreps):
            return torch.eye(irreps.dim, dtype=dtype, device=device)
        D_blocks = []
        for mul, ir in irreps:
            for _ in range(mul):
                D_blocks.append(torch.eye(ir.dim, dtype=dtype, device=device) * ir.t_p)
        return torch.block_diag(*D_blocks) if D_blocks else torch.empty(0, 0, dtype=dtype, device=device)

    def _apply_irreps_transform_for_equivariance_check(self, x: torch.Tensor, irreps, matrix: torch.Tensor = None, time_reversal: bool = False):
        if time_reversal and torch.is_complex(x):
            x = x.conj()
        D = torch.eye(irreps.dim, dtype=x.dtype, device=x.device)
        if matrix is not None:
            matrix_dtype = x.real.dtype if torch.is_complex(x) else x.dtype
            D = irreps.D_from_matrix(matrix.to(dtype=matrix_dtype, device="cpu")).to(dtype=x.dtype, device=x.device)
        if time_reversal:
            D_time = self._time_reversal_D_for_equivariance_check(irreps, dtype=x.dtype, device=x.device)
            D = torch.matmul(D_time, D)
        return torch.einsum("ij,zj->zi", D, x)

    def _equivariance_error(self, expected: torch.Tensor, actual: torch.Tensor, eps=1.0e-12):
        delta = (expected - actual).abs().max()
        scale = torch.maximum(expected.abs().max(), actual.abs().max()).clamp_min(eps)
        return delta, delta / scale

    @torch.no_grad()
    def check_equivariance(self, data: Data, f_node=None, f_edge=None, checks=("rotation", "inversion", "time_reversal"), verbose=True):
        if f_node is None or f_edge is None:
            f_node, f_edge = self._forward_for_equivariance_check(data)

        transforms = {}
        if "rotation" in checks:
            transforms["rotation"] = {
                "matrix": o3.rand_matrix(1).squeeze().to(dtype=data.D_hop.dtype, device=data.D_hop.device),
                "time_reversal": False,
            }
        if "inversion" in checks:
            transforms["inversion"] = {
                "matrix": -torch.eye(3, dtype=data.D_hop.dtype, device=data.D_hop.device),
                "time_reversal": False,
            }
        if "time_reversal" in checks:
            transforms["time_reversal"] = {"matrix": None, "time_reversal": True}

        results = {}
        for name, transform in transforms.items():
            data_transformed = self._transform_data_for_equivariance_check(data, **transform)
            f_node_transformed, f_edge_transformed = self._forward_for_equivariance_check(data_transformed)
            if not self.spin:
                node_irreps = self.node_endlinear.irreps_out
                edge_irreps = self.edge_endlinear.irreps_out
            else:
                node_irreps = self.node_endlinear_real.irreps_out
                edge_irreps = self.edge_endlinear_real.irreps_out
            f_node_expected = self._apply_irreps_transform_for_equivariance_check(f_node, node_irreps, **transform)
            f_edge_expected = self._apply_irreps_transform_for_equivariance_check(f_edge, edge_irreps, **transform)

            node_delta, node_ratio = self._equivariance_error(f_node_expected, f_node_transformed)
            edge_delta, edge_ratio = self._equivariance_error(f_edge_expected, f_edge_transformed)
            results[name] = {
                "node_delta": node_delta.item(),
                "edge_delta": edge_delta.item(),
                "node_ratio": node_ratio.item(),
                "edge_ratio": edge_ratio.item(),
            }
            if verbose:
                print(
                    f"{name}: node_delta={results[name]['node_delta']:.6e}, "
                    f"edge_delta={results[name]['edge_delta']:.6e}, "
                    f"node_ratio={results[name]['node_ratio']:.6e}, "
                    f"edge_ratio={results[name]['edge_ratio']:.6e}"
                )
        return results

    def forward(self, data: Data):
        # Embed "atom type", "edge length" and "direction"
        AtomType = data.AtomType
        edge_src, edge_dst = data.edge_index_hop
        edge_length = data.d_hop
        edge_vector = data.D_hop[:, [1, 2, 0]]

        node_emb = self.node_emb(AtomType)
        length_emb = self.edge_emb(edge_length)
        edge_sh = self.edge_sh_emb(edge_vector)

        f_node, f_edge = self.FeatureInitialize(node_emb=node_emb, length_emb=length_emb, edge_sh_emb=edge_sh, data=data)
        f_node, f_edge = self.messagepassing(f_node=f_node, f_edge=f_edge, edge_sh=edge_sh, node_emb=node_emb, length_emb=length_emb, data=data)

        # linear map to Hamilton's irreps
        if not self.spin:
            f_node = self.node_endlinear(f_node)
            f_edge = self.edge_endlinear(f_edge)
        else:
            f_node_real = self.node_endlinear_real(f_node)
            f_node_imag = self.node_endlinear_imag(f_node)
            f_edge_real = self.edge_endlinear_real(f_edge)
            f_edge_imag = self.edge_endlinear_imag(f_edge)

            f_node = f_node_real + 1.0j * f_node_imag
            f_edge = f_edge_real + 1.0j * f_edge_imag

        # l3 -> l1xl2
        DirectProduct = self.DirectSum_to_DirectProduct(f_node, f_edge, data)
        H_block = self.find_Hblock(DirectProduct, data)
        return H_block


if __name__ == "__main__":
    pass
