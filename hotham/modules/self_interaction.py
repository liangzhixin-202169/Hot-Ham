import torch
from e3nn import o3
from e3nn.o3 import Irreps
from .te3nn.to3 import TO3_Irreps, TO3_Linear
from ..modules.output_linear import OutputLinear
from .common import TO3LayerNormal


class TO3_GauntTensorProduct(torch.nn.Module):
    def __init__(self, irreps_in: TO3_Irreps, channel: int, symmetrized=True, para=None, **kwargs):
        super().__init__()
        assert isinstance(irreps_in, TO3_Irreps)
        self.irreps_in = irreps_in
        self.channel = channel
        self.symmetrized = symmetrized

        self.lmax = self.irreps_in.lmax
        self.dim_L = (self.lmax + 1) ** 2
        self.simplified_irreps_in = self.irreps_in.sort()[0].simplify()

        res_beta = 4 * (self.lmax + 1) if self.lmax % 2 == 1 else 4 * self.lmax
        res = (res_beta, res_beta - 1)
        self.tos2grid = o3.ToS2Grid(lmax=self.lmax, res=res)
        self.froms2grid = o3.FromS2Grid(res=res, lmax=self.lmax)

        if self.symmetrized:
            self.register_buffer("x_raw_idx", torch.tensor([0, 0, 0, 0, 1, 1, 1, 2, 2, 3], dtype=para.torch_int_dtype if para is not None else torch.int64))
            self.register_buffer("x_col_idx", torch.tensor([0, 1, 2, 3, 1, 2, 3, 2, 3, 3], dtype=para.torch_int_dtype if para is not None else torch.int64))
        else:
            self.register_buffer("x_raw_idx", torch.tensor([0, 1, 2, 3], dtype=para.torch_int_dtype if para is not None else torch.int64).repeat_interleave(4))
            self.register_buffer("x_col_idx", torch.tensor([0, 1, 2, 3], dtype=para.torch_int_dtype if para is not None else torch.int64).repeat(4))

        # order_strs = "ee,eo,oe,oo,ee,oo,oe,ee,eo,ee"
        parity_symbol = {-1: "o", 1: "e"}
        order_strs = []
        # ir1
        for p1 in [-1, 1]:
            for t_p1 in [-1, 1]:
                order_strs.append([])
                # ir2
                for p2 in [-1, 1]:
                    for t_p2 in [-1, 1]:
                        p3 = p1 * p2
                        t_p3 = t_p1 * t_p2
                        sub_order_strs = f"{parity_symbol[p3]}{parity_symbol[t_p3]}"
                        order_strs[-1].append(sub_order_strs)
        if self.symmetrized:
            for i in range(4):
                j = i
                while j:
                    order_strs[i].pop(0)
                    j -= 1

        order_strs = ",".join(sum(order_strs, start=[]))
        irreps_out = TO3_Irreps(None)
        for order_str in order_strs.split(","):
            irs_1_str = "+".join([f"1x{l}{order_str}" for l in range(self.lmax + 1)])
            irs_mul_str = "+".join([irs_1_str] * channel)
            irreps_out += TO3_Irreps(irs_mul_str)
        self.irreps_out = irreps_out

    def forward(self, x):
        """
        x = (oo,oe,eo,ee)
        x.T*x = (oo,oe,eo,ee).T * (oo,oe,eo,ee)
        ->                        idx                           mask
        ee,eo,oe,oo                0,  1,  2,  3,                true,  true,  true,  true,
        eo,ee,oo,oe                4,  5,  6,  7,               false,  true,  true,  true,
        oe,oo,ee,eo                8,  9, 10, 11,               false, false,  true,  true,
        oo,oe,eo,ee               12, 13, 14, 15                false, false, false,  true,

        by mask:
        y:             ee,eo,oe,oo,ee,oo,oe,ee,eo,ee
        complete idx:   0, 1, 2, 3, 5, 6, 7,10,11,15
        masked idx:     0, 1, 2, 3, 4, 5, 6, 7, 8, 9

        reorder
        y[0,4,7,9] = [ee,ee,ee,ee]
        y[1,8] = [eo,eo]
        y[2,6] = [oe,oe]
        y[3,5] = [oo,oo]

        To avoid compute the 4*4 matrix directly, broadcast x firstly
        x_raw = x[0,0,0,0,1,1,1,2,2,3]
        x_col = x[0,1,2,3,1,2,3,2,3,3]
        """
        n, _ = x.shape
        x = x.reshape(n, 4, self.channel, self.dim_L)
        x_grid = self.tos2grid(x)
        x_raw = x_grid[:, self.x_raw_idx]
        x_col = x_grid[:, self.x_col_idx]
        y_grid = x_raw * x_col
        y = self.froms2grid(y_grid)
        y = y.reshape(n, self.irreps_out.dim)
        return y


class Self_interaction(torch.nn.Module):
    def __init__(self, irreps_in: TO3_Irreps, irreps_out: TO3_Irreps, channel: int, symmetrized=True, para=None, **kwargs):
        super().__init__()
        self.irreps_in = irreps_in
        self.irreps_out = irreps_out
        self.lmax = max([irreps_in.lmax, irreps_out.lmax])
        self.symmetrized = symmetrized

        if isinstance(irreps_in, Irreps):
            irreps_class = Irreps
            linear_class = o3.Linear
        elif isinstance(irreps_in, TO3_Irreps):
            irreps_class = TO3_Irreps
            linear_class = TO3_Linear
        else:
            raise TypeError()
        assert isinstance(irreps_out, irreps_class)

        irreps_for_gaunt = TO3_Irreps(None)
        for p in ["o", "e"]:
            for p_t in ["o", "e"]:
                irs_1_str = "+".join([f"1x{l}{p}{p_t}" for l in range(self.lmax + 1)])
                irs_mul_str = "+".join([irs_1_str] * channel)
                irreps_for_gaunt += TO3_Irreps(irs_mul_str)
        self.irreps_for_gaunt = irreps_for_gaunt

        self.lin_prev = OutputLinear(irreps_in=self.irreps_in, irreps_out=self.irreps_for_gaunt, intdtype=para.torch_int_dtype if para is not None else torch.int64)
        self.to3_gtp = TO3_GauntTensorProduct(irreps_in=irreps_for_gaunt, symmetrized=self.symmetrized, channel=channel, para=para)
        self.lin_post = OutputLinear(irreps_in=self.to3_gtp.irreps_out, irreps_out=self.irreps_out, intdtype=para.torch_int_dtype if para is not None else torch.int64)
        self.layernormal = TO3LayerNormal(irreps_in=self.lin_post.irreps_out, floatdtype=para.torch_float_dtype if para is not None else torch.float32)

    def forward(self, x):
        out = self.lin_prev(x)
        out = self.to3_gtp(out)
        out = self.lin_post(out)
        out = self.layernormal(out)
        out = x + out
        return out
