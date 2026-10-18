import torch
from torch.nn import functional as F
from e3nn import o3, math


class NodeEmbedding(torch.nn.Module):
    def __init__(self, num_type: int):
        super().__init__()
        self.num_type = num_type
        self.linear = o3.Linear(o3.Irreps(f"{num_type}x0e"), o3.Irreps(f"{num_type}x0e"))
        self.out_dim = num_type

    def forward(self, type_array):
        type_one_hot = F.one_hot(type_array, num_classes=self.num_type).to(self.linear.weight.dtype)
        embedding = self.linear(type_one_hot)
        return embedding


# class EdgeEmbedding(torch.nn.Module):
#     def __init__(self, basis_size: int, rc: int, out_feature: int):
#         super().__init__()
#         self.basis_size = basis_size
#         self.out_feature = out_feature
#         self.lenght_emb = Chebyshev(rc, basis_size)
#         self.out_dim = basis_size

#     def forward(self, lenght: torch.tensor):
#         lenght_emb = self.lenght_emb(lenght)
#         return lenght_emb


class EdgeEmbedding(torch.nn.Module):
    def __init__(self, basis_size: int, rc, out_feature: int = None,
                 basis: str = "Gaussian"):
        super().__init__()
        self.basis_size = basis_size
        self.out_feature = out_feature
        if basis == "Gaussian":
            self.lenght_emb = Gaussian(rc, basis_size)
        elif basis == "Chebyshev":
            self.lenght_emb = Chebyshev(rc, basis_size)
        else:
            raise ValueError(f"Unknown basis in para: {basis}")
        self.out_dim = basis_size

    def forward(self, lenght: torch.tensor):
        lenght_emb = self.lenght_emb(lenght)
        return lenght_emb


# class Chebyshev(torch.nn.Module):
#     def __init__(self, rc, basis_size):
#         super().__init__()
#         self.rc = rc
#         self.rcinv = 1.0/rc
#         self.basis_size = basis_size

#     def forward(self, d12: torch.tensor):
#         fc = 0.5*torch.cos((3.1415927*self.rcinv)*d12)+0.5
#         fn = d12.new_zeros((len(fc), self.basis_size))
#         x = 2.0*(d12*self.rcinv-1)**2-1.0
#         fn[:, 0] = 1.0
#         fn[:, 1] = x
#         for m in range(2, self.basis_size):
#             fn[:, m] = 2.0*x*fn[:, m-1]-fn[:, m-2]
#         return (fn+1)*0.5*fc.reshape([-1, 1])


class Chebyshev(torch.nn.Module):
    def __init__(self, rc, basis_size):
        super().__init__()
        if isinstance(rc, float):
            self.rc = rc
            self.rc0 = 0
        elif isinstance(rc, list):
            self.rc = rc[1]
            self.rc0 = rc[0]
        self.basis_size = basis_size

    def scale(self, x):
        return (x - self.rc0) / (self.rc - self.rc0)

    def forward(self, d12: torch.tensor):
        d12_scaled = self.scale(d12)
        fc = 0.5 * torch.cos(3.1415927 * d12_scaled) + 0.5
        fn = d12.new_zeros((len(fc), self.basis_size))
        x = 2.0 * (d12_scaled - 1) ** 2 - 1.0
        fn[:, 0] = 1.0
        fn[:, 1] = x
        for m in range(2, self.basis_size):
            fn[:, m] = 2.0 * x * fn[:, m - 1] - fn[:, m - 2]
        return (fn + 1) * 0.5 * fc.reshape([-1, 1])


class Gaussian(torch.nn.Module):
    def __init__(self, rc, basis_size):
        super().__init__()
        if isinstance(rc, float):
            end = rc
            start = 0
        elif isinstance(rc, list):
            end = rc[1]
            start = rc[0]

        self.start = start
        self.end = end
        self.number = basis_size

    def forward(self, x):
        return math.soft_one_hot_linspace(x=x, start=self.start, end=self.end, number=self.number, basis="gaussian", cutoff=False)
