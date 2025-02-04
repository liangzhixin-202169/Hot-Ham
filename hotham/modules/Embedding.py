import torch
from torch.nn import functional as F
from e3nn import o3


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


class EdgeEmbedding(torch.nn.Module):
    def __init__(self, basis_size: int, rc: int, out_feature: int):
        super().__init__()
        self.basis_size = basis_size
        self.out_feature = out_feature
        self.lenght_emb = Chebyshev(rc, basis_size)
        self.out_dim = basis_size

    def forward(self, lenght: torch.tensor):
        lenght_emb = self.lenght_emb(lenght)
        return lenght_emb


class Chebyshev(torch.nn.Module):
    def __init__(self, rc, basis_size):
        super().__init__()
        self.rc = rc
        self.rcinv = 1.0/rc
        self.basis_size = basis_size

    def forward(self, d12: torch.tensor):
        fc = 0.5*torch.cos((3.1415927*self.rcinv)*d12)+0.5
        fn = d12.new_zeros((len(fc), self.basis_size))
        x = 2.0*(d12*self.rcinv-1)**2-1.0
        fn[:, 0] = 1.0
        fn[:, 1] = x
        for m in range(2, self.basis_size):
            fn[:, m] = 2.0*x*fn[:, m-1]-fn[:, m-2]
        return (fn+1)*0.5*fc.reshape([-1, 1])
