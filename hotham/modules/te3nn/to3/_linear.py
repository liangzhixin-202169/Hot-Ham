import torch
from e3nn.o3 import Linear as O3_Linear
from ._irreps import TO3_Irreps


class TO3_Linear(torch.nn.Module):
    def __init__(self, irreps_in: TO3_Irreps, irreps_out: TO3_Irreps, shared_weights=True):
        super().__init__()

        self.irreps_in = irreps_in
        self.irreps_out = irreps_out

        instructions = []
        for i, (mul1, (l1, p1, t_p1)) in enumerate(self.irreps_in):
            for j, (mul2, (l2, p2, t_p2)) in enumerate(self.irreps_out):
                if l1 == l2 and p1 == p2 and t_p1 == t_p2:
                    instructions.append((i, j))

        self.linear = O3_Linear(
            irreps_in=self.irreps_in.get_o3_irreps(),
            irreps_out=self.irreps_out.get_o3_irreps(),
            instructions=instructions,
            shared_weights=shared_weights,
        )

    def forward(self, x, weight=None):
        return self.linear(x, weight=weight)

    @property
    def weight_numel(self):
        return self.linear.weight_numel

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.irreps_in} -> {self.irreps_out} | {self.weight_numel} weights)"
