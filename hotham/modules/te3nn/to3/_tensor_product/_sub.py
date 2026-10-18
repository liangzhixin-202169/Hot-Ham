from typing import Iterator, Optional

import torch
from e3nn.o3._irreps import Irrep, Irreps
from e3nn.util import prod
from e3nn.o3 import TensorProduct
from .. import TO3_Irrep, TO3_Irreps


class TO3_FullyConnectedTensorProduct(TensorProduct):

    def __init__(self, irreps_in1, irreps_in2, irreps_out, irrep_normalization: str = None, path_normalization: str = None, **kwargs) -> None:
        irreps_in1 = TO3_Irreps(irreps_in1)
        irreps_in2 = TO3_Irreps(irreps_in2)
        irreps_out = TO3_Irreps(irreps_out)

        self.to3_irreps_in1 = irreps_in1
        self.to3_irreps_in2 = irreps_in2
        self.to3_irreps_out = irreps_out

        instr = [
            (i_1, i_2, i_out, "uvw", True, 1.0)
            for i_1, (_, ir_1) in enumerate(irreps_in1)
            for i_2, (_, ir_2) in enumerate(irreps_in2)
            for i_out, (_, ir_out) in enumerate(irreps_out)
            if ir_out in ir_1 * ir_2
        ]
        super().__init__(
            irreps_in1.get_o3_irreps(),
            irreps_in2.get_o3_irreps(),
            irreps_out.get_o3_irreps(),
            instr,
            irrep_normalization=irrep_normalization,
            path_normalization=path_normalization,
            **kwargs,
        )

    def __repr__(self) -> str:
        npath = sum(prod(i.path_shape) for i in self.instructions)
        return (
            f"{self.__class__.__name__}"
            f"({self.to3_irreps_in1.simplify()} x {self.to3_irreps_in2.simplify()} "
            f"-> {self.to3_irreps_out.simplify()} | {npath} paths | {self.weight_numel} weights)"
        )


class TO3_ElementwiseTensorProduct(TensorProduct):

    def __init__(self, irreps_in1, irreps_in2, filter_ir_out=None, irrep_normalization: str = None, **kwargs) -> None:
        irreps_in1 = TO3_Irreps(irreps_in1).simplify()
        irreps_in2 = TO3_Irreps(irreps_in2).simplify()
        if filter_ir_out is not None:
            try:
                filter_ir_out = [TO3_Irrep(ir) for ir in filter_ir_out]
            except ValueError:
                raise ValueError(f"filter_ir_out (={filter_ir_out}) must be an iterable of TO3_Irreps")

        assert irreps_in1.num_irreps == irreps_in2.num_irreps

        irreps_in1 = list(irreps_in1)
        irreps_in2 = list(irreps_in2)

        i = 0
        while i < len(irreps_in1):
            mul_1, ir_1 = irreps_in1[i]
            mul_2, ir_2 = irreps_in2[i]

            if mul_1 < mul_2:
                irreps_in2[i] = (mul_1, ir_2)
                irreps_in2.insert(i + 1, (mul_2 - mul_1, ir_2))

            if mul_2 < mul_1:
                irreps_in1[i] = (mul_2, ir_1)
                irreps_in1.insert(i + 1, (mul_1 - mul_2, ir_1))
            i += 1

        out = []
        instr = []
        for i, ((mul, ir_1), (mul_2, ir_2)) in enumerate(zip(irreps_in1, irreps_in2)):
            assert mul == mul_2
            for ir in ir_1 * ir_2:
                if filter_ir_out is not None and ir not in filter_ir_out:
                    continue

                i_out = len(out)
                out.append((mul, ir))
                instr += [(i, i, i_out, "uuu", False)]

        o3_irreps_in1 = Irreps([(mul, ir.get_o3_irrep()) for mul, ir in irreps_in1])
        o3_irreps_in2 = Irreps([(mul, ir.get_o3_irrep()) for mul, ir in irreps_in2])
        o3_irreps_out = Irreps([(mul, ir.get_o3_irrep()) for mul, ir in out])

        self.to3_irreps_in1 = TO3_Irreps(irreps_in1)
        self.to3_irreps_in2 = TO3_Irreps(irreps_in2)
        self.to3_irreps_out = TO3_Irreps(out)

        super().__init__(o3_irreps_in1, o3_irreps_in2, o3_irreps_out, instr, irrep_normalization=irrep_normalization, **kwargs)

    def __repr__(self) -> str:
        npath = sum(prod(i.path_shape) for i in self.instructions)
        return (
            f"{self.__class__.__name__}"
            f"({self.to3_irreps_in1.simplify()} x {self.to3_irreps_in2.simplify()} "
            f"-> {self.to3_irreps_out.simplify()} | {npath} paths | {self.weight_numel} weights)"
        )
