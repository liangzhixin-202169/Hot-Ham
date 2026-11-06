import itertools
import collections
from typing import List, Union

import torch

from e3nn.math import direct_sum, perm

# These imports avoid cyclic reference from o3 itself
# from . import _rotation
# from . import _wigner
from e3nn.o3 import _rotation
from e3nn.o3 import _wigner
from e3nn.o3 import Irrep as O3Irrep
from e3nn.o3 import Irreps as O3Irreps


class TO3_Irrep(tuple):

    def __new__(cls, l: Union[int, "Irrep", str, tuple], p=None, t_p=None):
        if p is None and t_p is None:
            if isinstance(l, TO3_Irrep):
                return l

            if isinstance(l, str):
                try:
                    name = l.strip()
                    l = int(name[:-2])
                    assert l >= 0
                    p = {
                        "e": 1,
                        "o": -1,
                        "y": (-1) ** l,
                    }[name[-2]]
                    t_p = {"e": 1, "o": -1}[name[-1]]
                except Exception:
                    raise ValueError(f'unable to convert string "{name}" into an Irrep')
            elif isinstance(l, tuple):
                l, p, t_p = l

        if not isinstance(l, int) or l < 0:
            raise ValueError(f"l must be positive integer, got {l}")
        if p not in (-1, 1):
            raise ValueError(f"parity must be on of (-1, 1), got {p}")
        if t_p not in (-1, 1):
            raise ValueError(f"time parity must be on of (-1, 1), got {t_p}")
        return super().__new__(cls, (l, p, t_p))

    def get_o3_irrep(self):
        """
        Return the ordinary e3nn.o3 irreducible representation
        obtained by forgetting the extra time-reversal index.
        """
        return O3Irrep(self.l, self.p)

    @property
    def l(self) -> int:  # noqa: E743
        r"""The degree of the representation, :math:`l = 0, 1, \dots`."""
        return self[0]

    @property
    def p(self) -> int:
        r"""The parity of the representation, :math:`p = \pm 1`."""
        return self[1]

    @property
    def t_p(self) -> int:
        r"""The time parity of the representation, :math:`t_p = \pm 1`."""
        return self[2]

    def __repr__(self):
        p = {+1: "e", -1: "o"}[self.p]
        t_p = {+1: "e", -1: "o"}[self.t_p]
        return f"{self.l}{p}{t_p}"

    # @classmethod
    # def iterator(cls, lmax=None):
    #     r"""Iterator through all the irreps of :math:`O(3)`

    #     Examples
    #     --------
    #     >>> it = Irrep.iterator()
    #     >>> next(it), next(it), next(it), next(it)
    #     (0e, 0o, 1o, 1e)
    #     """
    #     for l in itertools.count():
    #         yield Irrep(l, (-1) ** l)
    #         yield Irrep(l, -((-1) ** l))

    #         if l == lmax:
    #             break

    def D_from_angles(self, alpha, beta, gamma, k=None):
        r"""Matrix :math:`p^k D^l(\alpha, \beta, \gamma)`

        (matrix) Representation of :math:`O(3)`. :math:`D` is the representation of :math:`SO(3)`, see `wigner_D`.

        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\alpha` around Y axis, applied third.

        beta : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\beta` around X axis, applied second.

        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`
            Rotation :math:`\gamma` around Y axis, applied first.

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`
            How many times the parity is applied.

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`

        See Also
        --------
        o3.wigner_D
        Irreps.D_from_angles
        """
        if k is None:
            k = torch.zeros_like(alpha)

        alpha, beta, gamma, k = torch.broadcast_tensors(alpha, beta, gamma, k)
        return _wigner.wigner_D(self.l, alpha, beta, gamma) * self.p ** k[..., None, None]

    def D_from_quaternion(self, q, k=None):
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        q : `torch.Tensor`
            tensor of shape :math:`(..., 4)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*_rotation.quaternion_to_angles(q), k)

    def D_from_matrix(self, R):
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`

        Examples
        --------
        >>> m = Irrep(1, -1).D_from_matrix(-torch.eye(3))
        >>> m.long()
        tensor([[-1,  0,  0],
                [ 0, -1,  0],
                [ 0,  0, -1]])
        """
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(*_rotation.matrix_to_angles(R), k)

    def D_from_axis_angle(self, axis, angle):
        r"""Matrix of the representation, see `Irrep.D_from_angles`

        Parameters
        ----------
        axis : `torch.Tensor`
            tensor of shape :math:`(..., 3)`

        angle : `torch.Tensor`
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., 2l+1, 2l+1)`
        """
        return self.D_from_angles(*_rotation.axis_angle_to_angles(axis, angle))

    @property
    def dim(self) -> int:
        """The dimension of the representation, :math:`2 l + 1`."""
        return 2 * self.l + 1

    def is_scalar(self) -> bool:
        """Equivalent to ``l == 0 and p == 1 and t_p ==1``"""
        return self.l == 0 and self.p == 1 and self.t_p == 1

    def __mul__(self, other):
        r"""Generate the irreps from the product of two irreps.

        Returns
        -------
        generator of `e3nn.o3.Irrep`
        """
        other = TO3_Irrep(other)
        p = self.p * other.p
        t_p = self.t_p * other.t_p
        lmin = abs(self.l - other.l)
        lmax = self.l + other.l
        for l in range(lmin, lmax + 1):
            yield TO3_Irrep(l, p, t_p)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError

    def __rmul__(self, other):
        r"""
        >>> 3 * Irrep('1eo')
        3x1eo
        """
        assert isinstance(other, int)
        return TO3_Irreps([(other, self)])

    def __add__(self, other):
        return TO3_Irreps(self) + TO3_Irreps(other)

    def __contains__(self, _object):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError


class _MulIr(tuple):
    def __new__(cls, mul, ir=None):
        if ir is None:
            mul, ir = mul

        assert isinstance(mul, int)
        assert isinstance(ir, TO3_Irrep)
        return super().__new__(cls, (mul, ir))

    @property
    def mul(self) -> int:
        return self[0]

    @property
    def ir(self) -> TO3_Irrep:
        return self[1]

    @property
    def dim(self) -> int:
        return self.mul * self.ir.dim

    def __repr__(self):
        return f"{self.mul}x{self.ir}"

    def __getitem__(self, item) -> Union[int, TO3_Irrep]:  # pylint: disable=useless-super-delegation
        return super().__getitem__(item)

    def count(self, _value):
        raise NotImplementedError

    def index(self, _value):
        raise NotImplementedError


class TO3_Irreps(tuple):

    def __new__(cls, irreps=None) -> Union[_MulIr, "TO3_Irreps"]:
        if isinstance(irreps, TO3_Irreps):
            return super().__new__(cls, irreps)

        out = []
        if isinstance(irreps, TO3_Irrep):
            out.append(_MulIr(1, TO3_Irrep(irreps)))
        elif isinstance(irreps, str):
            try:
                if irreps.strip() != "":
                    for mul_ir in irreps.split("+"):
                        if "x" in mul_ir:
                            mul, ir = mul_ir.split("x")
                            mul = int(mul)
                            ir = TO3_Irrep(ir)
                        else:
                            mul = 1
                            ir = TO3_Irrep(mul_ir)

                        assert isinstance(mul, int) and mul >= 0
                        out.append(_MulIr(mul, ir))
            except Exception:
                raise ValueError(f'Unable to convert string "{irreps}" into an TO3_Irreps')
        elif irreps is None:
            pass
        else:
            for mul_ir in irreps:
                mul = None
                ir = None

                if isinstance(mul_ir, str):
                    mul = 1
                    ir = TO3_Irrep(mul_ir)
                elif isinstance(mul_ir, TO3_Irrep):
                    mul = 1
                    ir = mul_ir
                elif isinstance(mul_ir, _MulIr):
                    mul, ir = mul_ir
                elif len(mul_ir) == 2:
                    mul, ir = mul_ir
                    ir = TO3_Irrep(ir)

                if not (isinstance(mul, int) and mul >= 0 and ir is not None):
                    raise ValueError(f'Unable to interpret "{mul_ir}" as an to3_irrep.')

                out.append(_MulIr(mul, ir))
        return super().__new__(cls, out)

    def get_o3_irreps(self):
        """
        Return the ordinary e3nn.o3 irreps obtained by forgetting the
        extra time-reversal index, with multiplicities merged.
        """
        return O3Irreps([(mul, ir.get_o3_irrep()) for mul, ir in self])

    @staticmethod
    def spherical_harmonics(lmax, p=-1, t_p=1):
        return TO3_Irreps([(1, (l, p**l, t_p**l)) for l in range(lmax + 1)])

    def slices(self):
        s = []
        i = 0
        for mul_ir in self:
            s.append(slice(i, i + mul_ir.dim))
            i += mul_ir.dim
        return s

    def randn(self, *size, normalization="component", requires_grad=False, dtype=None, device=None):
        r"""Random tensor.

        Parameters
        ----------
        *size : list of int
            size of the output tensor, needs to contains a ``-1``

        normalization : {'component', 'norm'}

        Returns
        -------
        `torch.Tensor`
            tensor of shape ``size`` where ``-1`` is replaced by ``self.dim``

        Examples
        --------

        >>> Irreps("5x0e + 10x1o").randn(5, -1, 5, normalization='norm').shape
        torch.Size([5, 35, 5])

        >>> random_tensor = Irreps("2o").randn(2, -1, 3, normalization='norm')
        >>> random_tensor.norm(dim=1).sub(1).abs().max().item() < 1e-5
        True
        """
        di = size.index(-1)
        lsize = size[:di]
        rsize = size[di + 1 :]

        if normalization == "component":
            return torch.randn(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
        elif normalization == "norm":
            x = torch.zeros(*lsize, self.dim, *rsize, requires_grad=requires_grad, dtype=dtype, device=device)
            with torch.no_grad():
                for s, (mul, ir) in zip(self.slices(), self):
                    r = torch.randn(*lsize, mul, ir.dim, *rsize, dtype=dtype, device=device)
                    r.div_(r.norm(2, dim=di + 1, keepdim=True))
                    x.narrow(di, s.start, mul * ir.dim).copy_(r.reshape(*lsize, -1, *rsize))
            return x
        else:
            raise ValueError("Normalization needs to be 'norm' or 'component'")

    def __getitem__(self, i) -> Union[_MulIr, "TO3_Irreps"]:
        x = super().__getitem__(i)
        if isinstance(i, slice):
            return TO3_Irreps(x)
        return x

    def __contains__(self, ir) -> bool:
        ir = TO3_Irrep(ir)
        return ir in (irrep for _, irrep in self)

    def count(self, ir) -> int:
        r"""Multiplicity of ``ir``.

        Parameters
        ----------
        ir : `e3nn.o3.Irrep`

        Returns
        -------
        `int`
            total multiplicity of ``ir``
        """
        ir = TO3_Irreps(ir)
        return sum(mul for mul, irrep in self if ir == irrep)

    def index(self, _object):
        raise NotImplementedError

    def __add__(self, irreps):
        irreps = TO3_Irreps(irreps)
        return TO3_Irreps(super().__add__(irreps))

    def __mul__(self, other):
        r"""
        >>> (Irreps('2x1ee') * 3).simplify()
        6x1ee
        """
        if isinstance(other, Irreps):
            raise NotImplementedError("Use o3.TensorProduct for this, see the documentation")
        return TO3_Irreps(super().__mul__(other))

    def __rmul__(self, other):
        r"""
        >>> 2 * Irreps('0e + 1e')
        1x0e+1x1e+1x0e+1x1e
        """
        return TO3_Irreps(super().__rmul__(other))

    def simplify(self) -> "Irreps":
        """Simplify the representations.

        Returns
        -------
        `e3nn.o3.Irreps`

        Examples
        --------

        Note that simplify does not sort the representations.

        >>> Irreps("1e + 1e + 0e").simplify()
        2x1e+1x0e

        Equivalent representations which are separated from each other are not combined.

        >>> Irreps("1e + 1e + 0e + 1e").simplify()
        2x1e+1x0e+1x1e
        """
        out = []
        for mul, ir in self:
            if out and out[-1][1] == ir:
                out[-1] = (out[-1][0] + mul, ir)
            elif mul > 0:
                out.append((mul, ir))
        return TO3_Irreps(out)

    def remove_zero_multiplicities(self):
        """Remove any irreps with multiplicities of zero.

        Returns
        -------
        `e3nn.o3.Irreps`

        Examples
        --------

        >>> Irreps("4x0e + 0x1o + 2x3e").remove_zero_multiplicities()
        4x0e+2x3e

        """
        out = [(mul, ir) for mul, ir in self if mul > 0]
        return TO3_Irreps(out)

    def sort(self):
        r"""Sort the representations.

        Returns
        -------
        irreps : `e3nn.o3.Irreps`
        p : tuple of int
        inv : tuple of int

        Examples
        --------

        >>> Irreps("1e + 0e + 1e").sort().irreps
        1x0e+1x1e+1x1e

        >>> Irreps("2o + 1e + 0e + 1e").sort().p
        (3, 1, 0, 2)

        >>> Irreps("2o + 1e + 0e + 1e").sort().inv
        (2, 1, 3, 0)
        """
        Ret = collections.namedtuple("sort", ["irreps", "p", "t_p", "inv"])
        out = [(ir, i, mul) for i, (mul, ir) in enumerate(self)]
        out = sorted(out)
        inv = tuple(i for _, i, _ in out)
        p = perm.inverse(inv)
        t_p = perm.inverse(inv)
        irreps = TO3_Irreps([(mul, ir) for ir, _, mul in out])
        return Ret(irreps, p, t_p, inv)

    @property
    def dim(self) -> int:
        return sum(mul * ir.dim for mul, ir in self)

    @property
    def num_irreps(self) -> int:
        return sum(mul for mul, _ in self)

    @property
    def ls(self) -> List[int]:
        return [l for mul, (l, p, t_p) in self for _ in range(mul)]

    @property
    def lmax(self) -> int:
        if len(self) == 0:
            raise ValueError("Cannot get lmax of empty Irreps")
        return max(self.ls)

    def __repr__(self):
        return "+".join(f"{mul_ir}" for mul_ir in self)

    def D_from_angles(self, alpha, beta, gamma, k=None):
        r"""Matrix of the representation

        Parameters
        ----------
        alpha : `torch.Tensor`
            tensor of shape :math:`(...)`

        beta : `torch.Tensor`
            tensor of shape :math:`(...)`

        gamma : `torch.Tensor`
            tensor of shape :math:`(...)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return direct_sum(*[ir.D_from_angles(alpha, beta, gamma, k) for mul, ir in self for _ in range(mul)])

    def D_from_quaternion(self, q, k=None):
        r"""Matrix of the representation

        Parameters
        ----------
        q : `torch.Tensor`
            tensor of shape :math:`(..., 4)`

        k : `torch.Tensor`, optional
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*_rotation.quaternion_to_angles(q), k)

    def D_from_matrix(self, R):
        r"""Matrix of the representation

        Parameters
        ----------
        R : `torch.Tensor`
            tensor of shape :math:`(..., 3, 3)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        d = torch.det(R).sign()
        R = d[..., None, None] * R
        k = (1 - d) / 2
        return self.D_from_angles(*_rotation.matrix_to_angles(R), k)

    def D_from_axis_angle(self, axis, angle):
        r"""Matrix of the representation

        Parameters
        ----------
        axis : `torch.Tensor`
            tensor of shape :math:`(..., 3)`

        angle : `torch.Tensor`
            tensor of shape :math:`(...)`

        Returns
        -------
        `torch.Tensor`
            tensor of shape :math:`(..., \mathrm{dim}, \mathrm{dim})`
        """
        return self.D_from_angles(*_rotation.axis_angle_to_angles(axis, angle))


if __name__ == "__main__":
    # TO3_Irrep test
    # irrep = (0, 1, -1)
    # irrep = "1ee"
    # irrep = TO3_Irrep(irrep)
    # print(irrep.l, irrep.p, irrep.t_p)
    # print(irrep)
    # print(irrep.dim, irrep.is_scalar())
    # print(irrep.get_o3_irrep())

    # TO3_Irreps test
    # irreps = "1x0ee+2x1oe+3x2ee+4x0ee"
    irreps = [(1, (0, 1, 1)), (2, (1, -1, 1)), (3, (2, 1, 1)), (4, (0, 1, 1))]
    to3_irreps = TO3_Irreps(irreps)
    print(to3_irreps)
    print(to3_irreps[-1])
    print(to3_irreps[0:1])
    print(to3_irreps + to3_irreps)
    print(len(to3_irreps))
    print(to3_irreps.sort())
    print(to3_irreps.sort()[0].simplify())
    print(to3_irreps.dim, to3_irreps.num_irreps, to3_irreps.ls, to3_irreps.lmax)
    print(to3_irreps.get_o3_irreps())
    print(TO3_Irreps(TO3_Irrep("2ee")))
    print(TO3_Irrep("2ee") in to3_irreps)
