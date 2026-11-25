"""Convert OpenMX H0 to HotHam's ABACUS-style Hamiltonian CSR."""
import re
from itertools import groupby
import numpy as np
from ase.units import Hartree, Rydberg
from scipy.sparse import coo_matrix

# Signed one-based OpenMX orbital indices in HotHam's CSR output order.
ORBITAL_ORDER = {
    "s": [1],
    "p": [3, 1, -2],
    "d": [3, -5, 1, -4, 2],
    "f": [4, -5, 3, -6, 2, -7, -1],
}


def read_hamiltonian(filename):
    channels = {}
    blocks = None
    with open(filename) as stream:
        for line in stream:
            if "Kohn-Sham Hamiltonian" in line:
                if "iHks" in line:
                    raise ValueError("This converter supports scalar and collinear Hamiltonians")
                blocks = channels.setdefault(int(line.split("=")[-1]), {})
            elif line.startswith("Block") and blocks is not None:
                i, j, rx, ry, rz, nr, nc = map(int, line.split()[1:])
                values = np.array([next(stream).split() for _ in range(nr)], dtype=float)
                blocks[rx, ry, rz, i - 1, j - 1] = values.reshape(nr, nc)
            elif line.strip():
                blocks = None
    return channels


def write_csr(filename, blocks, orders):
    offsets = np.cumsum([0] + [len(order) for order in orders])
    dim = int(offsets[-1])
    keys = sorted(blocks)
    with open(filename, "w") as stream:
        stream.write(f"STEP: 0\nMatrix Dimension of H(R): {dim}\n")
        stream.write(f"Matrix number of H(R): {len({key[:3] for key in keys})}\n")
        for shift, cell_keys in groupby(keys, key=lambda key: key[:3]):
            rows, cols, values = [], [], []
            for key in cell_keys:
                i, j = key[3:]
                a, b = orders[i], orders[j]
                block = blocks[key][np.ix_(abs(a) - 1, abs(b) - 1)]
                block *= np.sign(a)[:, None] * np.sign(b)
                row, col = np.nonzero(block)
                rows.extend(row + offsets[i])
                cols.extend(col + offsets[j])
                values.extend(block[row, col] * (Hartree / Rydberg))
            matrix = coo_matrix((values, (rows, cols)), shape=(dim, dim)).tocsr()
            stream.write(f"{shift[0]} {shift[1]} {shift[2]} {matrix.nnz}\n")
            if matrix.nnz:
                stream.write(" ".join(f"{value:.16e}" for value in matrix.data) + "\n")
                stream.write(" ".join(map(str, matrix.indices)) + "\n")
                stream.write(" ".join(map(str, matrix.indptr)) + "\n")


def main(inputfile):
    orders_by_size = {}
    for shells in inputfile["orbit"].values():
        order = []
        for shell in shells:
            indices = ORBITAL_ORDER[re.search(r"[spdf]", shell).group()]
            offset = len(order)
            order.extend((1 if i > 0 else -1) * (offset + abs(i)) for i in indices)
        orders_by_size.setdefault(len(order), set()).add(tuple(order))
    channels = read_hamiltonian(inputfile["h0.txt"])
    if set(channels) not in ({0}, {0, 1}) or not all(channels.values()):
        raise ValueError("Expected one or two nonempty OpenMX Hamiltonian channels")
    sizes = {}
    for (*_, i, j), block in channels[0].items():
        sizes[i], sizes[j] = block.shape
    atom_orders = []
    for atom in range(len(sizes)):
        size = sizes[atom]
        candidates = orders_by_size.get(size, ())
        if len(candidates) != 1:
            raise ValueError(f"Cannot determine a unique orbital order for atom {atom + 1} with {size} orbitals")
        atom_orders.append(np.asarray(next(iter(candidates))))
    for spin, blocks in sorted(channels.items()):
        write_csr(f"h_0_{spin}.csr", blocks, atom_orders)


if __name__ == "__main__":
    inputfile = {
        "h0.txt": "h0.txt",
        "orbit": {
            "C": ["1s", "2s", "2p", "3p", "3d"],
        },
    }
    main(inputfile)
