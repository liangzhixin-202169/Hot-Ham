import sys
import numpy as np
from pathlib import Path
from ase.io import read, write
from ase.neighborlist import neighbor_list
from ase.geometry import get_distances
from scipy.optimize import linear_sum_assignment


def cluster_layers(atoms, tol=1.5):
    z = atoms.positions[:, 2]
    order = np.argsort(z)
    layers, current = [], [order[0]]
    for idx in order[1:]:
        if z[idx] - z[current[-1]] < tol:
            current.append(idx)
        else:
            layers.append(current)
            current = [idx]
    layers.append(current)
    layers.sort(key=lambda g: z[g].mean())
    return layers


def bfs_bipartite(n, adj):
    color = np.full(n, -1)
    for start in range(n):
        if color[start] != -1:
            continue
        color[start] = 0
        queue = [start]
        while queue:
            node = queue.pop(0)
            for nb in adj[node]:
                if color[nb] == -1:
                    color[nb] = 1 - color[node]
                    queue.append(nb)
    color[color == -1] = 0
    return color


def pair_by_hungarian(layer_atoms, idx_a, idx_b):
    pos_a = layer_atoms.positions[idx_a]
    pos_b = layer_atoms.positions[idx_b]
    _, D = get_distances(pos_a, pos_b, cell=layer_atoms.cell, pbc=layer_atoms.pbc)
    row, col = linear_sum_assignment(D)
    pairs = [(idx_a[r], idx_b[c]) for r, c in zip(row, col)]
    used_a = set(row)
    used_b = set(col)
    leftover = [idx_a[i] for i in range(len(idx_a)) if i not in used_a] + \
               [idx_b[i] for i in range(len(idx_b)) if i not in used_b]
    return pairs, leftover


def sort_pairs_by_center(pairs, layer_atoms):
    def center_xy(p):
        pa = layer_atoms.positions[p[0], :2]
        pb = layer_atoms.positions[p[1], :2]
        return tuple((pa + pb) / 2)
    return sorted(pairs, key=center_xy)


def sort_graphene_layer(layer_atoms, orig_idx):
    i_arr, j_arr = neighbor_list('ij', layer_atoms, cutoff=1.9)
    n = len(layer_atoms)
    adj = [[] for _ in range(n)]
    for i, j in zip(i_arr, j_arr):
        adj[i].append(j)
    color = bfs_bipartite(n, adj)
    idx_a = np.where(color == 0)[0]
    idx_b = np.where(color == 1)[0]
    pairs, leftover = pair_by_hungarian(layer_atoms, idx_a, idx_b)
    pairs = sort_pairs_by_center(pairs, layer_atoms)
    result = []
    for a, b in pairs:
        result.append(orig_idx[a])
        result.append(orig_idx[b])
    result += [orig_idx[i] for i in leftover]
    return result


def sort_bn_layer(layer_atoms, orig_idx):
    syms = np.array(layer_atoms.get_chemical_symbols())
    idx_b = np.where(syms == 'B')[0]
    idx_n = np.where(syms == 'N')[0]
    pairs, leftover = pair_by_hungarian(layer_atoms, idx_b, idx_n)
    pairs = sort_pairs_by_center(pairs, layer_atoms)
    result = []
    for b, n in pairs:
        result.append(orig_idx[b])
        result.append(orig_idx[n])
    result += [orig_idx[i] for i in leftover]
    return result


def sort_heterostructure(input_path, output_path=None, layer_tol=2):
    atoms = read(input_path)
    layers = cluster_layers(atoms, tol=layer_tol)
    print(f"读入 {len(atoms)} 个原子，识别到 {len(layers)} 层")

    sorted_indices = []
    for li, layer_idx in enumerate(layers):
        layer_idx_arr = np.array(layer_idx)
        layer_atoms = atoms[layer_idx_arr]
        syms = set(layer_atoms.get_chemical_symbols())
        z_avg = layer_atoms.positions[:, 2].mean()

        if syms <= {'C'}:
            kind = 'graphene'
            ordered = sort_graphene_layer(layer_atoms, layer_idx_arr)
        elif syms <= {'B', 'N'}:
            kind = 'bn'
            ordered = sort_bn_layer(layer_atoms, layer_idx_arr)
        else:
            kind = 'unknown'
            ordered = list(layer_idx_arr)

        print(f"  层{li+1:2d}  z={z_avg:.4f} Å  {len(layer_idx):3d} 原子  [{kind}]")
        sorted_indices.extend(ordered)

    new_atoms = atoms[sorted_indices]

    if output_path is None:
        p = Path(input_path)
        output_path = str(p.parent / (p.stem + '_sorted' + p.suffix))

    write(output_path, new_atoms)
    print(f"写出: {output_path}")
    return output_path


if __name__ == '__main__':
    inp = sys.argv[1] if len(sys.argv) > 1 else './model.xyz'
    tol = float(sys.argv[2]) if len(sys.argv) > 2 else 1.5
    sort_heterostructure(inp, layer_tol=tol)