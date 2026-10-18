import numpy as np
import torch


cubic_magnetic_dict = {0: [0], 1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2]}
lm_magnetic_dict = {0: [0], 1: [-1, 0, 1], 2: [-2, -1, 0, 1, 2]}


# magnetic quantum number
def generate_cubic_to_lm_matrix(cubic_mag_nums, lm_mag_nums, device='cpu', dtype=torch.float32):
    sqrt_2 = torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    inv_sqrt_2 = 1.0 / sqrt_2
    complexdtype = torch.complex128 if dtype == torch.float64 else torch.complex64
    transformation_matrix = torch.zeros([len(cubic_mag_nums), len(cubic_mag_nums)], device=device, dtype=complexdtype)

    for i in range(len(cubic_mag_nums)):
        magnetic_quantum = cubic_mag_nums[i]
        if magnetic_quantum == 0:
            j = lm_mag_nums.index(magnetic_quantum)
            transformation_matrix[i, j] = 1
        elif magnetic_quantum < 0:
            j = lm_mag_nums.index(magnetic_quantum)
            transformation_matrix[i, j] = 1.0j * inv_sqrt_2
            j = lm_mag_nums.index(-1 * magnetic_quantum)
            transformation_matrix[i, j] = 1.0j * inv_sqrt_2 * (-1) ** (magnetic_quantum + 1)
        elif magnetic_quantum > 0:
            j = lm_mag_nums.index(-1 * magnetic_quantum)
            transformation_matrix[i, j] = 1.0 * inv_sqrt_2
            j = lm_mag_nums.index(magnetic_quantum)
            transformation_matrix[i, j] = 1.0 * inv_sqrt_2 * (-1) ** magnetic_quantum
        else:
            raise Exception
    return transformation_matrix


def create_basis_lm(orbital_type: str):
    magnetic_list = lm_magnetic_dict[orbital_type]
    basis_lm = []
    for magnetic in magnetic_list:
        for spin_state in [1, -1]:
            basis_lm.append([orbital_type, magnetic, spin_state])
    return basis_lm


def compute_matrix_lm_basis(basis_lm, device='cpu', dtype=torch.float32):
    num_dimensions = len(basis_lm)
    matrix_LpSm = torch.zeros([num_dimensions, num_dimensions], device=device, dtype=dtype)
    matrix_LmSp = torch.zeros([num_dimensions, num_dimensions], device=device, dtype=dtype)
    matrix_LzSz = torch.zeros([num_dimensions, num_dimensions], device=device, dtype=dtype)

    if basis_lm is None:
        LdotS_matrix = torch.zeros([2, 2], device=device, dtype=dtype)
    else:
        for i in range(len(basis_lm)):
            row_index = i
            coeff, transformed_basis = map_LpSm(basis_lm[i])
            if transformed_basis in basis_lm:
                col_index = basis_lm.index(transformed_basis)
                matrix_LpSm[row_index, col_index] = coeff

            coeff, transformed_basis = map_LmSp(basis_lm[i])
            if transformed_basis in basis_lm:
                col_index = basis_lm.index(transformed_basis)
                matrix_LmSp[row_index, col_index] = coeff

            coeff, transformed_basis = map_LzSz(basis_lm[i])
            if transformed_basis in basis_lm:
                col_index = basis_lm.index(transformed_basis)
                matrix_LzSz[row_index, col_index] = coeff

        LdotS_matrix = 0.5 * (matrix_LpSm + matrix_LmSp + matrix_LzSz)
    return LdotS_matrix


def map_LpSm(lm_state):
    l, m, s = lm_state[0], lm_state[1], lm_state[2]
    if s == 1:
        coeff = np.sqrt((l - m) * (l + m + 1))
    else:
        coeff = 0
    return coeff, [l, m + 1, s - 2]


def map_LmSp(lm_state):
    l, m, s = lm_state[0], lm_state[1], lm_state[2]
    if s == -1:
        coeff = np.sqrt((l + m) * (l - m + 1))
    else:
        coeff = 0
    return coeff, [l, m - 1, s + 2]


def map_LzSz(lm_state):
    l, m, s = lm_state[0], lm_state[1], lm_state[2]
    coeff = m if s == 1 else -m
    return coeff, [l, m, s]


def compute_soc_matrix_in_cubic_basis(orbital, cubic_mag_nums=None, lm_mag_nums=None, device='cpu', dtype=torch.float32):
    if cubic_mag_nums is None:
        cubic_mag_nums = cubic_magnetic_dict[orbital]
    if lm_mag_nums is None:
        lm_mag_nums = lm_magnetic_dict[orbital]

    lm_basis = create_basis_lm(orbital)
    transformation_matrix = generate_cubic_to_lm_matrix(cubic_mag_nums, lm_mag_nums, device=device, dtype=dtype)
    soc_matrix_lm = compute_matrix_lm_basis(lm_basis, device=device, dtype=dtype)
    soc_matrix_lm_complex = torch.complex(soc_matrix_lm, torch.zeros_like(soc_matrix_lm))

    spin_transformation = torch.kron(transformation_matrix, torch.eye(2, device=device, dtype=transformation_matrix.dtype)).T
    spin_transpose_conjugate = torch.conj(spin_transformation.T)
    soc_matrix_cubic = spin_transpose_conjugate @ soc_matrix_lm_complex @ spin_transformation

    num_orbitals = len(cubic_mag_nums)
    assert len(soc_matrix_cubic) == 2 * num_orbitals
    soc_matrix_block = torch.zeros([2 * num_orbitals, 2 * num_orbitals], device=device, dtype=transformation_matrix.dtype)
    soc_matrix_block[:num_orbitals, :num_orbitals] = soc_matrix_cubic[::2, ::2]
    soc_matrix_block[num_orbitals:, num_orbitals:] = soc_matrix_cubic[1::2, 1::2]
    soc_matrix_block[:num_orbitals, num_orbitals:] = soc_matrix_cubic[::2, 1::2]
    soc_matrix_block[num_orbitals:, :num_orbitals] = soc_matrix_cubic[1::2, ::2]

    return soc_matrix_block



