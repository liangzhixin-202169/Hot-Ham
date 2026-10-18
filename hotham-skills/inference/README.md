# Inference
## LCAO Hamiltonian model
To infer LCAO Hamiltonian, you need a dataset and a model.

[`example_lcao`](example_lcao/) demonstrates how to infer the Hamiltonian of twisted multilayer graphene from a trained LCAO checkpoint and calculate its band structure. It includes an example structure, a dataset, and the scripts needed for prediction and diagonalization.

**Checkpoint:** The supplied `cp3000.pth` is a trained checkpoint for twisted multilayer graphene. For a new structure, prepare a matching inference dataset and follow the steps below.

Hot-Ham writes the predicted LCAO Hamiltonian as a sparse CSR file in ABACUS format.

### 1. Prepare the files
Put these files in the working directory:

- A trained checkpoint, such as `cp3000.pth`.
- A dataset, such as `trainset.pth`. You can generate one from either an overlap-only or an SCF calculation.
- A structure file, such as `model.xyz`, matching the structure in `trainset.pth`.

### 2. Predict the Hamiltonian
Create an input file, or slightly change your `hotham-input.yaml` used for training with these parameters:

- `prediction`: switch to `2` to export Hamiltonian matrices.
- `write`: include `h_pred` and `olp`; add `h_ref` if the dataset contains a reference Hamiltonian.
- `trainset`: path to the dataset, such as `./trainset.pth`.
- `init_from_checkpoint`: path to your trained model checkpoint, such as `./cp3000.pth`.
- `int_dtype` and `float_dtype`: use the settings from training; complex precision follows `float_dtype` automatically.

submit [`jobfile-hotham`](example_lcao/jobfile-hotham), this writes `h_pred.csr` and `olp.csr`, and optionally `h_ref.csr`.

### 3. Calculate the band structure
[`diagonal.py`](example_lcao/diagonal.py) first diagonalizes the Hamiltonian on a k-point grid to find the Fermi level and calculate the DOS. It then diagonalizes along the band path to calculate and plot the band structure.

Before running it, edit the `inputfile` dictionary near the end of the script:

- `device`: `cuda` or `cpu`.
- `structure`: path to `model.xyz`. 
- `H_pred_csr`: path to `h_pred.csr`. 
- `H_ref_csr`: optional path to `h_ref.csr`; set it to `None` or remove it if you have no reference Hamiltonian.
- `S_csr`: path to `olp.csr`.
- `band_kpoints_num`: total number of points along the band path passing to ASE. 
- `band_kpoints_path`: high-symmetry path passing to ASE.
- `band_kpoints_pbc`: periodic boundary conditions passing to ASE.

Other parameters are: 
- `int_dtype`: `int32` or `int64` (default: `int64`).
- `float_dtype`: `float32` or `float64` (default: `float32`); sets the matching complex dtype automatically.
- `valence_electrons`: a dictionary mapping each element in the structure to its number of valence electrons, used to find the Fermi level.
- `dos_kpoints_grid`: k-point mesh for DOS. 
- `dos_kpoints_use_irreducible_k`: whether to use only symmetry-irreducible k-points. 
- `dos_sigma`: Gaussian broadening in eV. 
- `dos_energy_range`: `[minimum, maximum, number of points]` in eV relative to the Fermi level.

After `jobfile-hotham` finishes, submit [`jobfile-diag`](example_lcao/jobfile-diag).

This runs `python diagonal.py -diag` and writes `band.png` and `eigen.npy`. It also calculates the density of states and writes `dos.png` and `dos.npy`. 

### 4. Export matrix_k
After predicting `h_pred.csr` and `olp.csr`, use the same `inputfile` settings in `diagonal.py` to export the Hamiltonian and overlap matrices at selected k-points. Use `-band` for points along `band_kpoints_path`, or `-dos` for points in `dos_kpoints_grid`:

```bash
python diagonal.py -band all
python diagonal.py -dos all
```

`all` exports every k-point sequentially in one process. Replace it with a zero-based k-point index, such as `0`, to export only that point; this lets you submit separate jobs for different indices and export them in parallel. The commands write to `matrix_band/k0/`, `matrix_band/k1/`, etc., or to the corresponding directories under `matrix_dos/`. The DOS command also writes `matrix_dos/kinfo.txt` with each grid point's coordinates and weight.

Each k-point directory contains `kN_H.npz` for the predicted Hamiltonian and `kN_S.npz` for the overlap matrix. If `H_ref_csr` is set, it also contains `kN_Href.npz` for the reference Hamiltonian. Each `.npz` stores the sparse CSR components `data`, `indices`, `indptr`, and `shape`.

### 5. Output the rR matrix
Training does not use the position matrix `rR`, so the training dataset does not need to contain it. To export `rR` during Hamiltonian reconstruction, first generate an inference dataset that includes `rR` for the target structure:

| Software | Calculation | Required files | How to generate them |
| :---: | :---: | :---: | :---: |
| ABACUS | SCF | `OUT.ABACUS/data-HR-sparse_SPIN0.csr`<br>`OUT.ABACUS/data-SR-sparse_SPIN0.csr`<br>`OUT.ABACUS/data-rR-sparse.csr`<br>`model.xyz` | Set `calculation scf`, `out_mat_hs2 1`, and `out_mat_r 1` in `INPUT`. |
| ABACUS | Overlap only | `OUT.ABACUS/SR.csr`<br>`OUT.ABACUS/data-rR-sparse.csr`<br>`model.xyz` | Set `calculation get_S` and `out_mat_r 1` in `INPUT`. |
| OpenMX | SCF | `hamiltonian.txt`<br>`model.xyz` | Set `HS.fileout On`, then run `./openmx2hotham openmx.scfout ham olp rr > hamiltonian.txt`. |
| OpenMX | Overlap only | `overlap.txt`<br>`model.xyz` | Use the [patched OpenMX](../datasets/README.md#1-before-start) and run `./openmx openmx.dat -olp -rr > log`. |

For ABACUS, `generate_dataset.py` reads `data-rR-sparse.csr` automatically. 

For OpenMX, it reads the x, y, and z components from `hamiltonian.txt` or `overlap.txt`. The overlap-only routes provide overlap and `rR` data without a reference Hamiltonian. 

If your existing `.pth` dataset was generated without `rR`, regenerate that dataset after preparing the DFT output. You can reuse the trained checkpoint; no retraining is needed. Set `trainset` to the new dataset, keep `prediction: 2` and `init_from_checkpoint`, and add `rR` to `write` in `hotham-input.yaml`:

```yaml
write: [h_pred, olp, rR]
```

Run the Hamiltonian export as in step 2. For a single structure, it writes `rR.csr` in ABACUS format in the working directory alongside the requested Hamiltonian and overlap files.

## OrthTB Hamiltonian model(WIP)

## OrthTB LSQT(WIP)