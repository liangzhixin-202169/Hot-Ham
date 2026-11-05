# Inference
## LCAO Hamiltonian model
To infer a LCAO Hamiltonian, you need a model, and turn your structure into a dataset waiting to be predicted.

Hot-Ham writes the predicted LCAO Hamiltonian as a sparse CSR file in ABACUS **old** format. See the [ABACUS CSR change log](https://abacus.deepmodeling.com/en/v3.11.0-beta11/advanced/interface/migration-guide-csr-format.html).

### 1. Prepare the files
Put these files in the working directory:

- A trained checkpoint, such as `model.pth`.
- A dataset, such as `trainset.pth`. You can generate one from either an overlap-only or an SCF calculation.
- A structure file, such as `model.xyz`, matching the structure in `trainset.pth`.

### 2. Predict the Hamiltonian
Create an input file, or slightly change your `hotham-input.yaml` used for training with these parameters:

- `trainset`: path to the dataset, such as `./trainset.pth`.
- `prediction`: switch to `2` to export Hamiltonian matrices.
- `write`: use `[h_pred, olp]` to export the predicted Hamiltonian and overlap matrix. Include a `h_ref` if it is in the dataset and you want to export it.
- `init_from_checkpoint`: path to your trained model checkpoint.
- Keep other parameters consistent with the checkpoint.

### 3. Calculate the band structure
To calculate the band structure, you need `diagonal.py`

It calculates the fermi energy and then the band structure.
Before running it, edit the `inputfile` dictionary near the end of it:

- `device`: `cuda` or `cpu`.
- `structure`: path to `model.xyz`.
- `spin`: `1`, `2`, or `4`.
- `H_pred_0_csr`: path to `h_pred_0.csr`.
- `H_pred_1_csr`: only needed for `spin=2`
- `H_ref_0_csr`: optional path to `h_ref_0.csr`.
- `H_ref_1_csr`: optional path, only needed for `spin=2`, 
- `S_csr`: path to `olp.csr`.
- `band_kpoints_num`: total number of points along the band path passing to ASE.
- `band_kpoints_path`: high-symmetry path passing to ASE.
- `band_kpoints_pbc`: periodic boundary conditions passing to ASE.

Other parameters are:
- `int_dtype`: `int32` or `int64`
- `float_dtype`: `float32` or `float64`. Complex dtype follows it.
- `valence_electrons`: a dictionary mapping each element in the structure to its number of valence electrons, used to find the Fermi level.
- `dos_kpoints_grid`: k-point mesh for DOS.
- `dos_kpoints_use_irreducible_k`: whether to use only symmetry-irreducible k-points.
- `dos_sigma`: Gaussian broadening in eV.
- `dos_energy_range`: `[minimum, maximum, number of points]` in eV relative to the Fermi level.

Run `python diagonal.py -diag` to calculate the band structure and density of states. The outputs are `band.png`, `eigen.npy`, `dos.png`, and `dos.npy`.

### 4. Examples

| Example | Hamiltonian type | Target |
| --- | --- | --- |
| [`example_lcao`](inference/example_lcao/) | LCAO | Band structure, DOS |

1. Enter the example directory and review `hotham-input.yaml`, `diagonal.py`, and both jobfiles.
2. Submit `jobfile-hotham` with `sbatch jobfile-hotham`. It writes `h_pred_0.csr`, `h_ref_0.csr`, and `olp.csr`.
3. After prediction finishes, submit `jobfile-diag` with `sbatch jobfile-diag`. It runs `python diagonal.py -diag` to produce the band structure and density of states.

Bands near and below the Fermi level may show good agreement, while high-energy unoccupied states above the Fermi level may show significant deviations.

### 5. Additional Functions: Export matrix_k
`diagonal.py` can also export the Hamiltonian and overlap matrices at selected k-points. Use `-band` for points along `band_kpoints_path`, or `-dos` for points in `dos_kpoints_grid`:

```bash
python diagonal.py -band all
python diagonal.py -dos all
```

`all` exports every k-point sequentially in one process. You can also export one matrix in a job below to enable parallel exportation.
```bash
python diagonal.py -band 0
```

The exportation write `*.npz` to `matrix_band/k0/`, `matrix_band/k1/`, etc., or to `matrix_dos/` with a `kinfo.txt` writing k_weights.


### 6. Additional Functions: Export the rR matrix
Training does not use the position matrix `rR`, but we support passing `rR` through the dataflow for convenience in post-processing.

First generate an inference dataset that includes `rR` for the target structure:

| Software | Calculation | Required files | How to generate them |
| --- | --- | --- | --- |
| ABACUS | SCF | `data-HR-sparse_SPIN0.csr`<br>`data-SR-sparse_SPIN0.csr`<br>`data-rR-sparse.csr`<br>`../model.xyz` | Run with `calculation scf`, `out_mat_hs2 1`, and `out_mat_r 1`. |
| ABACUS | Overlap only | `SR.csr`<br>`data-rR-sparse.csr`<br>`../model.xyz` | Run with `calculation get_S` and `out_mat_r 1`. |
| OpenMX | SCF | `hamiltonian.txt`<br>`model.xyz` | Run with `HS.fileout On`, then run `./openmx2hotham openmx.scfout ham olp rr > hamiltonian.txt`. |
| OpenMX | Overlap only | `overlap.txt`<br>`model.xyz` | Run with `./openmx openmx.dat -olp -rr > log`. |

Secondly, do normal prediction with `rR` in `write` to export a `rR.csr`

```yaml
write: [h_pred, olp, rR]
```

## OrthTB Hamiltonian model(WIP)

