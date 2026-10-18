# Datasets
To generate datasets, you need data and `generate_dataset.py`.

## LCAO Hamiltonian dataset
### 0. Introduction
The LCAO Hamiltonian datasets use structures and DFT matrix results. SCF data contains Hamiltonian and overlap matrices and can be used for training or inference. Overlap-only data is for inference.

We provide interface with ABACUS and OpenMX.

This directory contains four LCAO examples covering SCF and overlap-only calculations with ABACUS and OpenMX, separately. 

**These examples have no practical application value.**

### 1. Before start
If you use OpenMX, you need to patch the software. 

We now provide patches for two OpenMX versions: 3.9.9 and 4.0.1. Each version has two parts, `generate_dataset` and `patch`.

- [`generate_dataset`](../../interface/openmx/ver4.0.1/generate_dataset/) contains source for `openmx2hotham`, which converts `openmx.scfout` to `hamiltonian.txt`. To build it, run the following command in that directory:

  ```bash
  mpicc -O3 -fopenmp -fcommon openmx2hotham.c read_scfout.c -o openmx2hotham
  ```

  After an SCF calculation, run `openmx2hotham` to export the Hamiltonian and overlap matrices:

  ```bash
  ./openmx2hotham openmx.scfout ham olp > hamiltonian.txt
  ```

- [`patch`](../../interface/openmx/ver4.0.1/patch/) contains some modified OpenMX source code. Apply them to the OpenMX `source` directory and rebuild OpenMX. 

  Run the patched OpenMX with `-olp` to export required data without SCF

  ```bash
  ./openmx openmx.dat -olp
  ```

  This produces `overlap.txt`.

### 2. Prepare DFT data
| Software | Calculation | Required files | How to generate them |
| :---: | :---: | :---: | :---: |
| ABACUS | SCF | `OUT.ABACUS/data-HR-sparse_SPIN0.csr`<br>`OUT.ABACUS/data-SR-sparse_SPIN0.csr`<br>`model.xyz` | Run ABACUS with `calculation scf` and `out_mat_hs2 1` |
| ABACUS | Overlap only | `OUT.ABACUS/SR.csr`<br>`model.xyz` | Run ABACUS with `calculation get_S` |
| OpenMX | SCF | `hamiltonian.txt`<br>`model.xyz` | Run OpenMX with `HS.fileout On`, then run `openmx2hotham` with `ham olp` |
| OpenMX | Overlap only | `overlap.txt`<br>`model.xyz` | Run OpenMX with `-olp` |

### 3. Generate datasets
Use `tools/generate_dataset.py` to generate datasets with cpu.

Before running it, edit the `inputfile` dictionary near the end of the script:

- Set `trainset` and `testset` to the parent directory containing the corresponding DFT results. Comment out any dataset entry you do not need.
- Set `dft` to `abacus` or `openmx`.
- Set `hamiltonian_type` to `lcao`.
- Set `int_dtype` (`int32` or `int64`) and `float_dtype` (`float32` or `float64`) to match the training and inference settings. Complex dtype follows `float_dtype`.
- Set `orbit` to all orbitals in the LCAO basis, in the same order as the DFT matrices.
- `L_max` sets the highest angular momentum order stored in the dataset. It can be higher than the order used for training or inference. Change it only when needed.
- For ABACUS, `orbital_cutoff` depends on the numerical orbitals, and `nonlocal_cutoff` depends on the pseudopotential. Change them only when needed.
- Don't modify the orthTB only parameters.

The script writes `trainset.pth` and `testset.pth` in the current directory for the enabled entries.

### 4. Examples
| Example | Calculation | Dataset contents | Usage |
| :---: | :---: | :---: | :---: |
| [`example_lcao_abacus_h`](example_lcao_abacus_h/) | ABACUS SCF | Hamiltonian and overlap | Training or inference |
| [`example_lcao_abacus_s`](example_lcao_abacus_s/) | ABACUS overlap only | Overlap | Inference |
| [`example_lcao_openmx_h`](example_lcao_openmx_h/) | OpenMX SCF | Hamiltonian and overlap | Training or inference |
| [`example_lcao_openmx_s`](example_lcao_openmx_s/) | OpenMX overlap only | Overlap | Inference |

You first need to change the paths in the jobfiles and DFT inputs to match your setup, such as the paths to ABACUS/OpenMX, the OpenMX patch, the Conda environment, pseudopotentials, and orbitals.

Then follow the instructions below.

1. In `scf/` or `olp_only/`, submit the DFT job with `jobfile-*`
2. After the DFT job finishes, return to the example directory and submit `jobfile-generate_dataset`.

Each example calls its output `trainset.pth`. 

**It is the data itself not the filename that determines how the data can be used.**

## Orthogonal tight-binding Hamiltonian dataset
### 0. Introduction
The orthogonal tight-binding datasets use structures and DFT band results. Data with bands can be used for training or inference. Data without bands is for inference.

We provide interface with ABACUS and OpenMX.

This directory contains ABACUS and OpenMX band examples, plus a structure-only example.

### 1. Prepare DFT data
| Software | Calculation | Required files | How to generate them |
| :---: | :---: | :---: | :---: |
| ABACUS | SCF and NSCF band | `OUT.ABACUS/BANDS_1.dat`<br>`OUT.ABACUS/running_nscf.log`<br>`model.xyz` | Run SCF, then NSCF with `out_band 1` |
| OpenMX | SCF and band | `openmx.Band`<br>`model.xyz` | Run OpenMX with `Band.dispersion on` |
| - | None | `model.xyz` | Only need structures |

### 2. Generate datasets
Use `tools/generate_dataset.py` to generate datasets with cpu.

Before running it, edit the `inputfile` dictionary near the end of the script:

- Set `trainset` and `testset` to the parent directory containing the corresponding DFT results. Comment out any dataset entry you do not need.
- Set `dft` to `abacus` or `openmx`.
- Set `hamiltonian_type` to `orthtb`.
- Set `int_dtype` (`int32` or `int64`) and `float_dtype` (`float32` or `float64`) to match the training and inference settings. Complex dtype follows `float_dtype`.
- Set `orbit` to the higher-principal-quantum-number orbitals used by the TB model.
- `L_max` sets the highest angular momentum order stored in the dataset. It can be higher than the order used for training or inference. Change it only when needed.
- Set `rc` slightly larger than the longest atomic bond length.
- Set `k_gap` selects every nth band k-point; use `1` to keep all points.
- Don't modify the LCAO only parameters.

The script writes `trainset.pth` and `testset.pth` in the current directory for the enabled entries.

### 3. Examples
| Example | Calculation | Dataset contents | Usage |
| :---: | :---: | :---: | :---: |
| [`example_orthtb_abacus_band`](example_orthtb_abacus_band/) | ABACUS SCF and NSCF band | Structure and bands | Training or inference |
| [`example_orthtb_openmx_band`](example_orthtb_openmx_band/) | OpenMX SCF and band | Structure and bands | Training or inference |
| [`example_orthtb_no_band`](example_orthtb_no_band/) | None | Structure only | Inference |

You first need to change the paths in the jobfiles and DFT inputs to match your setup, such as the paths to ABACUS/OpenMX, the Conda environment, pseudopotentials, and orbitals.

Then follow the instructions below.
1. For the band examples, submit the DFT job in `scf/` with `jobfile-*` (SCF followed by NSCF for ABACUS).
2. After the DFT job finishes, or immediately for the structure-only example, submit `jobfile-generate_dataset` from the example directory.

