---
name: hotham-package
description: Guide the calling application through Hot-Ham installation requirements and the repository's dataset, training and inference acceptance tests. Use for Hot-Ham setup, validation and workflow guidance.
---

> Instructions for AI agents. Human readers should use README.md and the guides in hotham-tests/.

# Hot-Ham

## 1. Introduction

Hot-Ham learns electronic Hamiltonians in LCAO and orthogonal tight-binding representations. This repository contains the program, ABACUS/OpenMX interfaces, usage guides and runnable examples.

This skill supplies requirements, execution plans and acceptance criteria to the higher-level application. That application checks and provisions environments, selects compute resources, runs jobs and evaluates results. Pass the instructions below to it; do not independently inspect or change the user's environment, install software, compile engines or submit jobs. Use its actual interfaces when available; otherwise return the plan as structured text without inventing an integration protocol.

The directory containing this file is `REPO_ROOT`. Resolve resource links from it. Read the guide relevant to the task:

- [Dataset preparation](hotham-tests/datasets.md)
- [Training and benchmark criteria](hotham-tests/training.md)
- [Inference and post-processing](hotham-tests/inference.md)
- [Input parameters](hotham-tests/parameters.md); defaults are defined in [parameters.py](hotham/entrypoints/parameters.py)

## 2. Installation

Send the following checklist to the application. Request its environment-check results before planning the applicable examples. Checks and any required installation/build actions are the application's responsibility.

| Component | Requirement to communicate | If unavailable |
| --- | --- | --- |
| Python | Confirm the Python version and dependencies satisfy [setup.py](setup.py). | Request a suitable environment; dependent tests cannot run until ready. |
| CUDA/PyTorch | Confirm the selected device, driver and PyTorch CUDA build are compatible. Choose `torch-scatter` wheels using the installed PyTorch version and `torch.version.cuda`, not `nvidia-smi` alone. | Return the unmet requirement to the application. Do not silently switch the examples to CPU; let it resolve the execution environment. |
| Hot-Ham and dependencies | Install this repository using [README.md](README.md) and `setup.py`; a local install can use `python -m pip install -e "$REPO_ROOT"`. Install `torch-scatter` separately for the chosen PyTorch/CUDA combination. Confirm imports, dependency consistency, intended source identity and usable `torchrun`. | Request installation or repair through the application. |
| OpenMX — optional | Ask whether the user has OpenMX. If available, confirm its executable, version, basis/pseudopotential resources, converter and patch capabilities described below. | If the user says it is absent, skip the three OpenMX dataset examples. Do not require OpenMX installation. |
| ABACUS — optional | Ask whether the user has ABACUS. If available, confirm its executable/version, MPI setup, pseudopotentials and numerical orbitals, and support for SCF H/S export, `get_S` and band calculations. The dataset guide recommends 3.10.1 LTS. | If the user says it is absent, skip the three ABACUS dataset examples. Do not require ABACUS installation. |

OpenMX preparation requirements:

- Sources are supplied for [3.9.9](hotham-interface/openmx/ver3.9.9/) and [4.0.1](hotham-interface/openmx/ver4.0.1/). Ask the application to select matching resources; a version string does not establish that the required patch is present.
- SCF matrix conversion needs `openmx2hotham`. Ask the application to compile the matching `generate_dataset/` sources in an external build directory with:

  ```bash
  mpicc -O3 -fopenmp -fcommon openmx2hotham.c read_scfout.c -o openmx2hotham
  ```

- The overlap-only example needs a patched OpenMX supporting `-olp`. Supply the matching `patch/` location to the application so it can coordinate preparation and capability verification. Do not instruct it to overwrite an unknown or incompatible existing installation.

Absent optional engines only remove their dataset examples. The structure-only dataset example, both training examples and the inference example use supplied resources and remain in scope. An installed engine with a broken configuration is a prerequisite problem to report, not a reason to silently mark its examples as passed or skip them.

## 3. Post-installation tests

Request only the three groups below. Do not add short-run training, checkpoint-resume, k-space export, rR, magnetic/SOC or separate performance tests to this installation acceptance plan.

Ask the application to run copies of the examples in its chosen work directories. Keep the repository examples unchanged. It may adapt environment paths, launch commands and resource settings in those copies; retain the scientific inputs, model settings and benchmark conditions. Existing jobfiles describe one cluster and must be interpreted using the application's actual execution environment.

### 3.1 Dataset examples: seven cases

Follow [datasets.md](hotham-tests/datasets.md). For each applicable case, finish its calculation/conversion stages before running `python generate_dataset.py` from the copied example root. Overlap-only and structure-only cases do not require SCF.

| Example under `hotham-tests/datasets/` | Sequence to request | Skip condition |
| --- | --- | --- |
| [example_lcao_abacus_h_spin1](hotham-tests/datasets/example_lcao_abacus_h_spin1/) | SCF in `scf/` with H/S export → generate dataset | User has no ABACUS |
| [example_lcao_abacus_s_spin1](hotham-tests/datasets/example_lcao_abacus_s_spin1/) | `get_S` in `olp_only/` → generate dataset; one MPI process is recommended for `get_S` | User has no ABACUS |
| [example_lcao_openmx_h_spin1](hotham-tests/datasets/example_lcao_openmx_h_spin1/) | SCF in `scf/` → `openmx2hotham openmx.scfout ham olp > hamiltonian.txt` → generate dataset | User has no OpenMX |
| [example_lcao_openmx_s_spin1](hotham-tests/datasets/example_lcao_openmx_s_spin1/) | Patched OpenMX `-olp` in `olp_only/` → generate dataset | User has no OpenMX |
| [example_orthtb_abacus_band](hotham-tests/datasets/example_orthtb_abacus_band/) | SCF → NSCF band calculation in `scf/`, following the supplied jobfiles → generate dataset | User has no ABACUS |
| [example_orthtb_openmx_band](hotham-tests/datasets/example_orthtb_openmx_band/) | OpenMX SCF and band calculation in `scf/` → generate dataset | User has no OpenMX |
| [example_orthtb_no_band](hotham-tests/datasets/example_orthtb_no_band/) | Generate dataset directly from supplied `model.xyz`; no DFT step | None |

Tell the application to inspect the existing jobfiles before scheduling stages: some already include conversion or calculation preparation, which must not be duplicated. Require successful calculation completion and SCF convergence where applicable before conversion.

**Acceptance:** each executed example produces a new, nonempty `trainset.pth`. Ask the application to load it in the Hot-Ham environment and check that it contains at least one sample, nonempty atomic positions and finite numerical data. Check the expected content using the selected script's dataset representation: H and S for LCAO SCF, S for overlap-only, reference bands for OrthTB band cases, and structure data for the no-band case. Do not require H or bands in cases that intentionally omit them. These are lightweight checks on generated data, not additional scientific calculations.

### 3.2 Training examples: two cases

Request both supplied examples with their own datasets and configurations:

- [LCAO training](hotham-tests/training/example_lcao_spin1/)
- [OrthTB training](hotham-tests/training/example_tb/)

Use the training guide's launcher in each copied example directory, with the application's selected GPU count and port:

```bash
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" \
  -m hotham.entrypoints.main hotham-input.yaml > log.txt
```

**Acceptance:** use the **Reference results** under [Examples](hotham-tests/training.md#4-examples) as the authority for the evaluation epoch and conservative `Train_MSE`/`Train_MAE` limits. Both examples must meet their corresponding limits with finite metrics. Do not replace this with a two-epoch smoke test, weaken thresholds or use a different epoch to manufacture a pass. Do not duplicate the changing epoch counts and limits here.

Training uses the supplied training datasets, not outputs substituted from section 3.1. No new DFT calculation is needed. Report the actual checked epoch and both metrics; these checks measure the example's training fit, not generalization.

### 3.3 Inference example: band agreement

Request the supplied [LCAO inference example](hotham-tests/inference/example_lcao/) using [inference.md](hotham-tests/inference.md):

1. Use its own `model.pth`, `trainset.pth`, `model.xyz` and `hotham-input.yaml`. Do not substitute the training examples' checkpoints; their model settings differ.
2. Run the Hot-Ham launcher above with the inference YAML to export `h_pred_0.csr`, `h_ref_0.csr` and `olp.csr`.
3. Run `python diagonal.py -diag` once, preserving the supplied example configuration. It already calculates predicted/reference bands and generates `band.png` and `eigen.npy`; its normal DOS outputs need no separate acceptance test.

**Acceptance:** predicted and reference bands agree within the plotted energy window, from -2 to 2 eV relative to the Fermi level. Ask the higher-level AI to inspect the already generated `band.png`: the blue predicted and dashed reference dispersions should substantially overlap along the plotted path, without obvious systematic shifts or missing/distorted branches within that window. High-energy unoccupied states outside the plotted window may show significant deviations; these do not affect this example's acceptance. A file merely existing is not sufficient. Record the visual conclusion and any visible discrepancies.

For a numerical comparison, the application may instead or additionally read the existing `eigen.npy` dictionary and check that `eigenvalues` and `eigenvalues_ref` are finite and equally shaped. Compare corresponding values where the reference energy is within -2 to 2 eV, and report the selected range, number of values, MAE and maximum absolute difference in eV. This selection includes any prediction displaced outside the window from a reference value inside it. Use the saved energy alignment; do not shift bands to improve agreement. This is array arithmetic only: do not rerun DFT, inference or diagonalization, refine the grid or perform an extra fit. No numerical tolerance is currently specified by the example guide, so do not invent one and claim a quantitative pass. Use visual agreement for the acceptance conclusion, or have the application/user define a tolerance before numerical acceptance. If neither assessment is possible, return the missing assessment rather than a pass.

### Results to return

Ask the application to retain per-example commands, execution status, logs/output locations and the acceptance evidence: dataset sample/content checks, training epoch and metrics, and band comparison. Capture software/source identity and the environment used so results remain attributable.

Record `passed`, `failed`, `blocked` or `not_run` per example. For optional-engine skips, use `not_run` with an explicit reason such as “user has no OpenMX”; report applicable tests passed and skipped coverage separately. Prerequisite failures are `blocked`; execution or acceptance failures are `failed`. A scheduled or unexecuted test is not passed. Preserve useful failure evidence and return the issue to the application; do not automatically resubmit or change scientific settings. Temporary execution files are managed and cleaned by the application while retaining requested outputs and diagnostic evidence.
