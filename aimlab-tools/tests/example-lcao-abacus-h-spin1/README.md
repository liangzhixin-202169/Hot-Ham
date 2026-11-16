# example_lcao_abacus_h_spin1

Status: not_run (materials only).

Original entry/resources: `hotham-tests/datasets/example_lcao_abacus_h_spin1/`; guide: `hotham-tests/datasets.md` (repository-relative paths). Copy that directory to a fresh external work directory; keep its hierarchy and run from the copied root.

Run ABACUS SCF in scf/ with H/S export; confirm SCF convergence and nonempty CSR matrices.

Inspect the original jobfiles before scheduling: avoid repeating stages they already perform. Adapt executable/resource paths in the copy only. Then run `python generate_dataset.py`. Its inputfile dictionary is the configuration interface, not a YAML/CLI interface.

Acceptance: All required calculation/conversion commands and generator exit 0. Load newly generated trainset.pth in the actual consuming Hot-Ham environment: at least one sample, nonempty positions, finite numerical tensors, Hamiltonian H and overlap S.

External engine binaries, pseudopotentials and orbitals are environment prerequisites, not bundled resources. Missing prerequisites are blocked; an explicitly absent optional engine permits not_run with the reason. Do not claim joint verification without actually running the external stage and loading its output. Save commands, source SHA, dependency/device identity, exit records and output assessment in the external work directory.
