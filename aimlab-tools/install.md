# Installation guidance

All paths in this document are repository-relative. Select an isolated environment with Python >3.10 and a compatible PyTorch/CUDA/driver combination for the chosen GPU. CPU dataset preparation is supported. Supplied training/inference configurations retain their device settings; do not silently change them to claim GPU availability.

From a complete checkout, install with `python -m pip install "$REPO_ROOT"`, where REPO_ROOT is the caller-selected repository root. Preserve source cleanliness by building from a disposable copy if packaging writes metadata into source. The pins/ranges in setup.py are authoritative, including ase==3.25.0, e3nn==0.6.0, numpy==2.4.4 and torch>=2.4.0. Do not substitute versions merely to make installation appear successful. Install torch-scatter separately using the matching installed PyTorch version and torch.version.cuda, following README.md. Other requirements include torch-geometric, scipy, spglib, pyyaml, matplotlib and tqdm.

Use the chosen environment to verify dependency consistency and the actual Hot-Ham import source, then run the applicable real cases in tests/README.md on selected compute resources. A successful import is not function validation. CPU-only installation does not prove CUDA support. Minimum launcher topology is one process and one selected GPU for the bundled model examples; actual memory/time must be allocated from their unchanged settings, not assumed suitable for a login node.

Optional external-data producers:

- ABACUS: dataset guide recommends 3.10.1 LTS; require relevant SCF H/S export, get_S or band capabilities. Provide MPI, pseudopotentials and numerical orbitals via the selected environment; modify copied input paths only.
- OpenMX: repository supplies interface sources for 3.9.9 and 4.0.1 under hotham-interface/openmx/. Use version-matched converters and patches. Build the matching generate_dataset/openmx2hotham sources in an external build directory using `mpicc -O3 -fopenmp -fcommon openmx2hotham.c read_scfout.c -o openmx2hotham`. Overlap-only requires the matching -olp patch. Do not overwrite an existing unknown installation.

Engine binaries and basis files are not included. Missing prerequisites block affected cases; explicitly absent optional engines may be recorded not_run with reason. Structure-only generation, supplied training datasets and the inference checkpoint do not require external DFT installation. No installation or runtime check has been performed as part of preparing this delivery.
