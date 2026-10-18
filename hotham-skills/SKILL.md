---
name: hotham-skills
description: Guide Hot-Ham users through LCAO and OrthTB dataset preparation and training using the examples in this repository. Inference guidance is work in progress. Use for Hot-Ham workflow and input-parameter questions.
---

# Hot-Ham

Use the guide matching the user's task. Check the example inputs and jobfiles before adapting them; their software paths, Conda environments, and Slurm partitions belong to one cluster and must be changed for the user's setup.

- [Datasets](datasets/README.md): completed guide and examples for LCAO and OrthTB dataset generation from ABACUS, OpenMX, or structures alone.
- [Training](training/README.md): completed guide and examples for LCAO and OrthTB training and checkpoint resumption.
- [Inference](inference/README.md) (WIP): LCAO Hamiltonian export and band or DOS analysis; check the code and example inputs before use.
- [Parameters](parameters.md): look up input keys, defaults, and which settings a run requires.

The dataset and training examples cover both LCAO and OrthTB. They are illustrative; choose data and settings appropriate for your system.
