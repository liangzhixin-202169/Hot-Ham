---
name: hotham-aimlab-delivery
description: Install and configure Hot-Ham, prepare datasets, train LCAO or OrthTB Hamiltonian models, export LCAO matrices and run the supplied band comparison using repository-local resources.
---

# Hot-Ham delivery

This file is the AIMatterLab delivery entry. `REPO_ROOT` is its parent directory's parent; links here are relative to this file, while manifest paths are relative to REPO_ROOT. Source baseline inspected: `a92dd76bd3102441a6076d015ede63dd8d78c657`. The declaration's eventual Git commit must be obtained from its actual checkout; this baseline is not a registration or test result.

Use for dataset preparation, LCAO/OrthTB training and LCAO prediction. Read [installation](install.md), [operation manual](manual.md), [cross-tool interfaces](runtime-composition.md), [machine declaration](manifest.json) and the relevant [test case](tests/README.md). Original scientific guides: [datasets](../hotham-tests/datasets.md), [training](../hotham-tests/training.md), [inference](../hotham-tests/inference.md), [parameters](../hotham-tests/parameters.md). Parameter defaults come from [implementation](../hotham/entrypoints/parameters.py); dataset generators and diagonal.py have their own editable inputfile dictionaries.

## Prepare and operate

1. Bind the requested checkout, environment, device and external output directory. Follow install.md; do not infer installed/configured status from this delivery. Dependency pins are authoritative in setup.py. External engines are optional per dataset route.
2. Select a manifest function and matching case. For dataset routes, copy the original example hierarchy into a fresh work directory, prepare the required external outputs, and adjust the copied generator's inputfile dictionary to actual scientific input paths and orbit definitions. Run `python generate_dataset.py` from that directory. No general dataset CLI is claimed.
3. For training, prepare YAML and its trainset file in the work directory. Required scientific choices include representation, orbital basis, cutoffs and model settings. Run `python -m torch.distributed.run --standalone --nproc_per_node=1 -m hotham.entrypoints.main hotham-input.yaml`. Capture stdout/stderr to log.txt. Multi-device execution requires explicitly selected process/device resources.
4. For LCAO prediction, prepare the compatible checkpoint and dataset; set prediction: 2, init_from_checkpoint and write in YAML as described in manual.md. Invoke the same launcher. For bands, subsequently run the copied example's `python diagonal.py -diag` after reviewing its inputfile dictionary and structure/matrix paths. Preserve alignment and scientific settings for acceptance.
5. Locate outputs relative to the work directory or the explicit paths in YAML. Check actual exit codes and required nonempty outputs. Tests additionally check dataset content, finite training loss/metrics with readable checkpoints at epochs 5 and 10, or parseable finite matrices and usable reconstruction outputs, as specified in their READMEs. Training runs 10 epochs with checkpoint_interval: 5. File existence alone does not prove output usability; scientific accuracy and visual band agreement are not acceptance requirements.

## Configuration and evidence

Keep source/examples unchanged. Stage all referenced YAML resources with their relative hierarchy; manifest file ports do not automatically rewrite embedded paths. Dataset source-data is a staging directory, consumed through script-local paths rather than a CLI directory flag. Production dataset argv uses the staged, reviewed generate_dataset.py; its declared entrypoint identifies the original script to copy. No wrapper substitutes a test for a production call.

Use only authorized compute resources; substantial training/DFT/diagonalization belongs on compute nodes. Missing resources/dependencies are blocked, execution or output failures failed, unexecuted work not_run. Save version, device/environment identity, commands, exits, logs and assessments outside the source tree. Do not repeat unknown jobs or silently change a backend/model/threshold. Only actual successful installation and function tests justify passed within their measured scope.

Current delivery: installation/tests/release are not_run. The distributed engine contract has no Hot-Ham entry: runtime adaptation remains pending, and this skill does not create it or grant activation. Uncommitted declaration files are preparation only. During later user configuration, use actual installation/test evidence to create an environment-specific execution skill; daily use follows that verified execution snapshot, with these delivery materials retained for explicit repair/update.
