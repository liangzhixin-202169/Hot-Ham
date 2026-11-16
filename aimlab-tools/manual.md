# Hot-Ham operation manual

Hot-Ham learns electronic Hamiltonians in LCAO and orthogonal tight-binding representations. Scope here: seven existing dataset conversion routes, LCAO/OrthTB training, and LCAO CSR prediction followed by the existing band/DOS example. Additional magnetic/SOC, rR, matrix-k and performance workflows are not certified by these cases. Resolve paths from repository root; see [installation](install.md), [interfaces](runtime-composition.md) and [tests](tests/README.md).

## Inputs and outputs

| Function | Required inputs and configuration | Outputs in work directory |
| --- | --- | --- |
| Seven dataset routes | Copied original example hierarchy, structures and required external H/S or bands; reviewed generator inputfile dictionary specifying paths, orbit basis and representation | trainset.pth; H/S, S, reference bands or structure-only content according to route |
| Training | YAML plus its trainset .pth; representation, basis, cutoff and model settings | log.txt captured by caller; checkpoint/cp*.pth at checkpoint_interval under model_save_path |
| LCAO prediction | YAML, compatible checkpoint and dataset; prediction: 2, init_from_checkpoint, write: [h_pred, h_ref, olp] for the example | h_pred_0.csr, h_ref_0.csr, olp.csr (ABACUS old CSR) |
| Subsequent example post-processing | model.xyz, exported CSR and copied diagonal.py inputfile settings | band.png, eigen.npy, dos.png, dos.npy |

The declared functions have required inputs/outputs and no independent CLI parameters; scientific parameters live in their native YAML/Python configuration. There are no additional optional machine ports. For production prediction, h_ref is requested only when reference H exists; the declared example requires it. Do not assume every dataset has H or bands. YAML trainset/init_from_checkpoint/model_save_path paths resolve from the run directory. Stage referenced resources preserving paths. The source-data directory for dataset functions is staging input; copy its hierarchy and selected generator into the run directory. Production command runs that copied, reviewed generator.

Choose a fresh caller-owned external work directory for every case. Copy the selected example without mutating the repository. Do not overwrite previous results or run examples in-place. Relative matrix and model paths must agree with the configuration. Work outputs, environment files and logs are not source delivery resources.

## Parameters and execution

Show representation, input data, checkpoint (inference), device, orbit, rc, prediction/write and output location first: these define the scientific task and resource choice. For training also show epoch, lr, batch_size and checkpoint_interval. Place channel widths, L_max, dtype, optimizer/scheduler details and distributed rendezvous controls in advanced settings; changing model architecture still requires checkpoint compatibility. This is display guidance for native configuration, not invented CLI flags. Defaults and allowed keys are in hotham/entrypoints/parameters.py and hotham-tests/parameters.md; retain original example overrides for acceptance. Required scientific choices must be supplied explicitly rather than guessed from defaults.

Dataset: finish the route's prerequisite calculations/conversion, then `python generate_dataset.py` in its copied example root. Each generator contains an inputfile dictionary; edit only the work copy. Consult hotham-tests/datasets.md and the per-case README.

Training/prediction from the staged work directory:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 -m hotham.entrypoints.main hotham-input.yaml > log.txt 2>&1
```

Training uses prediction: 0. Inference uses its own bundled model and settings, not a training example checkpoint. For post-processing, review diagonal.py's structure, spin, valence electrons, dtype/device, CSR paths, DOS grid and band path. Then run once:

```bash
python diagonal.py -diag > diagonal.log 2>&1
```

Band energies use the saved Fermi alignment in eV; integration acceptance checks usable outputs, without predicted/reference accuracy comparisons. Atomic structures and matrix units follow the original converter/diagonal implementations and guides; do not treat CSR values as eV without the consumer's conversion.

## Completion and failures

Require actual command exits and content checks in tests/README.md. Training completes 10 epochs with finite loss/metrics and readable checkpoints at epochs 5 and 10. Inference checks nonempty parseable CSR matrices with consistent dimensions and finite values, plus readable reconstruction arrays/images with finite numerical content and shapes consistent with the configured inputs. No training-fit or band-agreement threshold is imposed. Missing binary, model, dataset, basis or incompatible dependency: blocked. Nonzero exit, missing/corrupt output or failed acceptance: failed. Unexecuted: not_run. Preserve diagnostic logs; resolve original job state before any retry. Report the environment/version and covered functions without inferring wider scientific validity or tool activation.
