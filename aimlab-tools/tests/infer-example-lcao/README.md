# infer-example-lcao

Status: not_run (materials only).

Original resources: `hotham-tests/inference/example_lcao/`; original program: `hotham/entrypoints/main.py`. Resolve these from repository root. Copy the complete example to a fresh external work directory and enter it; preserve scientific settings and its own dataset and model.pth. Use a compatible selected GPU environment.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 -m hotham.entrypoints.main hotham-input.yaml > log.txt 2>&1
python diagonal.py -diag > diagonal.log 2>&1
```

Run subsequent stages only after previous success. A resource manager may select GPU count and rendezvous settings, preserving the original model/scientific inputs. Do not run these computations on a login node.

Acceptance: commands exit 0. CSR outputs h_pred_0.csr, h_ref_0.csr and olp.csr are nonempty and parseable, with consistent matrix dimensions and finite stored values. diagonal.py -diag exits 0; eigen.npy and dos.npy are readable with finite arrays and shapes consistent with the configured k-points, orbitals and energy grid; band.png and dos.png are readable nonempty images. No predicted/reference agreement or accuracy threshold is required.

Keep the supplied model and diagonal.py settings and run one normal diagonalization. Check actual output readability and finite numerical content; visual band overlap, MAE and maximum error are not acceptance requirements.

Save source SHA, environment/device, commands, exit codes, outputs and assessment in the external work directory. Missing prerequisites: blocked; command/acceptance failure: failed; unexecuted: not_run; passed requires actual output assessment.
