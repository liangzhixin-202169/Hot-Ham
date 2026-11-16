# train-example-lcao-spin1

Status: not_run (materials only).

Original resources: `hotham-tests/training/example_lcao_spin1/`; original program: `hotham/entrypoints/main.py`. Resolve these from repository root. Copy the complete example to a fresh external work directory and enter it; preserve scientific settings and its own dataset. Use a compatible selected GPU environment.

```bash
python -m torch.distributed.run --standalone --nproc_per_node=1 -m hotham.entrypoints.main hotham-input.yaml > log.txt 2>&1
```

Run subsequent stages only after previous success. A resource manager may select GPU count and rendezvous settings, preserving the original model/scientific inputs. Do not run these computations on a login node.

Acceptance: commands exit 0. Complete 10 epochs with finite loss and reported metrics; write nonempty checkpoint/cp5.pth and checkpoint/cp10.pth and confirm both checkpoints can be loaded in the selected Hot-Ham environment. No fit-accuracy threshold is required.

Use each example’s own dataset and configuration (epoch: 10, checkpoint_interval: 5). This checks execution and checkpoint output; it does not certify convergence or scientific accuracy.

Save source SHA, environment/device, commands, exit codes, outputs and assessment in the external work directory. Missing prerequisites: blocked; command/acceptance failure: failed; unexecuted: not_run; passed requires actual output assessment.
