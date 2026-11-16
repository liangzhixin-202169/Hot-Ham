# Hot-Ham integration tests

All cases: not_run; this delivery prepares files only. Paths below are repository-relative. Each case README is a redirection to original code/resources, not a second test implementation.

| Function | Case directory under aimlab-tools/tests/ | Original resources |
| --- | --- | --- |
| hotham.dataset.lcao-abacus-h-spin1 | example-lcao-abacus-h-spin1 | hotham-tests/datasets/example_lcao_abacus_h_spin1 |
| hotham.dataset.lcao-abacus-s-spin1 | example-lcao-abacus-s-spin1 | hotham-tests/datasets/example_lcao_abacus_s_spin1 |
| hotham.dataset.lcao-openmx-h-spin1 | example-lcao-openmx-h-spin1 | hotham-tests/datasets/example_lcao_openmx_h_spin1 |
| hotham.dataset.lcao-openmx-s-spin1 | example-lcao-openmx-s-spin1 | hotham-tests/datasets/example_lcao_openmx_s_spin1 |
| hotham.dataset.orthtb-abacus-band | example-orthtb-abacus-band | hotham-tests/datasets/example_orthtb_abacus_band |
| hotham.dataset.orthtb-no-band | example-orthtb-no-band | hotham-tests/datasets/example_orthtb_no_band |
| hotham.dataset.orthtb-openmx-band | example-orthtb-openmx-band | hotham-tests/datasets/example_orthtb_openmx_band |
| hotham.train | train-example-lcao-spin1 | hotham-tests/training/example_lcao_spin1 |
| hotham.train | train-example-tb | hotham-tests/training/example_tb |
| hotham.infer | infer-example-lcao | hotham-tests/inference/example_lcao |

Read ../install.md and ../runtime-composition.md first. Dataset cases run on CPU after the required external calculation; supplied training and inference cases require a selected compatible GPU environment. Run in fresh external work directories, with bounded scheduler allocations selected during configuration. Preserve original examples; inspect their jobfiles but do not reuse host/queue paths verbatim.

Order: install/configure environment, run applicable dataset cases, both training cases, then inference and its single diagonalization. Training/inference use their bundled datasets; they do not depend on regenerating the dataset examples or on each other's checkpoints. Dataset acceptance loads actual trainset.pth and checks positions, finite values and the expected H/S/band/structure content. Training completes 10 epochs, saves readable checkpoints every 5 epochs and reports finite loss/metrics. Inference requires parseable, dimensionally consistent finite matrices and usable reconstruction outputs, without accuracy or visual-agreement thresholds. Per-case READMEs and manifest specify exact outputs and criteria.

Record each case's version, environment/device, commands, exit codes, logs and checks in its external work directory, using passed/failed/blocked/not_run. Absent optional engines may be not_run with reason; broken installed engines are blocked. Retain failures; do not auto-resubmit unknown jobs, replace datasets/models or claim scientific accuracy. Training is limited to 10 epochs; external-data calculations retain their required producer stages. No unrelated regression tests are included.
