# Training
To train a Hot-Ham model, prepare dataset and a input file.

This directory contains examples for training LCAO Hamiltonian and Orthogonal tight-binding Hamiltonian.

**These examples have no practical application value.**

## 1. Prepare input file
You should first set the parameters in a input file.

### Shared
Set these in either training example:
- `hamiltonian_type`: `lcao` or `orthtb`.
- `prediction`: set to `0` for training.
- `trainset`: path to the training `*.pth` file.
- `testset`: path to the evaluation `*.pth` file.
- `int_dtype`: `int32` or `int64` for integer tensors in the main program.
- `float_dtype`: `float32` or `float64` for real tensors; complex precision follows automatically.
- `rc`: radial cutoff; for OrthTB, also the graph neighbor cutoff.
- `L_max`: maximum angular momentum order of model features.
- `Ce`: even-parity channels at each layer.
- `Co`: odd-parity channels at each layer.
- `lr`: initial learning rate.
- `epoch`: number of epochs in this run.
- `model_save_path`: directory for model outputs.
- `checkpoint_interval`: number of epochs between checkpoints.

See the [parameter list](../parameters.md) for defaults and additional settings.

### LCAO only
- `orbit`: all orbitals in the DFT basis, in the order used by the Hamiltonian matrices.
- `version`: LCAO model version; the example uses `1`.
- `spin`: whether to use the spinful model; the example uses `false`.
- `magnetism`: whether to use magnetic message passing; the example uses `false`.

### OrthTB
- `orbit`: higher-principal-quantum-number orbitals of the TB model, matching dataset generation.
- `band_range`: band index interval used for fitting.
- `e_range`: energy window in eV used for fitting.
- `using_element_net`: enable the element network; the example uses `true`.
- `using_soc`: enable onsite spin-orbit coupling; the example uses `false`.
- `using_dosloss`: include the DOS loss; the example uses `false`.
- `delta`: set for the system; the example uses `0.1`.

## 2. Start training
After preparing input file, submit Hot-Ham job with `torchrun`.

```
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" -m hotham.entrypoints.main hotham-input.yaml > log.txt
```

## 3. Resume training
To resume training, set:
- `prediction`: keep `0` for training.
- `init_from_checkpoint`: path to the checkpoint to resume from.
- `epoch`: number of **additional** epochs to train.
- `new_lr`: optional; overrides the learning rate saved in the checkpoint.

Keep the dataset and model settings consistent with the checkpoint.
