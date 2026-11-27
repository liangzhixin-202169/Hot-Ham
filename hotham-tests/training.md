# Training
To train a Hot-Ham model, prepare dataset and a input file.

We provide 2 examples for training LCAO Hamiltonian and orthogonal tight-binding Hamiltonian, separately.

## 1. Prepare input file
You should first set the parameters in a input file like `hotham-input.yaml`.

Here are some basic parameters you need to know when training a model, see the [parameter list](parameters.md) for more detailed parameters and defaults.

### Shared
- `trainset`: path to the training `*.pth` file.
- `testset`: optional path to the testing `*.pth` file; omit it to skip test evaluation during training.
- `hamiltonian_type`: `lcao` or `orthtb`.
- `prediction`: set to `0` for training.
- `orbit`: use all orbitals for LCAO, select orbitals for OrthTB.
- `rc`: Radial embedding cutoff in Å. Set it slightly larger than twice the largest orbital cutoff radius among all elements in the system.
- `L_max`: maximum angular momentum order of model features. Choose it according to the highest orbital in `orbit` and the SOC/magnetism settings; see the [parameter list](parameters.md#basic-parameters-shared).
- `lr`: initial learning rate.
- `epoch`: number of epochs in this run.
- `model_save_path`: directory for model outputs.
- `checkpoint_interval`: number of epochs between checkpoints.

### LCAO only
- `magnetism`: `nonmag`, `collinear`, or `non-collinear`, matching the dataset.
- `soc`: `true` or `false`, matching the dataset's spin-orbit coupling setting.
- `first_channel`: initial feature-channel count.
- `mid_channel`: message-passing channel sizes.

### OrthTB only
- `band_range`: band index interval `[start, end)` used for fitting.
- `e_range`: energy window in eV used for fitting.
- `using_element_net`: enable the element network; the example uses `true`.
- `using_soc`: enable onsite spin-orbit coupling; the example uses `false`.
- `using_dosloss`: include the DOS loss; the example uses `false`. If enabled, also set `dos_sigma` and `dos_weight`.
- `delta`: required to initialize the Huber loss; the current training loss does not use it. The example sets it to `0.1`.
- `Ce`: even-parity channels at each layer.
- `Co`: odd-parity channels at each layer; the list must have the same length as `Ce`.

## 2. Start training
After preparing input file, submit Hot-Ham job with `torchrun`.

```
torchrun --nproc_per_node="$NUM_GPUS" --master_port="$MASTER_PORT" -m hotham.entrypoints.main hotham-input.yaml > log.txt
```

## 3. Resume training
To resume training, keep the dataset and model settings consistent with the checkpoint, keep `prediction: 0` and set these:
- `init_from_checkpoint`: path to the checkpoint to resume from.
- `epoch`: number of additional epochs to train.
- `new_lr`: optional; overrides the learning rate saved in the checkpoint.

## 4. Examples
| Example | Hamiltonian type | Training target | Provided files |
| --- | --- | --- | --- |
| [`example_lcao_spin1`](training/example_lcao_spin1/) | LCAO | Hamiltonian | `trainset.pth`, `hotham-input.yaml`, `jobfile-hotham` |
| [`example_tb`](training/example_tb/) | Orthogonal tight-binding | Bands | `trainset.pth`, `hotham-input.yaml`, `jobfile-hotham` |

1. Enter the example directory and review `hotham-input.yaml` and `jobfile-hotham`.
2. Submit the training job with `sbatch jobfile-hotham`.
3. After the job finishes, check `log.txt` and the saved checkpoints in `model_save_path/checkpoint/` (`./checkpoint/` in these examples).

## 5. Advanced Tricks: Residual Hamiltonian Training
For LCAO models, the training target can be the difference between the converged Hamiltonian and a baseline Hamiltonian for the same structure, `ΔH = H − H₀`. During reconstruction, add `H₀` to the predicted `ΔH` to recover the full Hamiltonian.

To prepare `H₀`, start from the initial charge density and limit the SCF run to one iteration. Use `scf.maxIter 1` in OpenMX or `scf_nmax 1` in ABACUS. Keep the structure, basis, and other Hamiltonian settings consistent with the converged calculation. The baseline run only needs to produce the Hamiltonian, so SCF convergence is unnecessary.

- **OpenMX** — Place the converged `hamiltonian.txt` and the exported h0 baseline, renamed to `h0.txt`, in the same directory. One subtraction will be completed.
- **ABACUS** — Place the converged `data-HR-sparse_SPIN0.csr` and the exported h0 baseline, renamed to `data-H0R-sparse_SPIN0.csr` in the same directory. One subtraction will be completed. For systems with `SPIN=2`, you will also need `data-HR-sparse_SPIN1.csr` and `data-H0R-sparse_SPIN1.csr`.

- Set `residual_H` to `True` in `generate_dataset.py` to generate the residual datasets.
- Use the same training procedure as for the full Hamiltonian, only with the residual datasets as input.
- For band calculations with `diagonal.py`, prepare the corresponding baseline CSR for the inference structure. For OpenMX, you will need [`h0.txt_to_csr.py`](../hotham/tools/h0.txt_to_csr.py) to convert `h0.txt` to CSR format.
