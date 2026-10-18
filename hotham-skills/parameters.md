# Parameter list

Defaults below come from `hotham/entrypoints/parameters.py`; `—` means the parameter must be supplied when its run mode uses it.

### Basic parameters (Shared)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| `hamiltonian_type` | — | str | Hamiltonian model: `lcao` or `orthtb`. |
| `prediction` | — | int | Run mode: `0` trains, `1` evaluates, `2` exports predicted matrices. |
| `device` | `"cuda"` | str | Compute device, such as `cpu` or `cuda`. |
| `int_dtype` | `"int64"` | str | `int32` or `int64`. |
| `float_dtype` | `"float32"` | str | `float32` or `float64`; sets the matching complex dtype automatically. |
| `orbit` | — | dict | LCAO: include all orbitals of the DFT numerical basis in matrix order. <br>OrthTB: select the high-principal-quantum-number orbitals used by the TB model.<br>Keep the list and order consistent across dataset generation, training, and inference.|
| `basis` | `"Gaussian"` | str | Radial distance embedding type. |
| `basis_size` | `32` | int | Number of radial embedding features. |
| `Ce` | — | list | Even-parity feature channels at each network layer; unused in LCAO version 2. |
| `Co` | — | list | Odd-parity feature channels at each network layer; unused in LCAO version 2. |
| `edge_include_sc` | `True` | bool | Controls onsite self-interaction edges and their Wigner matrices in data generation. |
| `L_max` | — | int | Maximum angular-momentum order of equivariant features. |
| `rc` | — | float | Radial embedding cutoff in Å for both models; OrthTB also uses it as the neighbor cutoff when generating graph edges. |
| `nn_split_scale` | `1` | int | Number of edge chunks used by Gaunt, Fourier, and SO2 convolutions; `1` disables splitting. |

Use the same `int_dtype` and `float_dtype` for training and inference with a checkpoint.

### Training parameters (Shared)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| **Datasets and batching** | | | |
| `trainset` | — | str | Path for `*.pth`; required for LCAO runs and OrthTB training or sparse export. |
| `testset` | — | str | Path for `*.pth`; required for LCAO training or evaluation and OrthTB training or evaluation. |
| `batch_size` | `1` | int | NN batch_size|
| `shuffle` | `False` | bool | Whether to shuffle training data |
| **Optimizer** | | | |
| `optimizer` | `"AdamW"` | str | NN optimizer|
| `lambda_2` | `0` | float | Weight decay for the optimizer. |
| `lr` | — | float | Initial training learning rate. |
| `new_lr` | `None` | float | Optional learning rate override when resuming; otherwise use the checkpoint value. |
| **Learning-rate scheduler** | | | |
| `lr_scheduler` | `"ReduceLROnPlateau"` | str | Learning-rate scheduler. Another choice is `ExponentialLR` |
| `factor` | `0.9` | float | Learning-rate reduction factor for `ReduceLROnPlateau`. |
| `patience` | `50` | int | Patience for `ReduceLROnPlateau`. |
| `threshold` | `0.5` | float | Improvement threshold for `ReduceLROnPlateau`. |
| `gamma` | `0.999` | float | Decay factor for `ExponentialLR`. |
| **Checkpoints** | | | |
| `model_save_path` | — | str | Directory for model and checkpoint |
| `checkpoint_interval` | — | int | Save every this many epochs and at the end of the run. |
| `init_from_checkpoint` | `None` | str | model path used both for resuming training and inference |
| **Training run** | | | |
| `epoch` | — | int | Number of epochs to train in this run. |
| `seed` | `42` | int | Seed for PyTorch, NumPy, and Python |

### Training parameters (LCAO)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| **Switches** | | | |
| `version` | — | int | `1` for non-mag, `2` for mag(in dev) |
| `model_type` | `0` | int | Selects the Gaunt-convolution implementation (`0` or `1`) for LCAO version 1; unused in version 2. |
| `spin` | — | bool | Enable spinful LCAO model and loss handling. |
| `magnetism` | — | bool | Enable the magnetic message-passing branch of the LCAO model. |
| **Other** | | | |
| `fix_average` | `False` | bool | Use the average number of edges per node in the training set to normalize neighbor aggregation. |
| `loss_abort_threshold` | `None` | float | LCAO training-loss threshold that aborts an unstable run. |
| `refresh_o3_linear_in_eval` | `False` | bool | Refresh LCAO O(3) linear layers during evaluation. |
| `restrict_last_out` | `True` | bool | Limit the last LCAO feature layer to output-supported angular orders. |
| `using_layernorm` | `True` | bool | Enable LCAO layer normalization. |
| `using_layernorm2` | `False` | bool | Enable alternate convolution layer normalization. |
| `using_rearrangement_linear` | `True` | bool | Use rearranged output linear layers. |

### Training parameters (LCAO in dev)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| **Model channels** | | | |
| `first_channel` | — | int | Initial feature-channel count in the version-2 LCAO model. |
| `mid_channel` | — | list | Magnetic message-passing channel sizes in the version-2 LCAO model. |
| `to3` | — | list | Magnetic message-passing channel sizes in the version-1 LCAO model. |
| **Magnetic message passing** | | | |
| `mag_cutoff` | — | list | Cutoff radius for the magnetic embedding. |
| `reduced_to3` | — | bool | Reduce magnetic TO3 irreducible representations to output-compatible ones. |
| `self_interaction` | — | bool | Enable self-interaction blocks in magnetic message passing. |
| `self_interaction_channel` | — | list | Per-layer channel counts for magnetic self-interaction. |
| `self_interaction_by_edge` | — | bool | Enable edge-based self-interaction in magnetic message passing. |
| `using_layernorm3` | — | bool | Enable magnetic message-passing layer normalization. |
| `using_longrange` | — | bool | Enable long-range magnetic message-passing blocks. |
| **Loss** | | | |
| `H_target` | — | str | Spinful LCAO Hamiltonian loss target: `real`, `imag`, or `complete`. |

### Training parameters (OrthTB)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| `band_range` | — | list[int] | Band index interval `[start, end)` used by the band and DOS losses during training and evaluation; required for OrthTB modes `0` and `1`. |
| `e_range` | — | list[float] | Energy window `[min, max]` in eV used by the band loss during training and evaluation; required for OrthTB modes `0` and `1`. |
| `delta` | — | float | To be written |
| `dos_sigma` | — | float | Gaussian broadening width for the DOS loss when `using_dosloss` is enabled. |
| `dos_weight` | — | float | To be written |
| `dump_sparseH` | — | str | To be written |
| `structures_path` | — | str | To be written |
| `using_dosloss` | — | bool | To be written |
| `using_element_net` | — | bool | To be written |
| `using_soc` | — | bool | Enable onsite atomic SOC in OrthTB training, evaluation, and sparse Hamiltonian export. Nonmagnetic materials only, with `s`, `p`, and `d` orbitals. |
| `weight_M` | — | float | To be written |
| `weight_V` | — | float | To be written |

### Inference parameters (Shared)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| `write` | — | list | Selects outputs for `prediction: 2`. <br>LCAO keys: `h_pred`, `h_ref`, `olp`, `rR`. OrthTB keys: `h_pred`, `olp`, `rR`, `opos`. |

### Other parameters in .py files
| Script | Documentation |
| --- | --- |
| `generate_dataset.py` | [Parameters in generate_dataset.py](datasets/README.md#3-generate-a-dataset) |
| `diagonal.py` | [Parameters in diagonal.py](inference/README.md#3-calculate-the-band-structure) |
