# Parameter list
Here are the parameters and defaults used in `hotham-input.yaml`. `—` means the parameter has no default and must be supplied when needed.

For other parameters used in specific `*.py` files, check the table below.

| Script | Documentation |
| ---| --- |
| `generate_dataset.py` | [Parameters in generate_dataset.py](datasets.md#3-generate-datasets) |
| `diagonal.py` | [Parameters in diagonal.py](inference.md#3-calculate-the-band-structure) |

### Basic parameters (Shared)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| `hamiltonian_type` | — | str | Hamiltonian model: `lcao` or `orthtb`. |
| `prediction` | — | int | Run mode: `0` trains, `1` evaluates, `2` exports predicted matrices, `-1` profiles(dev only). For `1`, each provided dataset is evaluated separately. |
| `device` | `cuda` | str | `cpu` or `cuda`. |
| `int_dtype` | `int64` | str | `int32` or `int64`. |
| `float_dtype` | `float32` | str | `float32` or `float64`. Complex dtype follows it. |
| `orbit` | — | dict | LCAO: all orbitals of the DFT numerical basis. <br>OrthTB: High-principal-quantum-number orbitals used by the TB model. |
| `basis` | `Chebyshev` | str | Radial distance embedding type. Another choice is `Gaussian`. |
| `basis_size` | `32` | int | Number of radial embedding features. |
| `L_max` | — | int | Maximum angular-momentum order of equivariant features, must >= 2*l for the highest orbital in `orbit`.<br>For example, if `f`(l=3) is in `orbit`, use `L_max >= 6`.<br>The `L_max` for dataset must >= the `L_max` for training. |
| `rc` | — | float | Radial embedding cutoff in Å for both models. Set it slightly larger than twice the largest orbital cutoff radius among all elements in the system.<br>OrthTB also uses it as the neighbor cutoff when generating graph edges. |
| `nn_split_scale` | `1` | int | Number of edge chunks used in convolution; `1` disables splitting. |

### Training parameters (Shared)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| **Datasets** |  |  |  |
| `trainset` | — | str | Path for `*.pth` |
| `testset` | `None` | str or None | Path for `*.pth` |
| `shuffle` | `False` | bool | Whether to shuffle training data |
| **Optimizer** |  |  |  |
| `optimizer` | `AdamW` | str | NN optimizer |
| `lambda_2` | `0` | float | Weight decay strength for optimizer. |
| `lr` | — | float | Initial training learning rate. |
| `new_lr` | `None` | float | Optional learning rate override when resuming; otherwise use the checkpoint value. |
| **Learning-rate scheduler** |  |  |  |
| `lr_scheduler` | `ReduceLROnPlateau` | str | Learning-rate scheduler. Another choice is `ExponentialLR` |
| `factor` | `0.9` | float | Learning-rate reduction factor for `ReduceLROnPlateau`. |
| `patience` | `50` | int | Patience for `ReduceLROnPlateau`. |
| `threshold` | `0.5` | float | Improvement threshold for `ReduceLROnPlateau`. |
| `gamma` | `0.999` | float | Decay factor for `ExponentialLR`. |
| **Checkpoints** |  |  |  |
| `model_save_path` | — | str | Path to save checkpoint |
| `checkpoint_interval` | — | int | Save a checkpoint every this many epochs and at the end of training. |
| `init_from_checkpoint` | `None` | str | model path used both for resuming training and inference |
| **Training run** |  |  |  |
| `epoch` | — | int | Number of epochs in this train. |
| `seed` | `42` | int | Seed for Python, NumPy, and PyTorch |

### Training parameters (LCAO)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| **Switchs**  |   |   |   |
| `lcaoversion` | `0` | int | `0` for the default TO2 convolution, `-1` for the legacy Gaunt+SO2 convolution (dev only, some parameters below may not apply, no mag, no soc), `1` for the faster TO2 convolution (in dev) |
| `magnetism` | — | str | `nonmag`, `collinear`, or `non-collinear`. |
| `soc` | — | bool | whether to include spin-orbit coupling. |
| `self_interaction_by_edge` | `False` | bool | Additional feature generation.<br>Set to `True` for slower computation with higher parameters. |
| `longrange` | `False` | bool | Enable long-range LCAO message-passing blocks (in dev). |
| `using_layernorm` | `2` | int | Layer normalization methods: `2` centers every representation and scales only scalars; `3` centers and scales every representation. |
| **Model** |  |  |  |
| `mag_cutoff` | — | list | Cutoff radius for the magnetic embedding, needed when `magnetism` is not `nonmag`. |
| `first_channel` | — | int | Initial feature-channel count in the current LCAO model. |
| `mid_channel` | — | list | Required LCAO message-passing channel sizes. |
| **Other** |  |  |  |
| `fix_average` | `False` | bool | Neighbor aggregation: `False` averages each node’s neighbor features; `True` divides their sum by the average number of edges per node in the training set. |

### Training parameters (OrthTB)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| `Ce` | — | list | even-parity feature channels for each layer |
| `Co` | — | list | odd-parity feature channels for each layer |
| `band_range` | — | list | To be written. |
| `e_range` | — | list | To be written. |
| `delta` | — | float | To be written |
| `dos_sigma` | — | float | To be written |
| `dos_weight` | — | float | To be written |
| `dump_sparseH` | — | str | To be written |
| `using_dosloss` | — | bool | To be written |
| `using_element_net` | — | bool | To be written |
| `using_soc` | — | bool | To be written |

### Inference parameters (Shared)
| Parameter | Default | Type | Description |
| --- | --- | --- | --- |
| `write` | — | list | Selects outputs for `prediction: 2`. LCAO keys: `h_pred`, `h_ref`, `olp`, `rR`. OrthTB keys: `h_pred`, `olp`, `rR`, `opos`. |
