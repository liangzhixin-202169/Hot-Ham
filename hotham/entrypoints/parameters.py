import ase.data
import torch
import ase
import numpy as np


# User-facing keys documented in hotham-skills/parameters.md. Internal keys such
# as num_types, rank, and N_average are calculated by the program.
INPUT_PARAMETERS = frozenset({
    "hamiltonian_type", "prediction", "device", "int_dtype", "float_dtype",
    "orbit", "basis", "basis_size", "Ce", "Co",
    "edge_include_sc", "L_max", "rc", "trainset",
    "testset", "batch_size", "shuffle", "optimizer", "lambda_2", "lr",
    "new_lr", "lr_scheduler", "factor", "patience", "threshold", "gamma",
    "model_save_path", "checkpoint_interval", "init_from_checkpoint",
    "epoch", "seed", "version", "model_type", "fix_average",
    "loss_abort_threshold", "refresh_o3_linear_in_eval", "restrict_last_out",
    "using_layernorm", "using_layernorm2", "using_rearrangement_linear",
    "magnetism", "spin", "first_channel", "mid_channel", "to3",
    "mag_cutoff", "reduced_to3", "self_interaction",
    "self_interaction_channel", "self_interaction_by_edge", "using_layernorm3",
    "using_longrange", "H_target", "band_range", "delta", "dos_sigma", "dos_weight",
    "dump_sparseH", "e_range", "k_gap", "nn_split_scale", "structures_path",
    "using_dosloss", "using_element_net", "using_soc", "weight_M", "weight_V",
    "work_ratio", "write",
})


def validate_input_parameters(para):
    if not isinstance(para, dict):
        raise ValueError("Input file must contain a mapping of parameters")
    unknown = sorted(set(para) - INPUT_PARAMETERS, key=str)
    if unknown:
        raise ValueError("Unknown input parameter(s): " + ", ".join(map(repr, unknown)))

    required = ["hamiltonian_type", "prediction", "orbit"]
    required_datasets = {
        ("lcao", 0): ("trainset", "testset"),
        ("lcao", 1): ("trainset", "testset"),
        ("lcao", 2): ("trainset",),
        ("orthtb", 0): ("trainset", "testset"),
        ("orthtb", 1): ("testset",),
        ("orthtb", 2): ("trainset",),
    }
    required.extend(required_datasets.get((para.get("hamiltonian_type"), para.get("prediction")), ()))
    if para.get("hamiltonian_type") == "orthtb" and para.get("prediction") in (0, 1):
        required.extend(("band_range", "e_range"))
    missing = [name for name in required if para.get(name) is None]
    if missing:
        raise ValueError("Missing required input parameter(s): " + ", ".join(missing))
    if para.get("hamiltonian_type") == "orthtb" and para.get("prediction") not in (0, 1, 2):
        raise ValueError("Invalid OrthTB prediction mode")
    if para.get("hamiltonian_type") == "lcao" and para.get("prediction") not in (0, 1, 2):
        raise ValueError("Invalid LCAO prediction mode")


class Parameters(dict):
    def __init__(self, para: dict):
        validate_input_parameters(para)
        super().__init__()
        self.update(self.set_default_parameters())
        self.update(para)
        self.convert()

        self.atomic_numbers = [ase.data.atomic_numbers[ele] for ele in self.orbit.keys()]
        self.num_types = len(self.atomic_numbers)

    def set_default_parameters(self):
        default_dict = {}

        default_dict["model_type"] = 0
        default_dict["batch_size"] = 1

        default_dict["int_dtype"] = "int64"
        default_dict["float_dtype"] = "float32"
        default_dict["device"] = "cuda"

        default_dict["edge_include_sc"] = True
        default_dict["shuffle"] = False

        # optimizer
        default_dict["optimizer"] = "AdamW"
        default_dict["lambda_2"] = 0
        default_dict["lr_scheduler"] = "ReduceLROnPlateau"
        default_dict["factor"] = 0.9
        default_dict["patience"] = 50
        default_dict["threshold"] = 0.5
        default_dict["gamma"] = 0.999  # only used by ExponentialLR

        # Random seed
        default_dict["seed"] = 42

        # other params
        default_dict["basis"] = "Gaussian"
        default_dict["basis_size"] = 32
        default_dict["init_from_checkpoint"] = None
        default_dict["new_lr"] = None
        default_dict["loss_abort_threshold"] = None
        default_dict["using_CoordinateTransformation"] = True
        default_dict["nn_split_scale"] = 1
        default_dict["fix_average"] = False
        default_dict["N_average"] = torch.tensor(-1.0)
        default_dict["using_layernorm"] = True
        default_dict["using_layernorm2"] = False
        default_dict["restrict_last_out"] = True
        default_dict["using_rearrangement_linear"] = True
        default_dict["refresh_o3_linear_in_eval"] = False

        return default_dict

    def convert(self):
        if self["int_dtype"] == "int32":
            self.numpy_int_dtype = np.int32
            self.torch_int_dtype = torch.int32
        elif self["int_dtype"] == "int64":
            self.numpy_int_dtype = np.int64
            self.torch_int_dtype = torch.int64
        if self["float_dtype"] == "float32":
            self.numpy_float_dtype = np.float32
            self.numpy_complex_dtype = np.complex64
            self.torch_float_dtype = torch.float32
            self.torch_complex_dtype = torch.complex64
        elif self["float_dtype"] == "float64":
            self.numpy_float_dtype = np.float64
            self.numpy_complex_dtype = np.complex128
            self.torch_float_dtype = torch.float64
            self.torch_complex_dtype = torch.complex128
        self["N_average"] = self["N_average"].to(dtype=self.torch_float_dtype)

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Input' object has no attribute '{key}'")

    def __getitem__(self, key):
        try:
            return super().__getitem__(key)
        except KeyError:
            if key in INPUT_PARAMETERS:
                raise ValueError(f"Missing required input parameter {key!r} for this run") from None
            raise

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == "__main__":
    pass
