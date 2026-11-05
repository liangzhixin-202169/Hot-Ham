import ase.data
import torch
import ase
import numpy as np


# User-facing keys documented in hotham-skills/parameters.md. Internal keys such
# as num_types, rank, and N_average are calculated by the program.
INPUT_PARAMETERS = frozenset({
    "hamiltonian_type", "prediction", "device", "int_dtype", "float_dtype",
    "orbit", "basis", "basis_size", "Ce", "Co",
    "L_max", "rc", "trainset",
    "testset", "shuffle", "optimizer", "lambda_2", "lr",
    "new_lr", "lr_scheduler", "factor", "patience", "threshold", "gamma",
    "model_save_path", "checkpoint_interval", "init_from_checkpoint",
    "epoch", "seed", "fix_average",
    "magnetism", "soc", "first_channel", "mid_channel", "lcaoversion",
    "mag_cutoff", "using_layernorm", "self_interaction_by_edge",
    "longrange", "band_range", "delta", "dos_sigma", "dos_weight",
    "dump_sparseH", "e_range", "nn_split_scale",
    "using_dosloss", "using_element_net", "using_soc",
    "write",
})


def validate_input_parameters(para):
    if not isinstance(para, dict):
        raise ValueError("Input file must contain a mapping of parameters")
    unknown = sorted(set(para) - INPUT_PARAMETERS, key=str)
    if unknown:
        raise ValueError("Unknown input parameter(s): " + ", ".join(map(repr, unknown)))

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

        default_dict["lcaoversion"] = 0
        default_dict["self_interaction_by_edge"] = False
        default_dict["using_layernorm"] = 2
        default_dict["testset"] = None

        default_dict["int_dtype"] = "int64"
        default_dict["float_dtype"] = "float32"
        default_dict["device"] = "cuda"

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
        default_dict["basis"] = "Chebyshev"
        default_dict["basis_size"] = 32
        default_dict["init_from_checkpoint"] = None
        default_dict["new_lr"] = None
        default_dict["using_CoordinateTransformation"] = True
        default_dict["nn_split_scale"] = 1
        default_dict["fix_average"] = False
        default_dict["longrange"] = False
        default_dict["N_average"] = torch.tensor(-1.0)

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
