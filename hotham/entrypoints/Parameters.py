import ase.data
import torch
import ase


class Parameters(dict):
    def __init__(self, para: dict):
        super().__init__()
        self.update(self.set_default_parameters())
        self.set_default_parameters()
        self.update(para)
        self.convert()

        self.atomic_numbers = [ase.data.atomic_numbers[ele] for ele in self.orbit.keys()]
        self.num_types = len(self.atomic_numbers)

    def set_default_parameters(self):
        default_dict = {}

        default_dict["model_type"] = 0
        default_dict["batch_size"] = 1
        default_dict["prediction"] = 0

        default_dict["intdtype"] = "int64"
        default_dict["floatdtype"] = "float32"
        default_dict["complexdtype"] = "complex64"
        default_dict["device"] = "cpu"
        default_dict["band_min"] = None
        default_dict["band_max"] = None
        default_dict["band_weight"] = None
        default_dict["band_zerosweight"] = None

        default_dict["model_init_path"] = None

        # Dataset path
        default_dict["trainset"] = None
        default_dict["valset"] = None
        default_dict["testset"] = None
        default_dict["edge_include_sc"] = True
        default_dict["shuffle"] = True
        default_dict["dft"] = None

        # optimizer
        default_dict["lr"] = 0.05
        default_dict["optimizer"] = "AdamW"
        default_dict["lambda_2"] = 0
        default_dict["lr_scheduler"] = "ReduceLROnPlateau"
        default_dict["factor"] = 0.9
        default_dict["patience"] = 50
        default_dict["threshold"] = 0.5
        default_dict["gamma"] = 0.999  # only used by ExponentialLR
        default_dict["new_lr"] = None

        # Model init and save
        default_dict["init_from_model"] = None
        default_dict["init_from_checkpoint"] = None
        default_dict["save_interval"] = 100
        default_dict["checkpoint_interval"] = 100

        # other params
        default_dict["using_CoordinateTransformation"] = True
        default_dict["split_stru"] = 0
        default_dict["Co"] = None
        default_dict["fix_average"] = False
        default_dict["N_average"] = torch.tensor(-1.0)
        default_dict["using_layernorm"] = True
        default_dict["using_layernorm2"] = False
        default_dict["restrict_last_out"] = True
        default_dict["using_rearrangement_linear"] = True

        return default_dict

    def convert(self):
        self["intdtype"] = getattr(torch, self["intdtype"])
        self["floatdtype"] = getattr(torch, self["floatdtype"])
        self["complexdtype"] = getattr(torch, self["complexdtype"])

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Input' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == "__main__":
    pass
