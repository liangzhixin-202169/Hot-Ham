import torch


class Input_Calc(dict):
    def __init__(self, para: dict):
        super().__init__()
        self.update(self.set_default_parameters())
        self.update(para)
        self.convert()

    def set_default_parameters(self):
        default_dict = {}

        default_dict["edge_include_sc"] = True
        default_dict["using_CoordinateTransformation"] = True
        default_dict["split_stru"] = 0
        default_dict["fix_average"] = False
        default_dict["N_average"] = torch.tensor(-1.0)

        default_dict["model_path"] = "./model.pth"
        default_dict["ref_band"] = None

        default_dict["intdtype"] = torch.int32,
        default_dict["floatdtype"] = torch.float32,
        default_dict["device"] = "cpu"

        default_dict["band_energy_min"] = None
        default_dict["band_energy_max"] = None

        default_dict["testset"] = None
        default_dict["valset"] = None
        default_dict["dft"] = None
        default_dict["shuffle"] = False

        return default_dict

    def convert(self):
        self["intdtype"] = getattr(torch, self["intdtype"])
        self["floatdtype"] = getattr(torch, self["floatdtype"])
        self["complexdtype"] = getattr(torch, self["complexdtype"])
        self["batch_size"] = 1

    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(f"'Input' object has no attribute '{key}'")

    def __setattr__(self, key, value):
        self[key] = value


if __name__ == "__main__":
    import sys
    import json5
    with open(sys.argv[1], "r") as f:
        inputfile = json5.load(f)
    para = Input(inputfile)
