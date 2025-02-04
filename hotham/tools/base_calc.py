import torch
from entrypoints.model import Model
from tools.input_calc import Input_Calc


class Base_Calc(object):
    def __init__(self, para: dict, **kwargs):
        # Read calculation parameters
        self.para = Input_Calc(para=para)

        self.intdtype = self.para.intdtype
        self.floatdtype = self.para.floatdtype
        self.device = self.para.device

        # Load model
        self.model = Model(self.para)
        if self.para.init_from_checkpoint is not None:
            checkpoint = torch.load(self.para.init_from_checkpoint, map_location=torch.device(self.device))
            checkpoint_weights = self.Version_Convertion(checkpoint['model_state_dict'], self.model.state_dict())
            current_state = self.model.state_dict()
            current_state.update(checkpoint_weights)
            self.model.load_state_dict(current_state)
        self.model.to(self.device)
        self.model.eval()

    def Version_Convertion(self, old_version: dict, current_version: dict):
        Name_Convertion = {"FourierConv_layer": "GauntConv_layer",
                           "ftp": "gtp",
                           "weight_ftp": "weight_gtp"}
        old_keys = old_version.keys()
        current_keys = current_version.keys()
        for key in old_keys:
            if key not in current_keys:
                words = key.split(".")
                new_words = []
                for word in words:
                    word = Name_Convertion.get(word, word)
                    new_words.append(word)
                new_key = ".".join(new_words)
                assert new_key in current_keys
                current_version[new_key] = old_version[key]
            else:
                current_version[key] = old_version[key]

        for key in current_keys:
            if "N_average" in key:
                current_version[key] = old_version["N_average"]

        return current_version


if __name__ == "__main__":
    import sys
    import os
    import json5
    assert os.path.exists(sys.argv[1])
    with open(sys.argv[1], "r") as f:
        inputfile = json5.load(f)
    para = Base_Calc(inputfile)
