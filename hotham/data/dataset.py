import os
from typing import Union
import torch
from ..entrypoints.parameters import Parameters

class HothamData(object):
    def __init__(self, para: Union[dict, Parameters], dataset):
        assert os.path.exists(dataset)
        if torch.__version__ >= "2.0.0":
            self.dataset = torch.load(dataset, weights_only=False)
        else:
            self.dataset = torch.load(dataset)
