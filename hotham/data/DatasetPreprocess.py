import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from typing import Union
from ..entrypoints.Parameters import Parameters
from .Dataset import AbacusData, OpenmxData, HothamData


class DatasetPrepocess:
    def __init__(self, para: Union[dict, Parameters]):
        if para.dft == "abacus":
            DATACLASS = AbacusData
        elif para.dft == "openmx":
            DATACLASS = OpenmxData
        elif para.dft is None:
            DATACLASS = HothamData

        for dataset in ["trainset", "valset", "testset"]:
            if para[dataset] is not None:
                setattr(self, dataset, DATACLASS(para, para[dataset]).dataset)
            else:
                setattr(self, dataset, [])
            shuffle = para.shuffle if dataset == "trainset" else False
            if "local_rank" in para:
                shuffle = para.shuffle if dataset == "trainset" else False
                sampler = DistributedSampler(getattr(self, dataset),
                                             shuffle=shuffle,
                                             drop_last=False)
                loader = DataLoader(
                    getattr(self, dataset),
                    batch_size=para.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=0 if para.device.type == "cpu" else 4,
                    pin_memory=(para.device.type == "cuda")
                )
            else:
                loader = DataLoader(
                    getattr(self, dataset),
                    batch_size=para.batch_size,
                    shuffle=para.shuffle,
                    pin_memory=(para.device == "cuda")
                )
            loader_name = f"{dataset}_loader"
            setattr(self, loader_name, loader)

        if para.fix_average and para.prediction == 0:
            num_nodes, num_edges = 0, 0
            for data in self.trainset_loader:
                num_nodes += data.num_nodes
                num_edges += data.num_edges
            self.N_average = torch.tensor(num_edges/num_nodes)
            para.N_average = torch.tensor(num_edges/num_nodes)
        else:
            self.N_average = torch.tensor(-1.0)
            para.N_average = torch.tensor(-1.0)


if __name__ == "__main__":
    pass
