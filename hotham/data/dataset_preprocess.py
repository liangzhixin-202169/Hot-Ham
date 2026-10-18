import torch
from torch.utils.data.distributed import DistributedSampler
from torch_geometric.loader import DataLoader
from typing import Union
from ..entrypoints.parameters import Parameters
from .dataset import HothamData


class DatasetPrepocess:
    def __init__(self, para: Union[dict, Parameters]):
        for dataset in ["trainset", "testset"]:
            if para.get(dataset) is not None:
                setattr(self, dataset, HothamData(para, para[dataset]).dataset)
            else:
                setattr(self, dataset, [])
            shuffle = para.shuffle if dataset == "trainset" else False
            if "local_rank" in para:
                shuffle = para.shuffle if dataset == "trainset" else False
                sampler = DistributedSampler(getattr(self, dataset), shuffle=shuffle, seed=para.seed, drop_last=False)
                loader = DataLoader(
                    getattr(self, dataset),
                    batch_size=para.batch_size,
                    shuffle=False,
                    sampler=sampler,
                    num_workers=0,
                    pin_memory=(para.device.type == "cuda"),
                )
            else:
                loader = DataLoader(getattr(self, dataset), batch_size=para.batch_size, shuffle=shuffle, pin_memory=(para.device == "cuda"))
            loader_name = f"{dataset}_loader"
            setattr(self, loader_name, loader)

        if para.fix_average and para.prediction == 0 and para.hamiltonian_type == "lcao":
            num_nodes, num_edges = 0, 0
            for data in self.trainset_loader:
                num_nodes += data.num_nodes
                num_edges += data.num_edges
            self.N_average = torch.tensor(num_edges / num_nodes, dtype=para.torch_float_dtype)
            para.N_average = torch.tensor(num_edges / num_nodes, dtype=para.torch_float_dtype)
        else:
            self.N_average = torch.tensor(-1.0, dtype=para.torch_float_dtype)
            para.N_average = torch.tensor(-1.0, dtype=para.torch_float_dtype)


if __name__ == "__main__":
    pass
