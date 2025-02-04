import torch
from torch_geometric.loader import DataLoader
from typing import Union
from entrypoints.Parameters import Parameters
from .Dataset import AbacusData, OpenmxData


class DatasetPrepocess:
    def __init__(self, para: Union[dict, Parameters]):
        if para.dft == "abacus":
            DATACLASS = AbacusData
        elif para.dft == "openmx":
            DATACLASS = OpenmxData

        for dataset in ["trainset", "valset", "testset"]:
            if para[dataset] is not None:
                setattr(self, dataset, DATACLASS(para, para[dataset]).dataset)
            else:
                setattr(self, dataset, [])
            loader = DataLoader(getattr(self, dataset), batch_size=para.batch_size, shuffle=para.shuffle)
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
