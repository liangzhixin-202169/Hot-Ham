import torch
from torch_geometric.data import Data
import h5py
import os
from tools.base_calc import Base_Calc
from data.DatasetPreprocess import DatasetPrepocess


class Hamiltonian_Calc(Base_Calc):
    def __init__(self, para: dict, **kwargs):
        super().__init__(para, **kwargs)

        self.datapreprocess = DatasetPrepocess(self.para)
        self.trainloader = self.datapreprocess.trainset_loader

    def calculation(self, data: Data, structure_idx: int):
        num_atomtype = torch.max(data.AtomType)+1
        Node_OrbitalSum = self.model.AtomType_OrbitalSum[data.AtomType]
        OrbitalSum = Node_OrbitalSum.sum().item()
        Node_offset1 = torch.cat((torch.tensor([0], dtype=self.intdtype, device=data.AtomType.device), torch.cumsum(Node_OrbitalSum, dim=0)[:-1]), dim=0)

        HPred_block, self.GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
        if hasattr(data, "SR"):
            has_olp = True
        else:
            has_olp = False

        if hasattr(data, "HR"):
            has_href = True
        else:
            has_href = False
        Matrix = [[] for _ in range(num_atomtype**2)]
        for atomtype_1 in range(num_atomtype):
            for atomtype_2 in range(num_atomtype):
                index_HBlcok = atomtype_1*num_atomtype+atomtype_2
                if has_olp and has_href:
                    Matrix[index_HBlcok] = torch.stack([HPred_block[index_HBlcok], data.SR[index_HBlcok], data.HR[index_HBlcok]])
                elif has_olp:
                    Matrix[index_HBlcok] = torch.stack([HPred_block[index_HBlcok], data.SR[index_HBlcok]])
                else:
                    Matrix[index_HBlcok] = HPred_block[index_HBlcok][None, ...]

        os.makedirs(self.para.output_dir, exist_ok=True)
        with h5py.File(os.path.join(self.para.output_dir, f"Matrix-{structure_idx}.hdf5"), "w") as fp:
            fp.create_dataset("num_atomtype", data=num_atomtype.item())
            fp.create_dataset("num_edge", data=self.GraphEdgeIndex_to_BlockEdgeIndex.shape[0])
            fp.create_dataset("dim_matrix", data=OrbitalSum)
            fp.create_dataset("AtomType", data=data.AtomType.to("cpu").numpy())
            fp.create_dataset("Node_offset", data=Node_offset1.to("cpu").numpy())
            fp.create_dataset("index_edge", data=torch.cat([data.edge_index_hop.T, data.S_hop], dim=-1).to("cpu").numpy())
            for atomtype_1 in range(num_atomtype):
                for atomtype_2 in range(num_atomtype):
                    index_HBlcok = atomtype_1*num_atomtype+atomtype_2
                    fp.create_dataset(f"bondtype_matrix-{index_HBlcok}", data=Matrix[index_HBlcok].to("cpu").numpy())

    def run(self):
        with torch.no_grad():
            for batch_idx, data in enumerate(self.trainloader):
                self.calculation(data=data, structure_idx=batch_idx)


if __name__ == "__main__":
    import sys
    import json5
    from time import time

    def device_synchronize(input: dict):
        if input["device"] == "cuda":
            torch.cuda.synchronize()

    assert os.path.exists(sys.argv[1])
    with open(sys.argv[1], "r") as f:
        inputfile = json5.load(f)

    device_synchronize(input)
    time_begin = time()
    hamil_calc = Hamiltonian_Calc(inputfile)
    device_synchronize(input)
    time_finish = time()
    print("-"*50+f"\nTime used for initialization = {time_finish-time_begin:.3f} s.\n"+"-"*50)

    device_synchronize(input)
    time_begin = time()
    hamil_calc.run()
    time_finish = time()
    print("-"*50+f"\nTime used for hamiltonian calculation = {time_finish-time_begin:.3f} s.\n"+"-"*50)
