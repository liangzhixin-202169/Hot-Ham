import torch
from torch_geometric.data import Data
import numpy as np
import os
from .base_calc import Base_Calc
from ..data.DatasetPreprocess import DatasetPrepocess


class Hamiltonian_Calc(Base_Calc):
    def __init__(self, para: dict, **kwargs):
        super().__init__(para, **kwargs)

        self.datapreprocess = DatasetPrepocess(self.para)
        self.trainloader = self.datapreprocess.trainset_loader

    def calculation(self, data: Data, structure_idx: int):
        save_dict = {
            "AtomType": data.AtomType.to("cpu").numpy(),
            "AtomType_OrbitalSum": self.model.AtomType_OrbitalSum.to("cpu").numpy(),
            "offset": data.offset.to("cpu").numpy(),
            "n_type": self.model.num_atomtype,
            "lattice": data.lattice.to("cpu").numpy(),
            "pos": data.pos.to("cpu").numpy(),
            "edge_index": data.edge_index_hop.to("cpu").numpy(),
            "inv_edge_index": data.edge_inverse.to("cpu").numpy(),
            "D": data.D_hop.to("cpu").numpy(),
            "d": data.d_hop.to("cpu").numpy(),
            "S": data.S_hop.to("cpu").numpy(),
            "unique_cell_shift": data.unique_cell_shift.to("cpu").numpy(),
            "cell_shift_index": data.cell_shift_index.to("cpu").numpy()
        }

        if "HR" in data:
            save_dict["H_ref"] = self.tensor2numpy(data.HR)
        if "SR" in data:
            save_dict["SR"] = self.tensor2numpy(data.SR)

        data.to(self.device)
        HPred_block, self.GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
        save_dict["H_pred"] = self.tensor2numpy(HPred_block)

        np.save("HS.npy", save_dict)

    def run(self):
        with torch.no_grad():
            for batch_idx, data in enumerate(self.trainloader):
                self.calculation(data=data, structure_idx=batch_idx)


if __name__ == "__main__":
    pass
