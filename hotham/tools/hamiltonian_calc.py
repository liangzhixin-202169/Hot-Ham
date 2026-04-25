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
            "AtomType": data.AtomType,
            "AtomType_OrbitalSum": self.model.AtomType_OrbitalSum,
            "offset": data.offset,
            "n_type": self.model.num_atomtype,
            "lattice": data.lattice,
            "pos": data.pos,
            "edge_index": data.edge_index_hop,
            "inv_edge_index": data.edge_inverse,
            "D": data.D_hop,
            "d": data.d_hop,
            "S": data.S_hop,
            "unique_cell_shift": data.unique_cell_shift,
            "cell_shift_index": data.cell_shift_index,
            "AtomSymbol_to_AtomType": self.model.AtomSymbol_to_AtomType,
            "AtomType_to_AtomSymbol": self.model.AtomType_to_AtomSymbol
        }

        if "HR" in data:
            save_dict["H_ref"] = data.HR
        if "SR" in data:
            save_dict["SR"] = data.SR

        data.to(self.device)
        HPred_block, _ = self.model(data)
        save_dict["H_pred"] = HPred_block

        save_dict = self.tensor2numpy(save_dict)
        np.save("HS.npy", save_dict)

    def run(self):
        with torch.no_grad():
            for batch_idx, data in enumerate(self.trainloader):
                self.calculation(data=data, structure_idx=batch_idx)


if __name__ == "__main__":
    pass
