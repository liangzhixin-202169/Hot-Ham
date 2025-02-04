import torch
from Parameters import Parameters


class LossRecord():
    def __init__(self):
        self.__mse = 0.
        self.__mae = 0.
        self.__num_ele = 0.
        self.__mse_max = 0.
        self.__mae_max = 0.
        self.__mse_min = 0.
        self.__mae_min = 0.

    def update(self, mse, mae, num_ele):
        self.__mse += mse*num_ele
        self.__mae += mae*num_ele
        self.__num_ele += num_ele
        self.__mse_max = max([self.__mse_max, mse])
        self.__mae_max = max([self.__mae_max, mae])
        self.__mse_min = min([self.__mse_min, mse]) if self.__mse_min != 0 else mse
        self.__mae_min = min([self.__mae_min, mae]) if self.__mae_min != 0 else mae

    @property
    def mse_ave(self):
        return self.__mse/self.__num_ele if self.__num_ele != 0 else 0

    @property
    def mae_ave(self):
        return self.__mae/self.__num_ele if self.__num_ele != 0 else 0

    @property
    def mse_max(self):
        return self.__mse_max

    @property
    def mae_max(self):
        return self.__mae_max

    @property
    def mse_min(self):
        return self.__mse_min

    @property
    def mae_min(self):
        return self.__mae_min

    def compute_info(self, title: str):
        if self.__num_ele == 0:
            info = ""
        else:
            info = f"   {title}_MSE: {self.mse_ave:.7f}   {title}_MAE: {self.mae_ave:.7f}"
        return info

    def eval_info(self, title: str):
        info = f"{title}:\n" +\
            f"    MSE(eV^2): {self.mse_ave:.7f}    MAX: {self.mse_max:.7f}    MIN: {self.mse_min:.7f}\n" +\
            f"    MAE(eV):   {self.mae_ave:.7f}    MAX: {self.mae_max:.7f}    MIN: {self.mae_min:.7f}\n"
        return info

    def reset(self):
        self.__mse = 0.
        self.__mae = 0.
        self.__num_ele = 0.
        self.__mse_max = 0.
        self.__mae_max = 0.
        self.__mse_min = 0.
        self.__mae_min = 0.


class Lossfunction(object):
    def __init__(self, para: Parameters):
        self.para = para
        self.device = para.device

    def trainloss_band(self, eig, eig_ref, band_weight):
        MSEloss = 0.0
        num_kn = 0
        for structure_index in range(len(eig)):
            e = eig[structure_index]
            e_ref = eig_ref[structure_index]
            diff2 = (e-e_ref)**2
            num_kn = +e.numel()
            MSEloss += torch.sum(diff2*band_weight)
        MSEloss = MSEloss/num_kn
        return MSEloss

    def testloss_band(self, eig, eig_ref):
        MSEloss, MAEloss = 0.0, 0.0
        num_kn = 0
        for structure_index in range(len(eig)):
            e = eig[structure_index]
            e_ref = eig_ref[structure_index]
            num_kn = +e.numel()
            MSEloss += torch.sum((e-e_ref)**2)
            MAEloss += torch.sum((e-e_ref).abs())
        MSEloss = MSEloss/num_kn
        MAEloss = MAEloss/num_kn
        return MSEloss.item(), MAEloss.item()

    def trainloss_ham(self, H_block, GraphEdgeIndex_to_BlockEdgeIndex, AtomType_OrbitalSum, data):
        num_atomtype = torch.max(data.AtomType)+1
        MSEloss, MAEloss = 0.0, 0.0
        num_ele = 0
        for atomtype_1 in range(num_atomtype):
            for atomtype_2 in range(num_atomtype):
                index_HBlcok = atomtype_1*num_atomtype+atomtype_2
                h_pred = H_block[index_HBlcok]
                h_ref = data.HR[index_HBlcok]

                MSEloss += torch.sum(torch.pow((h_ref-h_pred), 2))
                MAEloss += torch.sum(torch.abs(h_ref-h_pred))
                num_ele += h_pred.numel()

        return MSEloss/num_ele, MAEloss/num_ele, num_ele

    def testloss_ham(self, H_block, GraphEdgeIndex_to_BlockEdgeIndex, AtomType_OrbitalSum, data):
        num_atomtype = torch.max(data.AtomType)+1
        MSEloss, MAEloss = 0.0, 0.0
        num_ele = 0
        for atomtype_1 in range(num_atomtype):
            for atomtype_2 in range(num_atomtype):
                index_HBlcok = atomtype_1*num_atomtype+atomtype_2
                h_pred = H_block[index_HBlcok]
                h_ref = data.HR[index_HBlcok]

                MSEloss += torch.sum(torch.pow((h_ref-h_pred), 2))
                MAEloss += torch.sum(torch.abs(h_ref-h_pred))
                num_ele += h_pred.numel()

        return MSEloss/num_ele, MAEloss/num_ele, num_ele
