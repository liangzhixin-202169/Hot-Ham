import torch
import torch.distributed as dist
from ..entrypoints.parameters import Parameters
from ..modules.common import BasicInfo

class LossRecord:
    def __init__(self):
        self.__mse = 0.0
        self.__mae = 0.0
        self.__num_ele = 0.0
        self.__mse_max = 0.0
        self.__mae_max = 0.0
        self.__mse_min = 0.0
        self.__mae_min = 0.0

    def update(self, mse, mae, num_ele):
        self.__mse += mse * num_ele
        self.__mae += mae * num_ele
        self.__num_ele += num_ele
        self.__mse_max = max([self.__mse_max, mse])
        self.__mae_max = max([self.__mae_max, mae])
        self.__mse_min = min([self.__mse_min, mse]) if self.__mse_min != 0 else mse
        self.__mae_min = min([self.__mae_min, mae]) if self.__mae_min != 0 else mae

    @property
    def mse_ave(self):
        return self.__mse / self.__num_ele if self.__num_ele != 0 else 0

    @property
    def mae_ave(self):
        return self.__mae / self.__num_ele if self.__num_ele != 0 else 0

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
        info = (
            f"{title}:\n"
            + f"    MSE(eV^2): {self.mse_ave:.7f}    MAX: {self.mse_max:.7f}    MIN: {self.mse_min:.7f}\n"
            + f"    MAE(eV):   {self.mae_ave:.7f}    MAX: {self.mae_max:.7f}    MIN: {self.mae_min:.7f}\n"
        )
        return info

    def reset(self):
        self.__mse = 0.0
        self.__mae = 0.0
        self.__num_ele = 0.0
        self.__mse_max = 0.0
        self.__mae_max = 0.0
        self.__mse_min = 0.0
        self.__mae_min = 0.0

    def global_loss(self, device, dtype):
        globel_info = torch.tensor([self.__mse, self.__mae, self.__num_ele], device=device, dtype=dtype)
        dist.all_reduce(globel_info, op=dist.ReduceOp.SUM)
        mse_global = globel_info[0] / globel_info[2]
        mae_global = globel_info[1] / globel_info[2]
        return mse_global, mae_global

    def global_max(self, device, dtype):
        global_info = torch.tensor([self.mse_max, self.mae_max], device=device, dtype=dtype)
        dist.all_reduce(global_info, op=dist.ReduceOp.MAX)
        return global_info

    def global_min(self, device, dtype):
        global_info = torch.tensor([self.mse_min, self.mae_min], device=device, dtype=dtype)
        dist.all_reduce(global_info, op=dist.ReduceOp.MIN)
        return global_info


class Lossfunction(object):
    def __init__(self, para: Parameters, basicinfo: BasicInfo = None):
        self.para = para
        self.device = para.device
        self.basicinfo = basicinfo

    # def trainloss_band(self, eig, eig_ref, band_weight):
    #     MSEloss = 0.0
    #     num_kn = 0
    #     for structure_index in range(len(eig)):
    #         e = eig[structure_index]
    #         e_ref = eig_ref[structure_index]
    #         diff2 = (e-e_ref)**2
    #         num_kn += e.numel()
    #         MSEloss += torch.sum(diff2*band_weight)
    #     MSEloss = MSEloss/num_kn
    #     return MSEloss

    # def testloss_band(self, eig, eig_ref):
    #     MSEloss, MAEloss = 0.0, 0.0
    #     num_kn = 0
    #     for structure_index in range(len(eig)):
    #         e = eig[structure_index]
    #         e_ref = eig_ref[structure_index]
    #         num_kn = +e.numel()
    #         MSEloss += torch.sum((e-e_ref)**2)
    #         MAEloss += torch.sum((e-e_ref).abs())
    #     MSEloss = MSEloss/num_kn
    #     MAEloss = MAEloss/num_kn
    #     return MSEloss.item(), MAEloss.item()

    def ham_loss(self, H_block, AtomType_OrbitalSum, data):
        num_atomtype = self.basicinfo.n_type
        unique_atomtypes = torch.unique(data.AtomType)
        MSEloss, MAEloss = 0.0, 0.0
        num_ele = 0
        abs_diff = []
        for atomtype_1 in range(num_atomtype):
            if atomtype_1 not in unique_atomtypes:
                continue
            atomsymbol_1 = self.basicinfo.AtomType_to_AtomSymbol[atomtype_1]
            for atomtype_2 in range(num_atomtype):
                if atomtype_2 not in unique_atomtypes:
                    continue
                atomsymbol_2 = self.basicinfo.AtomType_to_AtomSymbol[atomtype_2]
                h_pred = H_block[atomsymbol_1][atomsymbol_2]
                h_ref = data.HR[atomsymbol_1][atomsymbol_2]

                if not self.para.spin:
                    abs_diff.append(torch.abs(h_ref - h_pred).flatten())
                else:
                    if self.para.H_target == "real":
                        abs_diff.append(torch.abs(h_ref.real - h_pred.real).flatten())
                    elif self.para.H_target == "imag":
                        abs_diff.append(torch.abs(h_ref.imag - h_pred.imag).flatten())
                    elif self.para.H_target == "complete":
                        abs_diff.append(torch.abs(h_ref - h_pred).flatten())
                    # abs_real_diff = torch.abs(h_ref.real-h_pred.real).flatten()
                    # abs_imag_diff = torch.abs(h_ref.imag-h_pred.imag).flatten()
                    # abs_diff = torch.cat([abs_real_diff,abs_imag_diff])
                    # MSEloss += torch.sum(torch.pow(abs_diff, 2))
                    # MAEloss += torch.sum(abs_diff)
                    # num_ele += abs_diff.numel()

        abs_diff = torch.cat(abs_diff)
        MSEloss = torch.pow(abs_diff, 2).mean()
        MAEloss = abs_diff.mean()
        num_ele = abs_diff.numel()
        return MSEloss, MAEloss, num_ele
        # return MSEloss / num_ele, MAEloss / num_ele, num_ele

    # def testloss_ham(self, H_block, AtomType_OrbitalSum, data):
    #     num_atomtype = self.basicinfo.n_type
    #     unique_atomtypes = torch.unique(data.AtomType)
    #     MSEloss, MAEloss = 0.0, 0.0
    #     num_ele = 0
    #     for atomtype_1 in range(num_atomtype):
    #         if atomtype_1 not in unique_atomtypes:
    #             continue
    #         atomsymbol_1 = self.basicinfo.AtomType_to_AtomSymbol[atomtype_1]
    #         for atomtype_2 in range(num_atomtype):
    #             if atomtype_2 not in unique_atomtypes:
    #                 continue
    #             atomsymbol_2 = self.basicinfo.AtomType_to_AtomSymbol[atomtype_2]
    #             h_pred = H_block[atomsymbol_1][atomsymbol_2]
    #             h_ref = data.HR[atomsymbol_1][atomsymbol_2]
    #             if h_ref.shape[1] == 1:
    #                 h_ref = h_ref.squeeze(dim=1)

    #             MSEloss += torch.sum(torch.pow((h_ref-h_pred), 2))
    #             MAEloss += torch.sum(torch.abs(h_ref-h_pred))
    #             num_ele += h_pred.numel()

    #     return MSEloss/num_ele, MAEloss/num_ele, num_ele

    # import from GPUTB2
    def trainloss_band_new3(self, eig, eig_ref,band_range): 
        
        pred = eig[..., band_range[0]:band_range[1]]
        ref = eig_ref[..., band_range[0]:band_range[1]]
        #num_kn = pred.numel()
        
        pred_sorted, _ = torch.sort(pred, dim=-1)
        ref_sorted, _ = torch.sort(ref, dim=-1)
        
        loss_E = torch.nn.functional.mse_loss(pred_sorted, ref_sorted)
        
        grad_pred = torch.diff(pred, dim=-2) # 注意：这里用原始 pred，不 sort
        curv_pred = torch.diff(grad_pred, dim=-2)
        
        loss_V = torch.mean(curv_pred**2)

        
        return  loss_E+self.para.weight_V*loss_V

    def trainloss_band_new(self, eig, eig_ref,band_range): 
        
        pred = eig[..., band_range[0]:band_range[1]]
        ref = eig_ref[..., band_range[0]:band_range[1]]
        #num_kn = pred.numel()
        
        #pred_sorted, _ = torch.sort(pred, dim=-1)
        #ref_sorted, _ = torch.sort(ref, dim=-1)
        
        loss_E = torch.nn.functional.mse_loss(pred, ref)

        grad_pred = torch.diff(pred, dim=-2)
        grad_ref = torch.diff(ref, dim=-2)

        loss_V = torch.nn.functional.mse_loss(grad_pred, grad_ref)
        
        curv_pred = torch.diff(grad_pred, dim=-2)
        curv_ref = torch.diff(grad_ref, dim=-2)
        
        loss_M = torch.nn.functional.mse_loss(curv_pred, curv_ref)

    
        return  loss_E+self.para.weight_V*loss_V+self.para.weight_M*loss_M

    def trainloss_band(self, eig, eig_ref, band_range, e_range):
        e = eig[...,band_range[0]:band_range[1]]
        e_ref = eig_ref[...,band_range[0]:band_range[1]]
        
        
        diff2=((e-e_ref)**2)
        diff=(e-e_ref).abs()
        
        
        mask = (e_ref >= e_range[0]) & (e_ref <= e_range[1])
        diff2=diff2[mask]
        diff=diff[mask]
        num_kn = diff.numel()

        #kw=torch.ones(diff.shape[1],device=self.device)
        
        #kw[int(k_weight[0]):int(k_weight[1])]*=k_weight[-1]
        
        # diff=torch.einsum("ijk,j->ijk",diff,kw)
        # diff2=torch.einsum("ijk,j->ijk",diff2,kw)

        
        MSEloss = torch.sum(diff2)/num_kn
        MAEloss = torch.sum(diff)/num_kn
        

        #selected=((e-e_ref)**2).max(dim=2)[0].max(dim=1)[1].item()
        #MSEloss_s = torch.sum(diff2[:,selected])/e[:,1].numel()
        #MAEloss_s = torch.sum(diff[:,selected])/e[:,1].numel()

        return MSEloss, MAEloss

    def trainloss_band2(self, eig, eig_ref, band_range, e_range):
        # 1. 截取指定的 band 范围
        e = eig[..., band_range[0]:band_range[1]]
        e_ref = eig_ref[..., band_range[0]:band_range[1]]
        
        # 2. 创建掩码 (保持与 e 相同的形状)
        mask = (e_ref >= e_range[0]) & (e_ref <= e_range[1])
        
        # 3. 计算所有点的平方差和绝对差，不在 mask 范围内的点设为 0
        # 这样可以保持 (..., k_points, bands) 的维度结构
        diff2_all = ((e - e_ref) ** 2) * mask
        diff_all = ((e - e_ref).abs()) * mask
        
        # 计算原始的 MSE 和 MAE
        num_kn = mask.sum() 
        if num_kn == 0: # 防止除以 0
            return eig.new_tensor(0.0), eig.new_tensor(0.0)
        
        MSEloss = diff2_all.sum() / num_kn
        MAEloss = diff_all.sum() / num_kn

        # --- 核心逻辑：计算 k 点误差并提取前 5 ---
        
        # 假设 eig 的形状是 (batch, k_points, bands) 或 (k_points, bands)
        # 我们对最后一个维度 (bands) 求和，得到每个 k 点的总误差
        # k_error_sq 形状将变为 (batch, k_points) 或 (k_points)
        k_error_sq = diff2_all.sum(dim=-1) 
        k_error_abs = diff_all.sum(dim=-1)

        # 将其展平，以便在所有 batch 和 k 点中找最大的 5 个
        k_error_sq_flat = k_error_sq.view(-1)
        k_error_abs_flat = k_error_abs.view(-1)

        # 提取前 5 个最大的误差值
        # 使用 min(5, ...) 是为了防止总点数少于 5 的极端情况
        top_k_num = min(5, k_error_sq_flat.numel())
        top5_mse_values, _ = torch.topk(k_error_sq_flat, top_k_num)
        top5_mae_values, _ = torch.topk(k_error_abs_flat, top_k_num)

        # 4. 将 top5 的误差加回到原本的 loss 上
        # 注意：这里是直接加和还是按比例加，取决于你的需求。通常建议除以 num_kn 保持量级一致
        MSEloss += top5_mse_values.sum() / num_kn
        MAEloss += top5_mae_values.sum() / num_kn

        return MSEloss, MAEloss


    def testloss_band(self, eig, eig_ref,band_range,e_range):
        e = eig[...,band_range[0]:band_range[1]]
        e_ref = eig_ref[...,band_range[0]:band_range[1]]
        mask = (e_ref >= e_range[0]) & (e_ref <= e_range[1])
        
        diff2=((e-e_ref)**2)[mask]
        diff=(e-e_ref).abs()[mask]
        num_kn = diff.numel()
        MSEloss = torch.sum(diff2)/num_kn
        MAEloss = torch.sum(diff)/num_kn
        return MSEloss.item(), MAEloss.item()

    def delta(self, energy_x,energy,width):
        x = -((energy_x - energy) / width)**2
        return torch.exp(x) / (torch.sqrt(x.new_tensor(torch.pi)) * width)

    def get_dos_loss(self,width,npts,eigens,eigens_ref,band_range,device,need_dos=False):
        eigs_ref=eigens_ref[...,band_range[0]:band_range[1]]
        eigs=eigens[...,band_range[0]:band_range[1]]
        emin_ref,emax_ref,=eigs_ref.min(),eigs_ref.max()
        energy_x_ref = torch.linspace(emin_ref, emax_ref, npts, device=device, dtype=eigs_ref.dtype).reshape(-1, 1)
        dos = torch.zeros(npts, device=device,dtype=eigs_ref.dtype)
        dos_ref = torch.zeros(npts, device=device,dtype=eigs_ref.dtype)

        for e_k, e_k_ref in zip(eigs, eigs_ref):
            dos += self.delta(energy_x_ref,e_k,width).sum(axis=-1)
            dos_ref += self.delta(energy_x_ref,e_k_ref,width).sum(axis=-1)

        dos/=eigs.shape[0]
        dos_ref/=eigs.shape[0]

        MAE_loss = torch.mean((dos - dos_ref).abs())
        MSE_loss = torch.mean((dos - dos_ref) ** 2)

        # import matplotlib.pyplot as plt

        # plt.plot(energy_x_ref.to("cpu"), dos_ref.to("cpu").detach())
        # plt.plot(energy_x_ref.to("cpu"), dos.to("cpu").detach(), linestyle="--")

        # plt.show()
        if need_dos:
            return dos,dos_ref,MSE_loss,MAE_loss
        else:
            return MSE_loss,MAE_loss
