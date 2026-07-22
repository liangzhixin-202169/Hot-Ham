import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import sys
from ase.units import Rydberg, Bohr
from ..entrypoints.Parameters import Parameters
from ..entrypoints.Lossfun import LossRecord, Lossfunction
from ..data.DatasetPreprocess import DatasetPrepocess
from ..modules.common import BasicInfo


class Kernel(torch.nn.Module):
    def __init__(self, para: Parameters):
        super().__init__()
        self.para = para
        self.intdtype = self.para.intdtype
        self.floatdtype = self.para.floatdtype
        self.device = self.para.device

        # Basic atom type and orbit information
        self.basicinfo = BasicInfo(para.orbit, device=self.device, intdtype=self.intdtype)

        # Read dataset
        self.datapreprocess = DatasetPrepocess(para)
        self.trainloader = self.datapreprocess.trainset_loader
        self.valsetloader = self.datapreprocess.valset_loader
        self.testsetloader = self.datapreprocess.testset_loader

        # Initilize model, optimizer and lr_scheduler
        self.init_model()

        # Define loss function
        self.lossfunction = Lossfunction(para, self.basicinfo)
        self.train_lossrecord = LossRecord()
        self.val_lossrecord = LossRecord()
        self.test_lossrecord = LossRecord()

    def run(self):
        if self.para.prediction == 0:
            self.train()
        elif self.para.prediction == 1:
            self.eval()
        elif self.para.prediction == 2:
            self.profile()
        elif self.para.prediction == 3:
            self.topyatb()
        else:
            raise ValueError(f"Invalid prediction value: {self.para.prediction}. Must be 0, 1, 2, or 3.")

    def train(self):
        if self.para.rank == 0:
            if self.checkpoint_info is not None:
                print(self.checkpoint_info)

        for epoch in range(self.start_epoch, self.start_epoch+self.para.epoch):
            self.model.train()
            self.trainloader.sampler.set_epoch(epoch)
            self.train_lossrecord.reset()
            self.val_lossrecord.reset()
            self.test_lossrecord.reset()

            for data in self.trainloader:
                data.to(device=self.device)
                self.optimizer.zero_grad()
                H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                mse, mae, num_ele = self.lossfunction.trainloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.module.AtomType_OrbitalSum, data)
                (torch.sqrt(mse)+mae).backward()
                self.optimizer.step()
                self.train_lossrecord.update(mse.item(), mae.item(), num_ele)

            if self.para.lr_scheduler == "ExponentialLR":
                current_lr = self.lr_scheduler.get_last_lr()[0]
            elif self.para.lr_scheduler == "ReduceLROnPlateau":
                current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']

            mse_global, mae_global = self.train_lossrecord.global_loss(self.device)
            if self.para.rank == 0:
                info_global = f"Epoch:{epoch+1:>5}   lr: {current_lr:.6f}" + \
                    f"   Train_MSE: {mse_global:.7f}   Train_MAE: {mae_global:.7f}"

            if ((epoch+1) % self.para.checkpoint_interval == 0):
                self.model.eval()
                with torch.no_grad():
                    for data in self.valsetloader:
                        data.to(self.device)
                        H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                        val_loss_MSE, val_loss_MAE, num_ele = self.lossfunction.trainloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.module.AtomType_OrbitalSum, data)
                        self.val_lossrecord.update(val_loss_MSE.item(), val_loss_MAE.item(), num_ele)

                    mse_global, mae_global = self.val_lossrecord.global_loss(self.device)
                    if self.para.rank == 0:
                        info_global += f"   Val_MSE: {mse_global:.7f}   Val_MAE: {mae_global:.7f}"

                    for data in self.testsetloader:
                        data.to(self.device)
                        H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                        test_loss_MSE, test_loss_MAE, num_ele = self.lossfunction.trainloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.module.AtomType_OrbitalSum, data)
                        self.test_lossrecord.update(test_loss_MSE.item(), test_loss_MAE.item(), num_ele)

                    mse_global, mae_global = self.test_lossrecord.global_loss(self.device)
                    if self.para.rank == 0:
                        info_global += f"   Test_MSE: {mse_global:.7f}   Test_MAE: {mae_global:.7f}"

                if self.para.rank == 0:
                    self.save_checkpoint(epoch)

            if self.para.lr_scheduler == "ExponentialLR":
                self.lr_scheduler.step()
            elif self.para.lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler.step(self.train_lossrecord.mae_ave)

            if self.para.rank == 0:
                print(info_global)
                sys.stdout.flush()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            for data in self.trainloader:
                data.to(self.device)
                H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                train_loss_MSE, train_loss_MAE, num_ele = self.lossfunction.testloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.module.AtomType_OrbitalSum, data)
                self.train_lossrecord.update(train_loss_MSE.item(), train_loss_MAE.item(), num_ele)

            mse_global, mae_global = self.train_lossrecord.global_loss(self.device)
            mse_global_max, mae_global_max = self.train_lossrecord.global_max(self.device)
            mse_global_min, mae_global_min = self.train_lossrecord.global_min(self.device)
            if self.para.rank == 0:
                info_global = f"Train:\n" +\
                    f"    MSE(eV^2): {mse_global:.7f}    MAX: {mse_global_max:.7f}    MIN: {mse_global_min:.7f}\n" +\
                    f"    MAE(eV):   {mae_global:.7f}    MAX: {mae_global_max:.7f}    MIN: {mae_global_min:.7f}\n"

            for data in self.valsetloader:
                data.to(self.device)
                H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                val_loss_MSE, val_loss_MAE, num_ele = self.lossfunction.testloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.module.AtomType_OrbitalSum, data)
                self.val_lossrecord.update(val_loss_MSE.item(), val_loss_MAE.item(), num_ele)

            mse_global, mae_global = self.val_lossrecord.global_loss(self.device)
            mse_global_max, mae_global_max = self.val_lossrecord.global_max(self.device)
            mse_global_min, mae_global_min = self.val_lossrecord.global_min(self.device)
            if self.para.rank == 0:
                info_global += f"Val:\n" +\
                    f"    MSE(eV^2): {mse_global:.7f}    MAX: {mse_global_max:.7f}    MIN: {mse_global_min:.7f}\n" +\
                    f"    MAE(eV):   {mae_global:.7f}    MAX: {mae_global_max:.7f}    MIN: {mae_global_min:.7f}\n"

            for data in self.testsetloader:
                data.to(self.device)
                H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                test_loss_MSE, test_loss_MAE, num_ele = self.lossfunction.testloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.module.AtomType_OrbitalSum, data)
                self.test_lossrecord.update(test_loss_MSE.item(), test_loss_MAE.item(), num_ele)

            mse_global, mae_global = self.test_lossrecord.global_loss(self.device)
            mse_global_max, mae_global_max = self.test_lossrecord.global_max(self.device)
            mse_global_min, mae_global_min = self.test_lossrecord.global_min(self.device)
            if self.para.rank == 0:
                info_global += f"Test:\n" +\
                    f"    MSE(eV^2): {mse_global:.7f}    MAX: {mse_global_max:.7f}    MIN: {mse_global_min:.7f}\n" +\
                    f"    MAE(eV):   {mae_global:.7f}    MAX: {mae_global_max:.7f}    MIN: {mae_global_min:.7f}\n"

            if self.para.rank == 0:
                print(info_global)

    def profile(self):
        def trace_handler(p):
            output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
            print(output)
            p.export_chrome_trace((os.path.join(self.para.model_save_path, './profile.json')))

        skip_first = 10
        wait = 5
        warmup = 5
        active = 5
        repeat = 1

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA
            ],
            schedule=torch.profiler.schedule(
                skip_first=skip_first,
                wait=wait,
                warmup=warmup,
                active=active,
                repeat=repeat
            ),
            with_stack=False,
            on_trace_ready=trace_handler,
            record_shapes=True
        ) as prof:
            for _ in range((skip_first+wait+warmup+active)*repeat):
                for data in self.trainloader:
                    with torch.profiler.record_function("Model Forward"):
                        H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                    with torch.profiler.record_function("Compute Loss"):
                        mse, mae, num_ele = self.lossfunction.trainloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.AtomType_OrbitalSum, data)
                    with torch.profiler.record_function("Model Backward"):
                        (torch.sqrt(mse)+mae).backward()
                    with torch.profiler.record_function("Optim Step"):
                        self.optimizer.step()
                prof.step()
    
    def topyatb(self):
        if self.para.rank != 0:
            return
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.eval()
        if not hasattr(self.para, "write"):
            raise AttributeError("para.write is required for prediction == 3")
        with torch.no_grad():
            for batch_idx, data in enumerate(self.trainloader):
                AtomType = data.AtomType
                unique_cell_shift = data.unique_cell_shift.cpu().numpy()
                n_cell = len(unique_cell_shift)
                def tensor2tensor(obj, device):
                    if isinstance(obj, torch.Tensor):
                        return obj.to(device)
                    elif isinstance(obj, dict):
                        return {k: tensor2tensor(v, device) for k, v in obj.items()}
                    elif isinstance(obj, list):
                        return [tensor2tensor(v, device) for v in obj]
                    else:
                        return obj
                def get_wigner_D(order):
                    D = [
                        torch.tensor([[1.0]]),
                        torch.tensor([[0, 1, 0],
                                      [0, 0, -1],
                                      [-1, 0, 0]]),
                        torch.tensor([[0, 0, 1, 0, 0],
                                      [0, 0, 0, -1, 0],
                                      [0, -1, 0, 0, 0],
                                      [0, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 0]]),
                        torch.tensor([[0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, -1, 0, 0],
                                      [0, 0, -1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0],
                                      [0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, -1],
                                      [-1, 0, 0, 0, 0, 0, 0]])
                    ]
                    if isinstance(order, int):
                        order = [order]
                    return torch.block_diag(*[D[l] for l in order])

                def rotate2abacus(block, AtomType):
                    num_atomtype = model.num_atomtype
                    unique_atomtypes = torch.unique(AtomType)
                    AtomType_to_AtomSymbol = model.AtomType_to_AtomSymbol
                    AtomSymbol_to_AMList = model.AtomSymbol_to_AMList

                    for atomtype_1 in range(num_atomtype):
                        if atomtype_1 not in unique_atomtypes:
                            continue
                        atomsymbol_1 = AtomType_to_AtomSymbol[atomtype_1]
                        winger_D_1 = get_wigner_D(AtomSymbol_to_AMList[atomsymbol_1])

                        for atomtype_2 in range(num_atomtype):
                            if atomtype_2 not in unique_atomtypes:
                                continue
                            atomsymbol_2 = AtomType_to_AtomSymbol[atomtype_2]
                            winger_D_2 = get_wigner_D(AtomSymbol_to_AMList[atomsymbol_2])

                            block[atomsymbol_1][atomsymbol_2] = block[atomsymbol_1][atomsymbol_2].to(winger_D_1.dtype)
                            shape = block[atomsymbol_1][atomsymbol_2].shape
                            block[atomsymbol_1][atomsymbol_2] = block[atomsymbol_1][atomsymbol_2].reshape((-1,) + shape[-2:])
                            block[atomsymbol_1][atomsymbol_2] = torch.einsum(
                                "ij,zjk,kl->zil",
                                winger_D_1,
                                block[atomsymbol_1][atomsymbol_2],
                                winger_D_2.T
                            )
                            block[atomsymbol_1][atomsymbol_2] = block[atomsymbol_1][atomsymbol_2].reshape(shape)

                    return block

                def block_r(block, data):
                    num_atomtype = model.num_atomtype
                    AtomType = data.AtomType
                    unique_atomtypes = torch.unique(AtomType)
                    AtomType_to_AtomSymbol = model.AtomType_to_AtomSymbol
                    index_edge = data.edge_index_hop
                    num_edge = index_edge.shape[1]
                    unique_cell_shift_inner = data.unique_cell_shift
                    cell_shift_index = data.cell_shift_index
                    n_cell_inner = len(unique_cell_shift_inner)
                    offset = data.offset
                    AtomType_OrbitalSum = model.AtomType_OrbitalSum
                    dim_matrix = int(sum(AtomType_OrbitalSum[atomtype].item() for atomtype in AtomType))

                    Mr = torch.zeros((n_cell_inner, dim_matrix, dim_matrix), dtype=torch.float32)
                    EdgeNumber = torch.arange(num_edge, dtype=torch.long)

                    for atomtype_1 in range(num_atomtype):
                        if atomtype_1 not in unique_atomtypes:
                            continue
                        atomsymbol_1 = AtomType_to_AtomSymbol[atomtype_1]

                        for atomtype_2 in range(num_atomtype):
                            if atomtype_2 not in unique_atomtypes:
                                continue
                            atomsymbol_2 = AtomType_to_AtomSymbol[atomtype_2]

                            mr = block[atomsymbol_1][atomsymbol_2]
                            mr = mr.squeeze(1)
                            mask_12 = (AtomType[index_edge[0, :]] == atomtype_1) * (AtomType[index_edge[1, :]] == atomtype_2)
                            edge_12 = EdgeNumber[mask_12]
                            sub_cell_shift_index = cell_shift_index[edge_12]
                            offset0 = offset[index_edge[0, edge_12]]
                            offset1 = offset[index_edge[1, edge_12]]
                            dim0, dim1 = mr.shape[-2:]

                            for i in range(dim0):
                                for j in range(dim1):
                                    Mr[sub_cell_shift_index, offset0 + i, offset1 + j] = mr[:, i, j]

                    return Mr, dim_matrix

                prefix = "" if len(self.trainloader) == 1 else f"{batch_idx}_"

                if "h_ref" in self.para.write:
                    h_ref = rotate2abacus(data.HR, AtomType)
                    h_ref, dim_matrix = block_r(h_ref, data)
                    with open(f"{prefix}h_ref.csr", "w") as f:
                        f.write("STEP: 0\n")
                        f.write(f"Matrix Dimension of H(R): {dim_matrix}\n")
                        f.write(f"Matrix number of H(R): {n_cell}\n")

                        for i_cell in range(n_cell):
                            cell_shift = unique_cell_shift[i_cell].tolist()
                            mr = (h_ref[i_cell] / Rydberg).to_sparse_csr()
                            row_ptr = mr.crow_indices()
                            col_ind = mr.col_indices()
                            values = mr.values()
                            nnz = len(values)

                            f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]} {nnz}\n")
                            if nnz != 0:
                                f.write(" ".join(f"{x.item():.8e}" for x in values) + "\n")
                                f.write(" ".join(str(x.item()) for x in col_ind) + "\n")
                                f.write(" ".join(str(x.item()) for x in row_ptr) + "\n")

                if "olp" in self.para.write:
                    olp = rotate2abacus(data.SR, AtomType)
                    olp, dim_matrix = block_r(olp, data)
                    with open(f"{prefix}olp.csr", "w") as f:
                        f.write("STEP: 0\n")
                        f.write(f"Matrix Dimension of S(R): {dim_matrix}\n")
                        f.write(f"Matrix number of S(R): {n_cell}\n")

                        for i_cell in range(n_cell):
                            cell_shift = unique_cell_shift[i_cell].tolist()
                            mr = olp[i_cell].to_sparse_csr()
                            row_ptr = mr.crow_indices()
                            col_ind = mr.col_indices()
                            values = mr.values()
                            nnz = len(values)

                            f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]} {nnz}\n")
                            if nnz != 0:
                                f.write(" ".join(f"{x.item():.8e}" for x in values) + "\n")
                                f.write(" ".join(str(x.item()) for x in col_ind) + "\n")
                                f.write(" ".join(str(x.item()) for x in row_ptr) + "\n")

                if "rR" in self.para.write:
                    rR_x = rotate2abacus(data.rR["x"], AtomType)
                    rR_y = rotate2abacus(data.rR["y"], AtomType)
                    rR_z = rotate2abacus(data.rR["z"], AtomType)
                    rR_x, dim_matrix = block_r(rR_x, data)
                    rR_y, dim_matrix = block_r(rR_y, data)
                    rR_z, dim_matrix = block_r(rR_z, data)

                    with open(f"{prefix}rR.csr", "w") as f:
                        f.write("STEP: 0\n")
                        f.write(f"Matrix Dimension of r(R): {dim_matrix}\n")
                        f.write(f"Matrix number of r(R): {n_cell}\n")

                        for i_cell in range(n_cell):
                            cell_shift = unique_cell_shift[i_cell].tolist()
                            mr_x = (rR_x[i_cell] / Bohr).to_sparse_csr()
                            mr_y = (rR_y[i_cell] / Bohr).to_sparse_csr()
                            mr_z = (rR_z[i_cell] / Bohr).to_sparse_csr()

                            row_ptr_x = mr_x.crow_indices()
                            col_ind_x = mr_x.col_indices()
                            values_x = mr_x.values()
                            nnz_x = len(values_x)

                            row_ptr_y = mr_y.crow_indices()
                            col_ind_y = mr_y.col_indices()
                            values_y = mr_y.values()
                            nnz_y = len(values_y)

                            row_ptr_z = mr_z.crow_indices()
                            col_ind_z = mr_z.col_indices()
                            values_z = mr_z.values()
                            nnz_z = len(values_z)

                            f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]}\n")

                            f.write(f"{nnz_x}\n")
                            if nnz_x != 0:
                                f.write(" ".join(f"{x.item():.8e}" for x in values_x) + "\n")
                                f.write(" ".join(str(x.item()) for x in col_ind_x) + "\n")
                                f.write(" ".join(str(x.item()) for x in row_ptr_x) + "\n")

                            f.write(f"{nnz_y}\n")
                            if nnz_y != 0:
                                f.write(" ".join(f"{x.item():.8e}" for x in values_y) + "\n")
                                f.write(" ".join(str(x.item()) for x in col_ind_y) + "\n")
                                f.write(" ".join(str(x.item()) for x in row_ptr_y) + "\n")

                            f.write(f"{nnz_z}\n")
                            if nnz_z != 0:
                                f.write(" ".join(f"{x.item():.8e}" for x in values_z) + "\n")
                                f.write(" ".join(str(x.item()) for x in col_ind_z) + "\n")
                                f.write(" ".join(str(x.item()) for x in row_ptr_z) + "\n")

                if "h_pred" in self.para.write:
                    data.to(self.device)
                    h_pred, _ = self.model(data)
                    h_pred = tensor2tensor(h_pred, "cpu")
                    data.to("cpu")
                    h_pred = rotate2abacus(h_pred, AtomType)
                    h_pred, dim_matrix = block_r(h_pred, data)

                    with open(f"{prefix}h_pred.csr", "w") as f:
                        f.write("STEP: 0\n")
                        f.write(f"Matrix Dimension of H(R): {dim_matrix}\n")
                        f.write(f"Matrix number of H(R): {n_cell}\n")

                        for i_cell in range(n_cell):
                            cell_shift = unique_cell_shift[i_cell].tolist()
                            mr = (h_pred[i_cell] / Rydberg).to_sparse_csr()
                            row_ptr = mr.crow_indices()
                            col_ind = mr.col_indices()
                            values = mr.values()
                            nnz = len(values)

                            f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]} {nnz}\n")
                            if nnz != 0:
                                f.write(" ".join(f"{x.item():.8e}" for x in values) + "\n")
                                f.write(" ".join(str(x.item()) for x in col_ind) + "\n")
                                f.write(" ".join(str(x.item()) for x in row_ptr) + "\n")

    def save_checkpoint(self, epoch: int):
        checkpoint_dir = os.path.join(self.para.model_save_path, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"cp{epoch+1}.pth")
        checkpoint = {"epoch": epoch+1,
                      'model_state_dict': self.model.state_dict(),
                      'optimizer_state_dict': self.optimizer.state_dict(),
                      'scheduler_state_dict': self.lr_scheduler.state_dict(),
                      "Train_MSE": self.train_lossrecord.mse_ave,
                      "Train_MAE": self.train_lossrecord.mae_ave,
                      "Val_MSE": self.val_lossrecord.mse_ave,
                      "Val_MAE": self.val_lossrecord.mae_ave,
                      "Test_MSE": self.test_lossrecord.mse_ave,
                      "Test_MAE": self.test_lossrecord.mae_ave}
        torch.save(checkpoint, checkpoint_file)

    def Version_Convertion(self, old_version: dict, current_version: dict):
        Name_Convertion = {"FourierConv_layer": "GauntConv_layer",
                           "ftp": "gtp",
                           "weight_ftp": "weight_gtp"}
        old_keys = old_version.keys()
        current_keys = current_version.keys()
        for key in old_keys:
            if key not in current_keys:
                if "N_average" in key:
                    continue
                words = key.split(".")
                # when using DDP, module is wraped within ddp, attributions should be accessed through ddp.module.atr
                new_words = ["module"]
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
                try:
                    current_version[key] = old_version["N_average"]
                except:
                    continue

        return current_version

    def init_model(self):
        self.start_epoch = 0
        self.checkpoint_info = None

        if self.para.prediction == 2:
            from ..entrypoints.model_profile import Model
        else:
            from ..entrypoints.model import Model

        self.model = Model(para=self.para)
        self.model.to(device=self.device)
        if self.device.type == "cuda":
            self.model = DDP(self.model,
                             device_ids=[self.para.local_rank],
                             output_device=self.para.local_rank,
                             broadcast_buffers=False,
                             gradient_as_bucket_view=True)
        else:
            self.model = DDP(self.model,
                             #  output_device=self.para.local_rank,
                             broadcast_buffers=False,
                             gradient_as_bucket_view=True)
        self.optimizer = getattr(torch.optim, self.para.optimizer)(self.model.parameters(),
                                                                   lr=self.para.lr,
                                                                   weight_decay=self.para.lambda_2)
        if self.para.lr_scheduler == "ExponentialLR":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.para.gamma)
        elif self.para.lr_scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.para.factor, patience=self.para.patience, threshold=self.para.threshold)

        if self.para.init_from_checkpoint is not None:
            checkpoint = torch.load(self.para.init_from_checkpoint, map_location="cpu")
            self.start_epoch = checkpoint['epoch']
            checkpoint_weights = self.Version_Convertion(checkpoint['model_state_dict'], self.model.state_dict())
            init_state = {k: v for k, v in checkpoint_weights.items() if k in self.model.state_dict()}
            current_state = self.model.state_dict()
            current_state.update(init_state)
            self.model.load_state_dict(current_state)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

            if self.para.new_lr is not None:
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.para.new_lr

            self.checkpoint_info = f"Epoch:{checkpoint['epoch']:>5}   lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.6f}   " +\
                f"Train_MSE: {checkpoint['Train_MSE']:.7f}   Train_MAE: {checkpoint['Train_MAE']:.7f}   " +\
                f"Val_MSE: {checkpoint['Val_MSE']:.7f}   Val_MAE: {checkpoint['Val_MAE']:.7f}   " +\
                f"Test_MSE: {checkpoint['Test_MSE']:.7f}   Test_MAE: {checkpoint['Test_MAE']:.7f}"
            if self.para.lr_scheduler == "ExponentialLR":
                self.lr_scheduler.step()
            elif self.para.lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler.step(checkpoint['Train_MAE'])
