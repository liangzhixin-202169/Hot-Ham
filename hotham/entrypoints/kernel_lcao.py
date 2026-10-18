import torch
from e3nn.o3 import Linear as O3_Linear
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import numpy as np
from ase.units import Rydberg, Bohr
import sys
import time
from ..entrypoints.parameters import Parameters
from ..entrypoints.lossfun import LossRecord, Lossfunction
from ..data.dataset_preprocess import DatasetPrepocess
from ..modules.common import BasicInfo
from ..modules.te3nn.to3 import TO3_Linear


class Kernel(torch.nn.Module):
    def __init__(self, para: Parameters):
        super().__init__()
        self.para = para
        self.device = self.para.device

        # Basic atom type and orbit information
        self.basicinfo = BasicInfo(para.orbit, device=self.device, intdtype=self.para.torch_int_dtype)

        # Read dataset
        self.datapreprocess = DatasetPrepocess(para)
        self.trainloader = self.datapreprocess.trainset_loader
        self.testsetloader = self.datapreprocess.testset_loader

        # Initilize model, optimizer and lr_scheduler
        self.init_model()

        # Define loss function
        self.lossfunction = Lossfunction(para, self.basicinfo)
        self.train_lossrecord = LossRecord()
        self.test_lossrecord = LossRecord()

    def run(self):
        if self.para.prediction == 0:
            self.train()
        elif self.para.prediction == 1:
            self.eval()
        elif self.para.prediction == 2:
            self.topyatb()

    def _rebuild_o3_linear(self, module: TO3_Linear):
        old = module.linear
        new = O3_Linear(
            irreps_in=old.irreps_in,
            irreps_out=old.irreps_out,
            instructions=[(ins.i_in, ins.i_out) for ins in old.instructions],
            biases=(old.bias_numel > 0),
            internal_weights=old.internal_weights,
            shared_weights=old.shared_weights,
        )
        new.to(device=old.weight.device, dtype=old.weight.dtype)
        new.load_state_dict(old.state_dict())
        module.linear = new

    def _refresh_o3_linears_for_eval(self):
        if not self.para.refresh_o3_linear_in_eval:
            return
        model = self.model.module if hasattr(self.model, "module") else self.model
        for module in model.modules():
            if isinstance(module, TO3_Linear):
                self._rebuild_o3_linear(module)

    def train(self):
        if self.para.rank == 0:
            if self.checkpoint_info is not None:
                print(self.checkpoint_info)

        for epoch in range(self.start_epoch, self.start_epoch + self.para.epoch):
            epoch_start = time.perf_counter()
            self.model.train()
            self.trainloader.sampler.set_epoch(epoch)
            self.train_lossrecord.reset()
            self.test_lossrecord.reset()

            for batch_index, data in enumerate(self.trainloader):
                data.to(self.device)
                self.optimizer.zero_grad()
                H_block = self.model(data)
                mse, mae, num_ele = self.lossfunction.ham_loss(H_block, self.model.module.AtomType_OrbitalSum, data)
                loss = mse + mae**2
                self._abort_if_train_loss_exploded(epoch, batch_index, mse, mae, loss)
                # (torch.sqrt(mse)+mae).backward()
                loss.backward()
                self.optimizer.step()
                self.train_lossrecord.update(mse.item(), mae.item(), num_ele)

            current_lr = self.optimizer.param_groups[0]["lr"]

            mse_global, mae_global = self.train_lossrecord.global_loss(self.device, self.para.torch_float_dtype)
            if self.para.rank == 0:
                info_global = f"Epoch:{epoch+1:>5}   lr: {current_lr:.3e}" + f"   Train_MSE: {mse_global:.7f}   Train_MAE: {mae_global:.7f}"

            if self.para.lr_scheduler == "ExponentialLR":
                self.lr_scheduler.step()
            elif self.para.lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler.step(mae_global.item())

            is_last_epoch = epoch + 1 == self.start_epoch + self.para.epoch
            if (epoch + 1) % self.para.checkpoint_interval == 0 or is_last_epoch:
                self.model.eval()
                with torch.no_grad():
                    for data in self.testsetloader:
                        data.to(self.device)
                        H_block = self.model(data)
                        test_loss_MSE, test_loss_MAE, num_ele = self.lossfunction.ham_loss(H_block, self.model.module.AtomType_OrbitalSum, data)
                        self.test_lossrecord.update(test_loss_MSE.item(), test_loss_MAE.item(), num_ele)

                    mse_global, mae_global = self.test_lossrecord.global_loss(self.device, self.para.torch_float_dtype)
                    if self.para.rank == 0:
                        info_global += f"   Test_MSE: {mse_global:.7f}   Test_MAE: {mae_global:.7f}"

                if self.para.rank == 0:
                    self.save_checkpoint(epoch)

            if self.para.rank == 0:
                info_global += f"   Time: {time.perf_counter() - epoch_start:.3f} s"
                print(info_global)
                sys.stdout.flush()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            for data in self.trainloader:
                self._refresh_o3_linears_for_eval()
                data.to(self.device)
                H_block = self.model(data)
                train_loss_MSE, train_loss_MAE, num_ele = self.lossfunction.ham_loss(H_block, self.model.module.AtomType_OrbitalSum, data)
                self.train_lossrecord.update(train_loss_MSE.item(), train_loss_MAE.item(), num_ele)

            mse_global, mae_global = self.train_lossrecord.global_loss(self.device, self.para.torch_float_dtype)
            mse_global_max, mae_global_max = self.train_lossrecord.global_max(self.device, self.para.torch_float_dtype)
            mse_global_min, mae_global_min = self.train_lossrecord.global_min(self.device, self.para.torch_float_dtype)
            if self.para.rank == 0:
                info_global = (
                    f"Train:\n"
                    + f"    MSE(eV^2): {mse_global:.7f}    MAX: {mse_global_max:.7f}    MIN: {mse_global_min:.7f}\n"
                    + f"    MAE(eV):   {mae_global:.7f}    MAX: {mae_global_max:.7f}    MIN: {mae_global_min:.7f}\n"
                )

            for data in self.testsetloader:
                self._refresh_o3_linears_for_eval()
                data.to(self.device)
                H_block = self.model(data)
                test_loss_MSE, test_loss_MAE, num_ele = self.lossfunction.ham_loss(H_block, self.model.module.AtomType_OrbitalSum, data)
                self.test_lossrecord.update(test_loss_MSE.item(), test_loss_MAE.item(), num_ele)

            mse_global, mae_global = self.test_lossrecord.global_loss(self.device, self.para.torch_float_dtype)
            mse_global_max, mae_global_max = self.test_lossrecord.global_max(self.device, self.para.torch_float_dtype)
            mse_global_min, mae_global_min = self.test_lossrecord.global_min(self.device, self.para.torch_float_dtype)
            if self.para.rank == 0:
                info_global += (
                    f"Test:\n"
                    + f"    MSE(eV^2): {mse_global:.7f}    MAX: {mse_global_max:.7f}    MIN: {mse_global_min:.7f}\n"
                    + f"    MAE(eV):   {mae_global:.7f}    MAX: {mae_global_max:.7f}    MIN: {mae_global_min:.7f}\n"
                )

            if self.para.rank == 0:
                print(info_global)

    def topyatb(self):
        if self.para.rank != 0:
            return
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.eval()
        if not hasattr(self.para, "write"):
            raise AttributeError("para.write is required for prediction == 2")
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
                        torch.tensor([[1.0]], dtype=self.para.torch_float_dtype),
                        torch.tensor([[0, 1, 0],
                                      [0, 0, -1],
                                      [-1, 0, 0]], dtype=self.para.torch_float_dtype),
                        torch.tensor([[0, 0, 1, 0, 0],
                                      [0, 0, 0, -1, 0],
                                      [0, -1, 0, 0, 0],
                                      [0, 0, 0, 0, 1],
                                      [1, 0, 0, 0, 0]], dtype=self.para.torch_float_dtype),
                        torch.tensor([[0, 0, 0, 1, 0, 0, 0],
                                      [0, 0, 0, 0, -1, 0, 0],
                                      [0, 0, -1, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 1, 0],
                                      [0, 1, 0, 0, 0, 0, 0],
                                      [0, 0, 0, 0, 0, 0, -1],
                                      [-1, 0, 0, 0, 0, 0, 0]], dtype=self.para.torch_float_dtype)
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

                            winger_D_1 = winger_D_1.to(block[atomsymbol_1][atomsymbol_2].dtype)
                            winger_D_2 = winger_D_2.to(block[atomsymbol_1][atomsymbol_2].dtype)
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
                    unique_cell_shift_inner = data.unique_cell_shift
                    n_cell_inner = len(unique_cell_shift_inner)
                    cell_shift_index = data.cell_shift_index.cpu().numpy()
                    offset = data.offset.cpu().numpy()
                    edge_src = index_edge[0].cpu().numpy()
                    edge_dst = index_edge[1].cpu().numpy()
                    atom_type_np = AtomType.cpu().numpy()
                    AtomType_OrbitalSum = model.AtomType_OrbitalSum
                    dim_matrix = int(sum(AtomType_OrbitalSum[atomtype].item() for atomtype in AtomType))

                    mr_index = []
                    mr_value = []

                    for atomtype_1 in range(num_atomtype):
                        if atomtype_1 not in unique_atomtypes:
                            continue
                        atomsymbol_1 = AtomType_to_AtomSymbol[atomtype_1]

                        for atomtype_2 in range(num_atomtype):
                            if atomtype_2 not in unique_atomtypes:
                                continue
                            atomsymbol_2 = AtomType_to_AtomSymbol[atomtype_2]

                            mr = block[atomsymbol_1][atomsymbol_2]
                            mr = mr.squeeze(1).cpu().numpy()
                            mask_12 = (atom_type_np[edge_src] == atomtype_1) & (atom_type_np[edge_dst] == atomtype_2)
                            edge_12 = np.nonzero(mask_12)[0]
                            if len(edge_12) == 0:
                                continue

                            sub_cell_shift_index = cell_shift_index[edge_12]
                            offset0 = offset[edge_src[edge_12]]
                            offset1 = offset[edge_dst[edge_12]]
                            dim0, dim1 = mr.shape[-2:]

                            local_row = np.arange(dim0, dtype=self.para.numpy_int_dtype)
                            local_col = np.arange(dim1, dtype=self.para.numpy_int_dtype)
                            row_grid, col_grid = np.meshgrid(local_row, local_col, indexing="ij")
                            row_grid = row_grid.reshape(1, -1)
                            col_grid = col_grid.reshape(1, -1)

                            cell_idx = np.repeat(sub_cell_shift_index[:, None], dim0 * dim1, axis=1)
                            row_idx = offset0[:, None] + row_grid
                            col_idx = offset1[:, None] + col_grid
                            values = mr.reshape(mr.shape[0], -1)

                            nonzero_mask = values != 0
                            if not np.any(nonzero_mask):
                                continue

                            mr_index.append(np.stack([
                                cell_idx[nonzero_mask],
                                row_idx[nonzero_mask],
                                col_idx[nonzero_mask]
                            ], axis=1))
                            mr_value.append(values[nonzero_mask])

                    if len(mr_index) == 0:
                        return {
                            "index": np.zeros((0, 3), dtype=self.para.numpy_int_dtype),
                            "value": np.zeros(
                                (0,),
                                dtype=self.para.numpy_complex_dtype if self.para.spin else self.para.numpy_float_dtype,
                            )
                        }, dim_matrix

                    return {
                        "index": np.concatenate(mr_index, axis=0),
                        "value": np.concatenate(mr_value, axis=0)
                    }, dim_matrix

                csr_digits = 8 if self.para.numpy_float_dtype == np.float32 else 16
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
                            mask = h_ref["index"][:, 0] == i_cell
                            cell_index = h_ref["index"][mask][:, 1:]
                            values = h_ref["value"][mask] / Rydberg
                            nnz = len(values)

                            f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]} {nnz}\n")
                            if nnz != 0:
                                row_order = np.lexsort((cell_index[:, 1], cell_index[:, 0]))
                                cell_index = cell_index[row_order]
                                values = values[row_order]
                                rows = cell_index[:, 0].astype(self.para.numpy_int_dtype, copy=False)
                                cols = cell_index[:, 1].astype(self.para.numpy_int_dtype, copy=False)
                                counts = np.bincount(rows, minlength=dim_matrix).astype(self.para.numpy_int_dtype, copy=False)
                                row_ptr = np.concatenate((np.zeros(1, dtype=self.para.numpy_int_dtype), np.cumsum(counts, dtype=self.para.numpy_int_dtype)))
                                f.write(" ".join(f"({x.real:.{csr_digits}e},{x.imag:.{csr_digits}e})" if np.iscomplexobj(x) else f"{x:.{csr_digits}e}" for x in values) + "\n")
                                f.write(" ".join(str(x) for x in cols) + "\n")
                                f.write(" ".join(str(x) for x in row_ptr) + "\n")

                if "olp" in self.para.write:
                    olp = rotate2abacus(data.SR, AtomType)
                    olp, dim_matrix = block_r(olp, data)
                    with open(f"{prefix}olp.csr", "w") as f:
                        f.write("STEP: 0\n")
                        f.write(f"Matrix Dimension of S(R): {dim_matrix}\n")
                        f.write(f"Matrix number of S(R): {n_cell}\n")

                        for i_cell in range(n_cell):
                            cell_shift = unique_cell_shift[i_cell].tolist()
                            mask = olp["index"][:, 0] == i_cell
                            cell_index = olp["index"][mask][:, 1:]
                            values = olp["value"][mask]
                            nnz = len(values)

                            f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]} {nnz}\n")
                            if nnz != 0:
                                row_order = np.lexsort((cell_index[:, 1], cell_index[:, 0]))
                                cell_index = cell_index[row_order]
                                values = values[row_order]
                                rows = cell_index[:, 0].astype(self.para.numpy_int_dtype, copy=False)
                                cols = cell_index[:, 1].astype(self.para.numpy_int_dtype, copy=False)
                                counts = np.bincount(rows, minlength=dim_matrix).astype(self.para.numpy_int_dtype, copy=False)
                                row_ptr = np.concatenate((np.zeros(1, dtype=self.para.numpy_int_dtype), np.cumsum(counts, dtype=self.para.numpy_int_dtype)))
                                f.write(" ".join(f"({x.real:.{csr_digits}e},{x.imag:.{csr_digits}e})" if np.iscomplexobj(x) else f"{x:.{csr_digits}e}" for x in values) + "\n")
                                f.write(" ".join(str(x) for x in cols) + "\n")
                                f.write(" ".join(str(x) for x in row_ptr) + "\n")

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
                            mask_x = rR_x["index"][:, 0] == i_cell
                            mask_y = rR_y["index"][:, 0] == i_cell
                            mask_z = rR_z["index"][:, 0] == i_cell

                            cell_index_x = rR_x["index"][mask_x][:, 1:]
                            values_x = rR_x["value"][mask_x] / Bohr
                            nnz_x = len(values_x)

                            cell_index_y = rR_y["index"][mask_y][:, 1:]
                            values_y = rR_y["value"][mask_y] / Bohr
                            nnz_y = len(values_y)

                            cell_index_z = rR_z["index"][mask_z][:, 1:]
                            values_z = rR_z["value"][mask_z] / Bohr
                            nnz_z = len(values_z)

                            f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]}\n")

                            f.write(f"{nnz_x}\n")
                            if nnz_x != 0:
                                row_order_x = np.lexsort((cell_index_x[:, 1], cell_index_x[:, 0]))
                                cell_index_x = cell_index_x[row_order_x]
                                values_x = values_x[row_order_x]
                                rows_x = cell_index_x[:, 0].astype(self.para.numpy_int_dtype, copy=False)
                                cols_x = cell_index_x[:, 1].astype(self.para.numpy_int_dtype, copy=False)
                                counts_x = np.bincount(rows_x, minlength=dim_matrix).astype(self.para.numpy_int_dtype, copy=False)
                                row_ptr_x = np.concatenate((np.zeros(1, dtype=self.para.numpy_int_dtype), np.cumsum(counts_x, dtype=self.para.numpy_int_dtype)))
                                f.write(" ".join(f"({x.real:.{csr_digits}e},{x.imag:.{csr_digits}e})" if np.iscomplexobj(x) else f"{x:.{csr_digits}e}" for x in values_x) + "\n")
                                f.write(" ".join(str(x) for x in cols_x) + "\n")
                                f.write(" ".join(str(x) for x in row_ptr_x) + "\n")

                            f.write(f"{nnz_y}\n")
                            if nnz_y != 0:
                                row_order_y = np.lexsort((cell_index_y[:, 1], cell_index_y[:, 0]))
                                cell_index_y = cell_index_y[row_order_y]
                                values_y = values_y[row_order_y]
                                rows_y = cell_index_y[:, 0].astype(self.para.numpy_int_dtype, copy=False)
                                cols_y = cell_index_y[:, 1].astype(self.para.numpy_int_dtype, copy=False)
                                counts_y = np.bincount(rows_y, minlength=dim_matrix).astype(self.para.numpy_int_dtype, copy=False)
                                row_ptr_y = np.concatenate((np.zeros(1, dtype=self.para.numpy_int_dtype), np.cumsum(counts_y, dtype=self.para.numpy_int_dtype)))
                                f.write(" ".join(f"({x.real:.{csr_digits}e},{x.imag:.{csr_digits}e})" if np.iscomplexobj(x) else f"{x:.{csr_digits}e}" for x in values_y) + "\n")
                                f.write(" ".join(str(x) for x in cols_y) + "\n")
                                f.write(" ".join(str(x) for x in row_ptr_y) + "\n")

                            f.write(f"{nnz_z}\n")
                            if nnz_z != 0:
                                row_order_z = np.lexsort((cell_index_z[:, 1], cell_index_z[:, 0]))
                                cell_index_z = cell_index_z[row_order_z]
                                values_z = values_z[row_order_z]
                                rows_z = cell_index_z[:, 0].astype(self.para.numpy_int_dtype, copy=False)
                                cols_z = cell_index_z[:, 1].astype(self.para.numpy_int_dtype, copy=False)
                                counts_z = np.bincount(rows_z, minlength=dim_matrix).astype(self.para.numpy_int_dtype, copy=False)
                                row_ptr_z = np.concatenate((np.zeros(1, dtype=self.para.numpy_int_dtype), np.cumsum(counts_z, dtype=self.para.numpy_int_dtype)))
                                f.write(" ".join(f"({x.real:.{csr_digits}e},{x.imag:.{csr_digits}e})" if np.iscomplexobj(x) else f"{x:.{csr_digits}e}" for x in values_z) + "\n")
                                f.write(" ".join(str(x) for x in cols_z) + "\n")
                                f.write(" ".join(str(x) for x in row_ptr_z) + "\n")

                if "h_pred" in self.para.write:
                    data.to(self.device)
                    h_pred = self.model(data)
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
                            mask = h_pred["index"][:, 0] == i_cell
                            cell_index = h_pred["index"][mask][:, 1:]
                            values = h_pred["value"][mask] / Rydberg
                            nnz = len(values)

                            f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]} {nnz}\n")
                            if nnz != 0:
                                row_order = np.lexsort((cell_index[:, 1], cell_index[:, 0]))
                                cell_index = cell_index[row_order]
                                values = values[row_order]
                                rows = cell_index[:, 0].astype(self.para.numpy_int_dtype, copy=False)
                                cols = cell_index[:, 1].astype(self.para.numpy_int_dtype, copy=False)
                                counts = np.bincount(rows, minlength=dim_matrix).astype(self.para.numpy_int_dtype, copy=False)
                                row_ptr = np.concatenate((np.zeros(1, dtype=self.para.numpy_int_dtype), np.cumsum(counts, dtype=self.para.numpy_int_dtype)))
                                f.write(" ".join(f"({x.real:.{csr_digits}e},{x.imag:.{csr_digits}e})" if np.iscomplexobj(x) else f"{x:.{csr_digits}e}" for x in values) + "\n")
                                f.write(" ".join(str(x) for x in cols) + "\n")
                                f.write(" ".join(str(x) for x in row_ptr) + "\n")

    def save_checkpoint(self, epoch: int):
        checkpoint_dir = os.path.join(self.para.model_save_path, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"cp{epoch+1}.pth")
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "scheduler_step_completed": True,
            "Train_MSE": self.train_lossrecord.mse_ave,
            "Train_MAE": self.train_lossrecord.mae_ave,
            "Test_MSE": self.test_lossrecord.mse_ave,
            "Test_MAE": self.test_lossrecord.mae_ave,
        }
        torch.save(checkpoint, checkpoint_file)

    def _abort_if_train_loss_exploded(self, epoch: int, batch_index: int, mse: torch.Tensor, mae: torch.Tensor, loss: torch.Tensor):
        threshold = self.para.get("loss_abort_threshold")
        local_bad_loss = not torch.isfinite(loss).item()
        if threshold is not None:
            local_bad_loss = local_bad_loss or loss.detach().item() > threshold

        bad_loss = torch.tensor([local_bad_loss], dtype=self.para.torch_int_dtype, device=loss.device)
        if dist.is_initialized():
            dist.all_reduce(bad_loss, op=dist.ReduceOp.MAX)
        bad_loss = bool(bad_loss.item())
        if not bad_loss:
            return

        if local_bad_loss:
            reason = "nonfinite" if not torch.isfinite(loss).item() else f"above_threshold_{threshold}"
        else:
            reason = "other_rank"
        if self.para.rank == 0:
            self.save_loss_abort_checkpoint(epoch, batch_index, mse, mae, loss, reason)
            print(
                f"Abort training: loss {loss.detach().item()} at epoch {epoch + 1}, " f"batch {batch_index}, reason={reason}",
                flush=True,
            )
        raise RuntimeError(f"Training aborted because loss is {reason}")

    def save_loss_abort_checkpoint(self, epoch: int, batch_index: int, mse: torch.Tensor, mae: torch.Tensor, loss: torch.Tensor, reason: str):
        checkpoint_dir = os.path.join(self.para.model_save_path, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"loss_abort_epoch{epoch+1}_batch{batch_index}.pth")
        checkpoint = {
            "epoch": epoch + 1,
            "batch_index": batch_index,
            "reason": reason,
            "loss": loss.detach().item(),
            "mse": mse.detach().item(),
            "mae": mae.detach().item(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "Train_MSE": self.train_lossrecord.mse_ave,
            "Train_MAE": self.train_lossrecord.mae_ave,
            "Test_MSE": self.test_lossrecord.mse_ave,
            "Test_MAE": self.test_lossrecord.mae_ave,
        }
        torch.save(checkpoint, checkpoint_file)

    def Version_Convertion(self, old_version: dict, current_version: dict):
        Name_Convertion = {"FourierConv_layer": "GauntConv_layer", "ftp": "gtp", "weight_ftp": "weight_gtp"}
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
        checkpoint_path = self.para.get("init_from_checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path else None
        if checkpoint is None:
            initial_lr = self.para.lr
        else:
            initial_lr = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]

        if self.para.version == 2:
            from ..entrypoints.model_lcao_mag import Model
        else:
            from ..entrypoints.model_lcao import Model

        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(self.para.torch_float_dtype)
            self.model = Model(para=self.para)
        finally:
            torch.set_default_dtype(default_dtype)
        self.model._apply(lambda tensor: tensor.to(
            device=self.device,
            dtype=self.para.torch_complex_dtype if tensor.is_complex() else
                  self.para.torch_float_dtype if tensor.is_floating_point() else
                  tensor.dtype if tensor.dtype == torch.bool else self.para.torch_int_dtype
        ))
        if self.device.type == "cuda":
            self.model = DDP(
                self.model,
                device_ids=[self.para.local_rank],
                output_device=self.para.local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        else:
            self.model = DDP(
                self.model,
                #  output_device=self.para.local_rank,
                broadcast_buffers=False,
                gradient_as_bucket_view=True,
            )
        self.optimizer = getattr(torch.optim, self.para.optimizer)(params=self.model.parameters(), lr=initial_lr, weight_decay=self.para.lambda_2)
        if self.para.lr_scheduler == "ExponentialLR":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.para.gamma)
        elif self.para.lr_scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="min", factor=self.para.factor, patience=self.para.patience, threshold=self.para.threshold
            )

        if checkpoint is not None:
            self.start_epoch = checkpoint["epoch"]
            checkpoint_weights = self.Version_Convertion(checkpoint["model_state_dict"], self.model.state_dict())
            init_state = {k: v for k, v in checkpoint_weights.items() if k in self.model.state_dict()}
            current_state = self.model.state_dict()
            current_state.update(init_state)
            self.model.load_state_dict(current_state)
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if not checkpoint.get("scheduler_step_completed", False):
                if self.para.lr_scheduler == "ExponentialLR":
                    self.lr_scheduler.step()
                elif self.para.lr_scheduler == "ReduceLROnPlateau":
                    self.lr_scheduler.step(checkpoint["Train_MAE"])

            if self.para.get("new_lr") is not None:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.para.new_lr

            self.checkpoint_info = (
                f"Epoch:{checkpoint['epoch']:>5}   lr: {self.optimizer.state_dict()['param_groups'][0]['lr']:.3e}   "
                + f"Train_MSE: {checkpoint['Train_MSE']:.7f}   Train_MAE: {checkpoint['Train_MAE']:.7f}   "
                + f"Test_MSE: {checkpoint['Test_MSE']:.7f}   Test_MAE: {checkpoint['Test_MAE']:.7f}"
            )
