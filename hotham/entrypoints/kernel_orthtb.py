import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch_geometric.loader import DataLoader
import os
import sys
from .parameters import Parameters
from .lossfun import Lossfunction
from ..data.dataset_preprocess import DatasetPrepocess
from .model_orthtb import Model
import time
import matplotlib.pyplot as plt
#import matplotlib
#matplotlib.use('Agg')  # 禁用 GUI 渲染
import numpy as np
from ..utilities.soc import compute_soc_matrix_in_cubic_basis
from ase.units import Rydberg,Bohr

class Kernel(torch.nn.Module):
    def __init__(self, para: Parameters):
        super().__init__()
        self.para = para
        self.device = self.para.device

        # Read dataset
        self.datapreprocess = DatasetPrepocess(para)
        self.trainsetloader = self.datapreprocess.trainset_loader
        self.testsetloader = self.datapreprocess.testset_loader

        # Initilize model, optimizer and lr_scheduler
        self.start_epoch = 0
        self.checkpoint_info = None

        default_dtype = torch.get_default_dtype()
        try:
            torch.set_default_dtype(self.para.torch_float_dtype)
            self.model = Model(para=para, N_average=self.datapreprocess.N_average)
        finally:
            torch.set_default_dtype(default_dtype)

        self.model._apply(lambda tensor: tensor.to(
            device=self.device,
            dtype=self.para.torch_complex_dtype if tensor.is_complex() else
                  self.para.torch_float_dtype if tensor.is_floating_point() else
                  tensor.dtype if tensor.dtype == torch.bool else self.para.torch_int_dtype
        ))
        if self.para.rank == 0:
            print("Backward para num:",sum(p.numel() for p in self.model.parameters() if p.requires_grad))
        model_path = self.para.get("init_from_checkpoint")
        saved_data = torch.load(model_path, map_location="cpu", weights_only=False) if model_path else None
        optimizer_state = saved_data.get("optimizer_state") if isinstance(saved_data, dict) else None
        if model_path and self.para.prediction == 0 and (not isinstance(saved_data, dict) or "epoch" not in saved_data):
            raise ValueError("This OrthTB checkpoint has no epoch; its training epoch cannot be resumed.")
        if model_path and optimizer_state is None and self.para.prediction == 0:
            raise ValueError("This OrthTB checkpoint has no optimizer state; its learning rate cannot be resumed.")
        initial_lr = optimizer_state["param_groups"][0]["lr"] if optimizer_state is not None else self.para.lr
        self.optimizer = getattr(torch.optim, para.optimizer)(self.model.parameters(),
                                                                lr=initial_lr,
                                                                weight_decay=self.para.lambda_2)
        if self.para.lr_scheduler == "ExponentialLR":
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=para.gamma)
        elif self.para.lr_scheduler == "ReduceLROnPlateau":
            self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,factor=self.para.factor,patience=self.para.patience,threshold=self.para.threshold)


        self.lossfunction = Lossfunction(para)
        self.HuberLoss=torch.nn.HuberLoss(delta=self.para.delta,reduction='mean')
        self.initialize_model(saved_data)
        if self.para.prediction == 0:
            ddp_options = {"find_unused_parameters": False, "broadcast_buffers": False}
            if self.device.type == "cuda":
                ddp_options.update(device_ids=[self.para.local_rank], output_device=self.para.local_rank)
            self.model = DDP(self.model, **ddp_options)

    @property
    def raw_model(self):
        return self.model.module if isinstance(self.model, DDP) else self.model

    def _global_average(self, total, count):
        values = torch.stack((
            torch.as_tensor(total, device=self.device, dtype=self.para.torch_float_dtype),
            torch.as_tensor(count, device=self.device, dtype=self.para.torch_float_dtype),
        ))
        if dist.is_initialized():
            dist.all_reduce(values)
        return (values[0] / values[1]).item() if values[1].item() else float("nan")

    def _prediction_loader(self, loader):
        return DataLoader(loader.dataset, batch_size=self.para.batch_size, shuffle=False)

    def initialize_model(self, saved_data):
        if saved_data is None:
            return

        # 关键检测逻辑：判断是旧版还是新版格式
        if isinstance(saved_data, dict) and 'model_state' in saved_data:  # 新版格式
            model_state = saved_data['model_state']
            gradients = saved_data.get('gradients', {})
            optimizer_state = saved_data.get('optimizer_state', None)
        else:  # 旧版直接保存state_dict的格式
            model_state = saved_data  # 旧版数据直接作为模型参数
            gradients = {}  # 旧版无梯度
            optimizer_state = None  # 旧版无优化器状态

        # 加载模型参数
        model_state = {k.removeprefix('module.'): v for k, v in model_state.items()}
        gradients = {k.removeprefix('module.'): v for k, v in gradients.items()}
        current_model_state = self.raw_model.state_dict()
        filtered_model_state = {k: v for k, v in model_state.items() if k in current_model_state}
        current_model_state.update(filtered_model_state)
        self.raw_model.load_state_dict(current_model_state, strict=False)

        # 恢复梯度（旧版文件无梯度则跳过）
        for name, param in self.raw_model.named_parameters():
            if name in gradients:
                grad = gradients[name]
                param.grad = grad.to(device=self.para.device, dtype=param.dtype) if grad is not None else None

        # 恢复训练状态；new_lr 可覆盖 checkpoint 中的学习率。
        if optimizer_state is not None and self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer_state)
        if isinstance(saved_data, dict) and 'scheduler_state' in saved_data:
            self.lr_scheduler.load_state_dict(saved_data['scheduler_state'])
        if self.para.get('new_lr') is not None:
            for group in self.optimizer.param_groups:
                group['lr'] = self.para.new_lr
        if isinstance(saved_data, dict) and 'epoch' in saved_data:
            self.start_epoch = saved_data['epoch']


    def output_model(self, epoch):
        if self.para.rank != 0:
            return
        checkpoint_dir = os.path.join(self.para.model_save_path, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        model_path = os.path.join(checkpoint_dir, f"cp{epoch + 1}.pth")
        # 保存为新版格式（包含梯度+优化器）
        torch.save({
            'model_state': self.raw_model.state_dict(),
            'gradients': {n: p.grad.clone() if p.grad is not None else None
                        for n, p in self.raw_model.named_parameters()},
            'optimizer_state': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state': self.lr_scheduler.state_dict(),
            'epoch': epoch + 1,
        }, model_path)

    def run(self):
        if self.para.prediction == 0 :
            self.train()
        elif self.para.rank == 0:
            if self.para.prediction == 1:
                self.eval()
            elif self.para.prediction == 2:
                self.topyatb()
            
    def matrixes_k(self, matrixes: dict, k: torch.tensor,data):
        dim_matrix = data.Orbitalnum

        Mk = torch.zeros((k.shape[0], dim_matrix, dim_matrix), dtype=self.para.torch_complex_dtype, device=self.device)
        num_atomtype = self.para.num_types
        num_edge = data.num_edges
        edge_scr = data.edge_index_hop[0]
        edge_dst = data.edge_index_hop[1]

        AtomType = data.AtomType
        Node_offset = data.orbit_offset[0]
        #cell_shift = data.S_hop.to(k.dtype)
        for atomtype_1 in range(num_atomtype):
            for atomtype_2 in range(num_atomtype):
                index_HBlcok = atomtype_1*num_atomtype+atomtype_2
                Ms = matrixes[index_HBlcok]

                edge_12 = torch.arange(num_edge, dtype=self.para.torch_int_dtype, device=self.device)[(AtomType[edge_scr] == atomtype_1)*(AtomType[edge_dst] == atomtype_2)]
                cell_shift = data.S_hop[edge_12].to(dtype=k.dtype)
                exp_kr = torch.exp(2.j * torch.pi * (k @ cell_shift.T))
                Ms_k = torch.einsum("nij,kn->knij", Ms, exp_kr)
                dim0, dim1 = Ms.shape[-2:]
                Mk_flatten = Mk.reshape(Mk.shape[:-2]+(-1,))
                offset0 = Node_offset[edge_scr[edge_12]]
                offset1 = Node_offset[edge_dst[edge_12]]
                for i in range(dim0):
                    for j in range(dim1):
                        index = (offset0+i)*dim_matrix+(offset1+j)
                        Mk_flatten.index_add_(-1, index=index, source=Ms_k[..., i, j])

        return Mk#.contiguous()


    def compute_H_pre(self,data,H_block):
        kpath_list = data.kpoints
        if len(kpath_list.shape) != 2:
            kpath_list = kpath_list.reshape(-1, 3)
        Mk = self.matrixes_k(H_block, kpath_list,data)#.contiguous()
        
        eig = torch.linalg.eigvalsh(Mk)
        return eig[None,...]


    def find_soc_block(self,data,soc_val):
        model = self.raw_model

        soc_uu = torch.zeros((data.Orbitalnum, data.Orbitalnum), device=self.device, dtype=self.para.torch_complex_dtype)
        soc_ud = torch.zeros((data.Orbitalnum, data.Orbitalnum), device=self.device, dtype=self.para.torch_complex_dtype)

        soc_a_uu = getattr(self, "soc_a_uu") if hasattr(self, "soc_a_uu") else {}
        soc_a_ud =  getattr(self, "soc_a_ud") if hasattr(self, "soc_a_ud") else {}
        if not soc_a_uu or not soc_a_ud:
            for type_i in data.AtomType:
                num_orbs_i=(2*torch.as_tensor(model.AtomType_AMList.get(type_i.item()),device=self.device,dtype=self.para.torch_int_dtype)+1).sum()

                tmp_uu = torch.zeros([num_orbs_i, num_orbs_i],  dtype=self.para.torch_complex_dtype, device=self.device)
                tmp_ud = torch.zeros([num_orbs_i, num_orbs_i],  dtype=self.para.torch_complex_dtype, device=self.device)

                index_i = 0
                for or_i in model.AtomType_AMList.get(type_i.item()):
                    length = 2*or_i + 1

                    soc_orb = compute_soc_matrix_in_cubic_basis(or_i, device=self.device, dtype=self.para.torch_float_dtype)

                    tmp_uu[index_i:index_i+length, index_i:index_i+length] = soc_orb[:length,:length]
                    tmp_ud[index_i:index_i+length, index_i:index_i+length] = soc_orb[:length, length:]
                    index_i = index_i + length

                soc_a_uu.update({type_i.item():tmp_uu})
                soc_a_ud.update({type_i.item():tmp_ud})

            self.soc_a_uu = soc_a_uu
            self.soc_a_ud = soc_a_ud


        for index_i in range(data.num_nodes):

            or_start=data.orbit_offset[0][index_i]
            or_end=data.orbit_offset[0][index_i]+data.orbit_num_per_atom[0][index_i]
            type_i = data.AtomType[index_i]

            # get lambdas
            offset_i = 0
            lambdas = torch.zeros(data.orbit_num_per_atom[0][index_i].item(), device=self.device, dtype=self.para.torch_complex_dtype)

            for or_i in model.AtomType_AMList.get(type_i.item()):
                length = 2*or_i + 1
                lambdas[offset_i:offset_i+length] = soc_val[type_i][or_i]
                offset_i = offset_i + length

            soc_uu[or_start:or_end,or_start:or_end] = soc_a_uu.get(type_i.item()) @ torch.diag(lambdas)
            soc_ud[or_start:or_end, or_start:or_end] = soc_a_ud.get(type_i.item()) @ torch.diag(lambdas)

        soc_uu.contiguous()
        soc_ud.contiguous()
        self.soc_uu=soc_uu
        self.soc_ud=soc_ud

    def matrixes_k_soc(self, matrixes: dict, k: torch.tensor,data):
        dim_matrix = data.Orbitalnum

        Mk = torch.zeros((k.shape[0], dim_matrix, dim_matrix), dtype=self.para.torch_complex_dtype, device=self.device)
        num_atomtype = self.para.num_types
        num_edge = data.num_edges
        edge_scr = data.edge_index_hop[0]
        edge_dst = data.edge_index_hop[1]

        AtomType = data.AtomType
        Node_offset = data.orbit_offset[0]
        cell_shift = data.S_hop.to(k.dtype)
        for atomtype_1 in range(num_atomtype):
            for atomtype_2 in range(num_atomtype):
                index_HBlcok = atomtype_1*num_atomtype+atomtype_2
                Ms = matrixes[index_HBlcok]

                edge_12 = torch.arange(num_edge, dtype=self.para.torch_int_dtype, device=self.device)[(AtomType[edge_scr] == atomtype_1)*(AtomType[edge_dst] == atomtype_2)]
                exp_kr = torch.exp(2.j*torch.pi*(k@(cell_shift[edge_12].T)))
                Ms_k = torch.einsum("nij,kn->knij", Ms, exp_kr)
                dim0, dim1 = Ms.shape[-2:]
                Mk_flatten = Mk.reshape(Mk.shape[:-2]+(-1,))
                offset0 = Node_offset[edge_scr[edge_12]]
                offset1 = Node_offset[edge_dst[edge_12]]
                for i in range(dim0):
                    for j in range(dim1):
                        index = (offset0+i)*dim_matrix+(offset1+j)
                        Mk_flatten.index_add_(-1, index=index, source=Ms_k[..., i, j])

        MK=torch.kron(input=torch.eye(2, device=self.device, dtype=self.para.torch_complex_dtype), other=Mk)

        MK[:, :data.Orbitalnum, :data.Orbitalnum] += self.soc_uu.unsqueeze(0)
        MK[:, data.Orbitalnum:, data.Orbitalnum:] += self.soc_uu.conj().unsqueeze(0)
        MK[:, :data.Orbitalnum, data.Orbitalnum:] += self.soc_ud.unsqueeze(0)
        MK[:, data.Orbitalnum:, :data.Orbitalnum] += self.soc_ud.conj().unsqueeze(0)

        return MK#.contiguous()

    def compute_H_pre_soc(self,data,H_block,soc_val):
        self.find_soc_block(data,soc_val)
        kpath_list = data.kpoints
        if len(kpath_list.shape) != 2:
            kpath_list = kpath_list.reshape(-1, 3)
        Mk = self.matrixes_k_soc(H_block, kpath_list,data)
        
        eig = torch.linalg.eigvalsh(Mk)
        return eig[None,...]

    def train(self):
        self.model.train()
        self.global_step=0
        for epoch in range(self.start_epoch, self.start_epoch+self.para.epoch):
            epoch_start = time.perf_counter()
            self.model.train()
            self.trainsetloader.sampler.set_epoch(epoch)
            scheduler_stats = torch.zeros(2, dtype=self.para.torch_float_dtype, device=self.device)
            train_stats = torch.zeros(3, dtype=self.para.torch_float_dtype, device=self.device)
            train_count = 0

            for data in self.trainsetloader:
                data.to(self.device)
                self.optimizer.zero_grad()

                if self.para.using_soc:
                    H_block, self.GraphEdgeIndex_to_BlockEdgeIndex,soc_val = self.model(data)
                    Eig_pre=self.compute_H_pre_soc(data, H_block,soc_val)
                    Eig_ref=data.Eig_ref
                else:
                    H_block, self.GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                    Eig_pre=self.compute_H_pre(data, H_block)
                    Eig_ref=data.Eig_ref

                mse, mae = self.lossfunction.trainloss_band(Eig_pre, Eig_ref, self.para.band_range, self.para.e_range)
                #train_loss=self.lossfunction.trainloss_band_new(Eig_pre, Eig_ref,self.para.band_range)
                #train_loss=self.HuberLoss(Eig_pre[...,self.para.band_range[0]:self.para.band_range[1]],Eig_ref[...,self.para.band_range[0]:self.para.band_range[1]])
                train_loss=torch.sqrt(mse)+mae
                if self.para.using_dosloss:
                    mse_dos,mae_dos=self.lossfunction.get_dos_loss(self.para.dos_sigma,10000,Eig_pre[0],Eig_ref[0],self.para.band_range,self.device)
                    train_loss+=self.para.dos_weight*(torch.sqrt(mse_dos)+mae_dos)


                train_loss.backward()
                self.optimizer.step()
                scheduler_stats[0] += train_loss.detach() * data.num_graphs
                scheduler_stats[1] += data.num_graphs
                train_stats[0] += mse.detach() * data.num_graphs
                train_stats[1] += mae.detach() * data.num_graphs
                if self.para.using_dosloss:
                    train_stats[2] += mae_dos.detach() * data.num_graphs
                train_count += data.num_graphs

            if self.para.lr_scheduler == "ExponentialLR":
                self.lr_scheduler.step()
            elif self.para.lr_scheduler == "ReduceLROnPlateau":
                dist.all_reduce(scheduler_stats)
                self.lr_scheduler.step((scheduler_stats[0] / scheduler_stats[1]).item())

            current_lr = self.optimizer.param_groups[0]["lr"]
            train_mse = self._global_average(train_stats[0], train_count)
            train_mae = self._global_average(train_stats[1], train_count)
            if self.para.using_dosloss:
                train_dos_mae = self._global_average(train_stats[2], train_count)

            if self.para.rank == 0:
                info = (f"Epoch:{epoch+1:>5}   lr: {current_lr:.3e}"
                        f"   Train_MSE: {train_mse:.7f}   Train_MAE: {train_mae:.7f}")
                if self.para.using_dosloss:
                    info += f"   Train_MAE_DOS: {train_dos_mae:.7f}"

            is_last_epoch = epoch + 1 == self.start_epoch + self.para.epoch
            if (epoch + 1) % self.para.checkpoint_interval == 0 or is_last_epoch:
                self.model.eval()
                with torch.no_grad():
                    test_stats = torch.zeros(3, dtype=self.para.torch_float_dtype, device=self.device)
                    test_count = 0
                    for data in self.testsetloader:
                        data.to(self.device)
                        test_count += data.num_graphs

                        if self.para.using_soc:
                            H_block, self.GraphEdgeIndex_to_BlockEdgeIndex, soc_val = self.model(data)
                            Eig_pre = self.compute_H_pre_soc(data, H_block, soc_val)
                        else:
                            H_block, self.GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                            Eig_pre = self.compute_H_pre(data, H_block)

                        mse, mae = self.lossfunction.testloss_band(
                            Eig_pre, data.Eig_ref, self.para.band_range, self.para.e_range
                        )
                        test_stats[0] += mse * data.num_graphs
                        test_stats[1] += mae * data.num_graphs
                        if self.para.using_dosloss:
                            _, mae_dos = self.lossfunction.get_dos_loss(
                                self.para.dos_sigma, 10000, Eig_pre[0], data.Eig_ref[0],
                                self.para.band_range, self.device
                            )
                            test_stats[2] += mae_dos * data.num_graphs

                    mse_test = self._global_average(test_stats[0], test_count)
                    mae_test = self._global_average(test_stats[1], test_count)
                    if self.para.using_dosloss:
                        mae_dos_test = self._global_average(test_stats[2], test_count)

                self.output_model(epoch)
                if self.para.rank == 0:
                    info += f"   Test_MSE: {mse_test:.7f}   Test_MAE: {mae_test:.7f}"
                    if self.para.using_dosloss:
                        info += f"   Test_MAE_DOS: {mae_dos_test:.7f}"

            if self.para.rank == 0:
                info += f"   Time: {time.perf_counter() - epoch_start:.3f} s"
                print(info)
                sys.stdout.flush()

    def topyatb(self):
        self.model.eval()
        model = self.raw_model
        write = self.para.get("write")
        allowed = {"h_pred", "olp", "rR", "opos"}
        if not isinstance(write, (list, tuple)) or not write or any(
            not isinstance(item, str) or item not in allowed for item in write
        ):
            raise ValueError("OrthTB prediction=2 requires write to be a nonempty list of h_pred, olp, rR, opos")
        with torch.inference_mode():
            for batch_idx, data in enumerate(self._prediction_loader(self.trainsetloader)):
                data.to(self.device)
                if "h_pred" in write:
                    if self.para.using_soc:
                        H_block, _, soc_val = self.model(data)
                        self.find_soc_block(data, soc_val)
                    else:
                        H_block, _ = self.model(data)
                if data.num_graphs != 1:
                    raise ValueError("OrthTB CSR export requires batch_size=1")
                prefix = "" if len(self.trainsetloader.dataset) == 1 else f"{batch_idx}_"

                # ABACUS stores one CSR matrix per lattice shift R.  Accumulate
                # directly into those blocks so the R label and values cannot diverge.
                from scipy.sparse import coo_matrix

                i_hop, j_hop = data.edge_index_hop
                shifts = data.S_hop.cpu().numpy()
                unique_S_tuple_sort = sorted({tuple(map(int, shift)) for shift in shifts})
                unique_cell_shift = torch.tensor(unique_S_tuple_sort, device=self.device, dtype=self.para.torch_int_dtype)
                csr_dict = dict.fromkeys(unique_S_tuple_sort)
                orbital_dim = int(data.Orbitalnum.item())
                dim_matrix = 2 * orbital_dim if self.para.using_soc else orbital_dim
                atom_types = data.AtomType.cpu().numpy()
                sources = i_hop.cpu().numpy()
                targets = j_hop.cpu().numpy()
                offsets = data.orbit_offset[0].cpu().numpy()
                entries = {shift: ([], [], []) for shift in unique_S_tuple_sort}
                if (0, 0, 0) not in entries:
                    raise ValueError("OrthTB export requires the onsite R=(0,0,0) block")

                for atomtype_1 in range(model.num_atomtype) if "h_pred" in write else ():
                    for atomtype_2 in range(model.num_atomtype):
                        block_index = atomtype_1 * model.num_atomtype + atomtype_2
                        edge_ids = np.flatnonzero(
                            (atom_types[sources] == atomtype_1)
                            & (atom_types[targets] == atomtype_2)
                        )
                        blocks = H_block[block_index].detach().cpu().numpy()
                        for block, edge_id in zip(blocks, edge_ids):
                            shift = tuple(map(int, shifts[edge_id]))
                            rows, cols, values = entries[shift]
                            row0 = int(offsets[sources[edge_id]])
                            col0 = int(offsets[targets[edge_id]])
                            for local_row, local_col in np.ndindex(block.shape):
                                value = float(block[local_row, local_col])
                                if value != 0:
                                    rows.append(row0 + local_row)
                                    cols.append(col0 + local_col)
                                    values.append(value / Rydberg)
                                    if self.para.using_soc:
                                        rows.append(row0 + local_row + orbital_dim)
                                        cols.append(col0 + local_col + orbital_dim)
                                        values.append(value / Rydberg)

                if "h_pred" in write and self.para.using_soc:
                    rows, cols, values = entries[(0, 0, 0)]
                    soc_uu = self.soc_uu.cpu().numpy()
                    soc_ud = self.soc_ud.cpu().numpy()
                    for i, j in np.ndindex(orbital_dim, orbital_dim):
                        for row, col, value in (
                            (i, j, soc_uu[i, j]),
                            (i + orbital_dim, j + orbital_dim, soc_uu[i, j].conjugate()),
                            (i, j + orbital_dim, soc_ud[i, j]),
                            (i + orbital_dim, j, soc_ud[i, j].conjugate()),
                        ):
                            if value != 0:
                                rows.append(row)
                                cols.append(col)
                                values.append(value / Rydberg)

                csr_digits = 8 if self.para.numpy_float_dtype == np.float32 else 16

                def write_csr(path, label, blocks):
                    with open(path, "w") as f:
                        f.write("STEP: 0\n")
                        f.write(f"Matrix Dimension of {label}(R): {dim_matrix}\n")
                        f.write(f"Matrix number of {label}(R): {len(unique_S_tuple_sort)}\n")
                        for shift in unique_S_tuple_sort:
                            rows, cols, values = blocks[shift]
                            matrix = coo_matrix(
                                (np.asarray(values, dtype=self.para.numpy_complex_dtype if self.para.using_soc and label == "H" else self.para.numpy_float_dtype),
                                 (np.asarray(rows, dtype=self.para.numpy_int_dtype),
                                  np.asarray(cols, dtype=self.para.numpy_int_dtype))),
                                shape=(dim_matrix, dim_matrix)
                            ).tocsr()
                            matrix.eliminate_zeros()
                            f.write(f"{shift[0]} {shift[1]} {shift[2]} {matrix.nnz}\n")
                            if matrix.nnz:
                                f.write(" ".join(
                                    f"({v.real:.{csr_digits}e},{v.imag:.{csr_digits}e})" if self.para.using_soc and label == "H" else f"{v:.{csr_digits}e}"
                                    for v in matrix.data
                                ) + "\n")
                                f.write(" ".join(map(str, matrix.indices)) + "\n")
                                f.write(" ".join(map(str, matrix.indptr)) + "\n")

                if "h_pred" in write:
                    write_csr(f"{prefix}hrs1_nao.csr", "H", entries)
                overlap = {
                    shift: ([], [], []) for shift in unique_S_tuple_sort
                }
                if (0, 0, 0) not in overlap:
                    raise ValueError("OrthTB export requires the onsite R=(0,0,0) block")
                diagonal = list(range(dim_matrix))
                overlap[(0, 0, 0)] = (diagonal, diagonal, [1.0] * dim_matrix)
                if "olp" in write:
                    write_csr(f"{prefix}srs1_nao.csr", "S", overlap)

                if "opos" in write:
                    # Row i matches orbital i in the exported Hamiltonian CSR.
                    orbital_positions = np.empty((dim_matrix, 3), dtype=self.para.numpy_float_dtype)
                    positions = data.pos.cpu().numpy()
                    orbital_counts = model.AtomType_OrbitalSum
                    for atom_idx, atom_type in enumerate(atom_types):
                        start = int(offsets[atom_idx])
                        count = int(orbital_counts[int(atom_type)])
                        orbital_positions[start:start + count] = positions[atom_idx]
                    if self.para.using_soc:
                        orbital_positions[orbital_dim:] = orbital_positions[:orbital_dim]
                    np.save(f"{prefix}orbital_positions.npy", orbital_positions)

                if "rR" in write:
                    self._write_orthtb_rr(f"{prefix}rr.csr", data, model, dim_matrix,
                                          orbital_dim, unique_cell_shift, csr_dict, csr_digits)

    def _write_orthtb_rr(self, path, data, model, dim_matrix, orbital_dim,
                         unique_cell_shift, csr_dict, csr_digits):
        with open(path, "w") as f:
            f.write(f"STEP: 0\n")
            f.write(f"Matrix Dimension of r(R): {dim_matrix}\n")
            f.write(f"Matrix number of r(R): {len(unique_cell_shift)}\n")

            for key in csr_dict:
                if key ==(0,0,0):

                    Rx=torch.zeros((dim_matrix, dim_matrix), dtype=self.para.torch_float_dtype, device=self.device)
                    Ry=torch.zeros((dim_matrix, dim_matrix), dtype=self.para.torch_float_dtype, device=self.device)
                    Rz=torch.zeros((dim_matrix, dim_matrix), dtype=self.para.torch_float_dtype, device=self.device)

                    for i in range(data.num_nodes):
                        x,y,z=data.pos[i]/Bohr
                        l=model.AtomType_OrbitalSum[data.AtomType[i]]
                        x_mat=torch.eye(l,device=self.device,dtype=self.para.torch_float_dtype)*x
                        y_mat=torch.eye(l,device=self.device,dtype=self.para.torch_float_dtype)*y
                        z_mat=torch.eye(l,device=self.device,dtype=self.para.torch_float_dtype)*z
                        offset = data.orbit_offset[0][i]
                        Rx[offset:offset+l,offset:offset+l]=x_mat
                        Ry[offset:offset+l,offset:offset+l]=y_mat
                        Rz[offset:offset+l,offset:offset+l]=z_mat
                        if self.para.using_soc:
                            spin_offset = offset + orbital_dim
                            Rx[spin_offset:spin_offset+l,spin_offset:spin_offset+l]=x_mat
                            Ry[spin_offset:spin_offset+l,spin_offset:spin_offset+l]=y_mat
                            Rz[spin_offset:spin_offset+l,spin_offset:spin_offset+l]=z_mat

                    Rx=Rx.to_sparse_csr()
                    Ry=Ry.to_sparse_csr()
                    Rz=Rz.to_sparse_csr()


                    row_ptr_x = Rx.crow_indices()
                    col_ind_x = Rx.col_indices()
                    values_x = Rx.values()
                    f.write(f"{key[0]} {key[1]} {key[2]}\n")

                    f.write(f"{row_ptr_x.max()}\n")
                    f.write(" "+" ".join(f"{x.item():.{csr_digits}e}" for x in values_x) + "\n")
                    f.write(" "+" ".join(str(x.item()) for x in col_ind_x) + "\n")
                    f.write(" "+" ".join(str(x.item()) for x in row_ptr_x) + "\n")

                    row_ptr_y = Ry.crow_indices()
                    col_ind_y = Ry.col_indices()
                    values_y = Ry.values()

                    f.write(f"{row_ptr_y.max()}\n")
                    f.write(" "+" ".join(f"{x.item():.{csr_digits}e}" for x in values_y) + "\n")
                    f.write(" "+" ".join(str(x.item()) for x in col_ind_y) + "\n")
                    f.write(" "+" ".join(str(x.item()) for x in row_ptr_y) + "\n")

                    row_ptr_z = Rz.crow_indices()
                    col_ind_z = Rz.col_indices()
                    values_z = Rz.values()

                    f.write(f"{row_ptr_z.max()}\n")
                    f.write(" "+" ".join(f"{x.item():.{csr_digits}e}" for x in values_z) + "\n")
                    f.write(" "+" ".join(str(x.item()) for x in col_ind_z) + "\n")
                    f.write(" "+" ".join(str(x.item()) for x in row_ptr_z) + "\n")
                    pass
                else:
                    f.write(f"{key[0]} {key[1]} {key[2]}\n")
                    f.write("0\n")
                    f.write("0\n")
                    f.write("0\n")


                pass


        pass


    def save_with_incremental_filename(self,directory, file_name, format="pdf", transparent=True):
        """
        保存文件，如果文件名已存在，则自动递增编号。
        """
        if not os.path.exists(directory):
            os.makedirs(directory)  # 如果目录不存在，创建它

        base, ext = os.path.splitext(file_name)
        save_path = os.path.join(directory, file_name)
        counter = 1
        while os.path.exists(save_path):
            file_name = f"{base}{counter}{ext}"
            save_path = os.path.join(directory, file_name)
            counter += 1

        plt.savefig(save_path, format=format, transparent=transparent)
        return save_path

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            save_diff=[]
            save_ref=[]
            mae_all=0.0
            mae_dos_all=0.0
            for data in self._prediction_loader(self.testsetloader):
                data.to(self.device)
                if self.para.using_soc:
                    H_block, self.GraphEdgeIndex_to_BlockEdgeIndex,soc_val = self.model(data)
                    Eig_pre=self.compute_H_pre_soc(data, H_block,soc_val)
                    Eig_ref=data.Eig_ref
                else:
                    H_block, self.GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                    Eig_pre=self.compute_H_pre(data, H_block)
                    Eig_ref=data.Eig_ref
                if self.para.using_dosloss:
                    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
                else:
                    fig, ax = plt.subplots(1, 2, figsize=(12, 6))

                mse, mae = self.lossfunction.testloss_band(Eig_pre, Eig_ref,self.para.band_range,self.para.e_range)

                if self.para.using_dosloss:
                    dos,dos_ref,mse_dos,mae_dos=self.lossfunction.get_dos_loss( self.para.dos_sigma,10000,Eig_pre[0],Eig_ref[0],self.para.band_range,self.device,need_dos=True)
                    mae_dos_all+=mae_dos

                mae_all+=mae
                # 绘制图像
                Eig_ref=Eig_ref[...,self.para.band_range[0]:self.para.band_range[1]]
                Eig_pre=Eig_pre[...,self.para.band_range[0]:self.para.band_range[1]]
                ax[0].scatter(Eig_pre.to("cpu").flatten(), Eig_ref.to("cpu").flatten(), s=1)
                #ax[0].set_xlim(self.para.e_range[0], self.para.e_range[1])
                #ax[0].set_ylim(self.para.e_range[0], self.para.e_range[1])

                ax[1].plot(range(Eig_pre.shape[1]), Eig_pre[0].to("cpu").numpy(), color='red', linewidth=2)
                ax[1].plot(range(Eig_ref.shape[1]), Eig_ref[0].to("cpu").numpy(), color='grey',linestyle='--',dashes=(5, 5),  linewidth=2)
                ax[1].tick_params(direction='in')
                ax[1].tick_params(axis='x', length=0)
                ax[1].set_xlim(0, Eig_ref.shape[1] - 1)
                #ax[1].set_ylim(self.para.e_range[0], self.para.e_range[1])
                emin_ref,emax_ref,=Eig_ref.min().cpu(),Eig_ref.max().cpu()
                if self.para.using_dosloss:
                    ax[2].plot(dos.to("cpu").numpy(),np.linspace(emin_ref,emax_ref,10000,dtype=self.para.numpy_float_dtype), color='red', linewidth=1)
                    ax[2].plot(dos_ref.to("cpu").numpy(), np.linspace(emin_ref,emax_ref,10000,dtype=self.para.numpy_float_dtype),color='grey',linestyle='--',dashes=(5, 5),  linewidth=1)
                #ax[1].set_ylim(7.5, 10.5)


                save_directory = self.para.dump_sparseH
                file_name = "figure.pdf"
                saved_path = self.save_with_incremental_filename(save_directory, file_name)

                if self.para.using_dosloss:
                    print(f"prediction  MSE-EIG:  {mse:.5f}, MAE-EIG:  {mae:.5f}, MSE-EIG_dos:  {mse_dos:.5f}, MAE-EIG_dos:  {mae_dos:.5f},  {saved_path}")
                else:
                    print(f"prediction  MSE-EIG:  {mse:.5f}, MAE-EIG:  {mae:.5f},  {saved_path}")
                    
                file_name = "eigs.npy"
                saved_path = self.save_with_incremental_filename(save_directory, file_name)
                np.save(saved_path,Eig_pre.to("cpu"))
                # 显示图像
                plt.show()
                save_ref.append(Eig_ref.to("cpu")[0].flatten())
                save_diff.append((Eig_pre.to("cpu")[0]-Eig_ref.to("cpu")[0]).flatten())

            print(f"MAE-average:  {mae_all/len(self.testsetloader):.5f}")
            if self.para.using_dosloss:
                print(f"MAE-average_dos:  {mae_dos_all/len(self.testsetloader):.5f}")


            plt.clf()
            save_ref=torch.cat(save_ref).to("cpu").numpy()
            save_diff=torch.cat(save_diff).to("cpu").numpy()
            save_pre=save_ref+save_diff

            y_true = save_ref.astype(self.para.numpy_float_dtype, copy=False)
            y_pred = save_pre.astype(self.para.numpy_float_dtype, copy=False)
            ss_res = np.sum((y_true - y_pred) ** 2, dtype=self.para.numpy_float_dtype)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2, dtype=self.para.numpy_float_dtype)
            r2 = 1 - (ss_res / ss_tot)
            ax_hist = plt.gca()
            ax_hist.text(0.05, 0.95, f'$R^2 = {r2:.8f}$', transform=ax_hist.transAxes,fontsize=12, verticalalignment='top',bbox=dict(facecolor='white', alpha=0.6))

            np.savetxt(os.path.join(save_directory, "scatter_distx.txt"),save_pre)
            np.savetxt(os.path.join(save_directory, "scatter_disty.txt"),save_diff)

            plt.hist(save_diff, bins=50, edgecolor=None)
            plt.autoscale(enable=True, axis='both')
            plt.xlim(-0.1, 0.1)  # 设置 x 轴范围
            plt.autoscale(enable=True, axis='y')  # 只自动调整 y 轴
            plt.savefig("hist.pdf")
            
