import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os
import numpy as np
from ase.units import Rydberg, Bohr
import sys
import time
import json
import platform
from ..entrypoints.parameters import Parameters
from ..entrypoints.lossfun import LossRecord, Lossfunction
from ..data.dataset_preprocess import DatasetPrepocess
from ..modules.common import BasicInfo
from collections import defaultdict
from contextlib import contextmanager, nullcontext

class EpochProfile:
    def __init__(self, enabled, device, output_path=None, metadata=None):
        self.enabled = enabled
        self.device = device
        self.times = defaultdict(float)
        self.gpu_events = defaultdict(list)
        self.gpu_totals = defaultdict(float)
        self.output_path = output_path
        self.metadata = metadata or {}
        self.batch = None
        if enabled and output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "w") as stream:
                stream.write(json.dumps({"record": "metadata", "schema_version": 1,
                    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "python": platform.python_version(),
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
                    "timing": "observed training/evaluation; no dedicated warmup or replay",
                    "scope": "nested inclusive forward scopes; do not sum parents and children",
                    "comparison": {"core": "TO2Linear including rotations; excludes radial MLP",
                        "layer": "radial MLP + core; excludes wrapper self connections",
                        "wrapper": "radial MLP + core + wrapper self connections; may include chunk concatenation",
                        "backward": "whole-model loss backward only; no per-layer backward attribution"},
                    **self.metadata}) + "\n")
        self.start = time.perf_counter()

    def begin_batch(self, data, phase, batch_index, dataset_index, operator_sample=False):
        if not self.enabled:
            return
        self.batch = {
            "record": "structure", "phase": phase, "batch_index": batch_index,
            "dataset_index": int(dataset_index), "num_graphs": int(data.num_graphs),
            "num_nodes": int(data.num_nodes),
            "num_edges": int(data.edge_index_hop.shape[1]),
            "num_self_edges": int(data.mask_sc.sum().item()),
            "dtype": str(data.d_hop.dtype), "grad_enabled": torch.is_grad_enabled(),
            "operator_profiler_active": operator_sample,
            "first_batch_in_phase": batch_index == 0,
        }
        self.batch_times = dict(self.times)
        self.scope_metadata = {}

    def describe(self, name, **metadata):
        if self.enabled and self.batch is not None:
            self.scope_metadata.setdefault(name, []).append(metadata)

    def _drain_events(self):
        self._synchronize()
        elapsed = {}
        for name, events in self.gpu_events.items():
            elapsed[name] = sum(start.elapsed_time(end) for start, end in events)
            self.gpu_totals[name] += elapsed[name] / 1000
        self.gpu_events.clear()
        return elapsed

    def end_batch(self):
        if not self.enabled or self.batch is None:
            return
        gpu_ms = self._drain_events()
        wall_ms = {name: (value - self.batch_times.get(name, 0)) * 1000
                   for name, value in self.times.items()
                   if value != self.batch_times.get(name, 0)}
        self.batch.update(wall_ms=wall_ms, cuda_stream_ms=gpu_ms,
                          scopes=self.scope_metadata)
        if self.output_path:
            with open(self.output_path, "a") as stream:
                stream.write(json.dumps(self.batch) + "\n")
        self.batch = None

    def _synchronize(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    @contextmanager
    def section(self, name):
        if not self.enabled:
            yield
            return
        self._synchronize()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._synchronize()
            self.times[name] += time.perf_counter() - start

    @contextmanager
    def gpu_section(self, name):
        if not self.enabled:
            yield
            return
        if self.device.type != "cuda":
            with self.section(name):
                yield
            return
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self.gpu_events[name].append((start, end))

    def report(self, epoch, names, rank, gpu_names=()):
        if not self.enabled:
            return
        self._drain_events()
        total = time.perf_counter() - self.start
        values = torch.tensor([total] + [self.times[name] for name in names], dtype=torch.float64)
        if dist.is_initialized():
            dist.all_reduce(values, op=dist.ReduceOp.MAX)
        if rank == 0:
            details = "  ".join(f"{name}:{value:.3f}s" for name, value in zip(names, values[1:].tolist()))
            print(f"Profile epoch {epoch + 1} (rank max)  total:{values[0].item():.3f}s  {details}", flush=True)
        if gpu_names:
            gpu_values = torch.tensor([
                self.gpu_totals[name]
                if self.device.type == "cuda" else self.times[name]
                for name in gpu_names
            ], dtype=torch.float64)
            if dist.is_initialized():
                dist.all_reduce(gpu_values, op=dist.ReduceOp.MAX)
            if rank == 0:
                details = "  ".join(f"{name}:{value:.3f}s" for name, value in zip(gpu_names, gpu_values.tolist()))
                print(f"Profile model CUDA stream epoch {epoch + 1} (rank max)  {details}", flush=True)


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
        if self.para.prediction in (-1, 0):
            self.train()
        elif self.para.prediction == 1:
            self.eval()
        elif self.para.prediction == 2:
            self.topyatb()

    def train(self):
        if self.para.rank == 0:
            if self.checkpoint_info is not None:
                print(self.checkpoint_info)

        for epoch in range(self.start_epoch, self.start_epoch + self.para.epoch):
            epoch_start = time.perf_counter()
            profile = EpochProfile(
                self.para.prediction == -1, self.device,
                os.path.join(self.para.model_save_path, "profile", f"epoch{epoch+1}_rank{self.para.rank}.jsonl"),
                {"epoch": epoch + 1, "rank": self.para.rank,
                 "lcaoversion": self.para.lcaoversion, "nn_split_scale": self.para.nn_split_scale,
                 "trainset": str(self.para.get("trainset")), "testset": str(self.para.get("testset"))})
            self.model.train()
            self.trainloader.sampler.set_epoch(epoch)
            self.train_lossrecord.reset()
            self.test_lossrecord.reset()

            train_iter = iter(self.trainloader)
            batch_index = 0
            profile_train_indices = iter(self.trainloader.sampler) if profile.enabled else None
            while True:
                with profile.section("train_data"):
                    try:
                        data = next(train_iter)
                    except StopIteration:
                        break
                    data.to(self.device)
                sample_ops = profile.enabled and epoch == self.start_epoch + min(1, self.para.epoch - 1) and batch_index == 0
                if profile.enabled:
                    profile.begin_batch(data, "train", batch_index, next(profile_train_indices), sample_ops)
                activities = [torch.profiler.ProfilerActivity.CPU]
                if self.device.type == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                with profile.section("train_forward"):
                    self.optimizer.zero_grad()
                    operator_context = torch.profiler.profile(activities=activities) if sample_ops else nullcontext()
                    with operator_context as forward_operator_profile:
                        H_block = self.model(data, profile=profile if profile.enabled else None, phase="train")
                if sample_ops and self.para.rank == 0:
                    sort_key = "self_cuda_time_total" if self.device.type == "cuda" else "self_cpu_time_total"
                    print(f"Forward operator sample: epoch {epoch + 1}, batch {batch_index}", flush=True)
                    print(forward_operator_profile.key_averages().table(sort_by=sort_key, row_limit=15), flush=True)
                    if self.device.type == "cuda":
                        print(forward_operator_profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=10), flush=True)
                with profile.section("train_loss"):
                    mse, mae, num_ele = self.lossfunction.ham_loss(H_block, self.model.module.AtomType_OrbitalSum, data)
                    loss = torch.sqrt(mse) + mae
                with profile.section("train_backward"):
                    operator_context = torch.profiler.profile(activities=activities) if sample_ops else nullcontext()
                    with operator_context as operator_profile:
                        loss.backward()
                if sample_ops and self.para.rank == 0:
                    sort_key = "self_cuda_time_total" if self.device.type == "cuda" else "self_cpu_time_total"
                    print(f"Backward operator sample: epoch {epoch + 1}, batch {batch_index}", flush=True)
                    print(operator_profile.key_averages().table(sort_by=sort_key, row_limit=15), flush=True)
                    if self.device.type == "cuda":
                        print(operator_profile.key_averages().table(sort_by="self_cpu_time_total", row_limit=10), flush=True)
                with profile.section("train_optimizer"):
                    self.optimizer.step()
                with profile.section("train_metrics"):
                    self.train_lossrecord.update(mse.item(), mae.item(), num_ele)
                profile.end_batch()
                batch_index += 1

            current_lr = self.optimizer.param_groups[0]["lr"]

            with profile.section("epoch_metrics"):
                mse_global, mae_global = self.train_lossrecord.global_loss(self.device, self.para.torch_float_dtype)
            if self.para.rank == 0:
                info_global = f"Epoch:{epoch+1}  lr:{current_lr:.7f}  Train_MSE:{mse_global:.7f}  Train_MAE:{mae_global:.7f}"

            with profile.section("scheduler"):
                if self.para.lr_scheduler == "ExponentialLR":
                    self.lr_scheduler.step()
                elif self.para.lr_scheduler == "ReduceLROnPlateau":
                    self.lr_scheduler.step(mae_global.item())

            self.model.eval()
            with torch.no_grad():
                test_iter = iter(self.testsetloader)
                test_batch_index = 0
                profile_test_indices = iter(self.testsetloader.sampler) if profile.enabled else None
                while True:
                    with profile.section("test_data"):
                        try:
                            data = next(test_iter)
                        except StopIteration:
                            break
                        data.to(self.device)
                    if profile.enabled:
                        profile.begin_batch(data, "test", test_batch_index, next(profile_test_indices))
                    with profile.section("test_forward"):
                        H_block = self.model(data, profile=profile if profile.enabled else None, phase="test")
                    with profile.section("test_loss"):
                        test_loss_MSE, test_loss_MAE, num_ele = self.lossfunction.ham_loss(H_block, self.model.module.AtomType_OrbitalSum, data)
                    with profile.section("test_metrics"):
                        self.test_lossrecord.update(test_loss_MSE.item(), test_loss_MAE.item(), num_ele)
                    profile.end_batch()
                    test_batch_index += 1

                if self.para.get("testset") is not None:
                    with profile.section("epoch_metrics"):
                        mse_global, mae_global = self.test_lossrecord.global_loss(self.device, self.para.torch_float_dtype)
                    if self.para.rank == 0:
                        info_global += f"  Test_MSE:{mse_global:.7f}  Test_MAE:{mae_global:.7f}"

            is_last_epoch = epoch + 1 == self.start_epoch + self.para.epoch
            if self.para.rank == 0 and ((epoch + 1) % self.para.checkpoint_interval == 0 or is_last_epoch):
                with profile.section("checkpoint"):
                    self.save_checkpoint(epoch)

            model_names = []
            for phase in ("train", "test"):
                model_names.extend((f"{phase}_embedding", f"{phase}_feature_init", f"{phase}_messagepassing"))
                if self.para.lcaoversion == -1:
                    for layer_index in range(len(self.model.module.GauntConv_layer)):
                        prefix = f"{phase}_gaunt{layer_index}"
                        model_names.extend((prefix, f"{prefix}_self_connection", f"{prefix}_prepare",
                                            f"{prefix}_tensor_product", f"{prefix}_postprocess", f"{prefix}_norm"))
                    model_names.extend((f"{phase}_so2", f"{phase}_so2_self_connection", f"{phase}_so2_prepare",
                                        f"{phase}_so2_tensor_product", f"{phase}_so2_postprocess", f"{phase}_so2_norm"))
                else:
                    model_names.append(f"{phase}_mag_embedding")
                    for layer_index in range(len(self.model.module.messagepassing.irreps_mids)):
                        model_names.extend((
                            f"{phase}_layer{layer_index}_interaction",
                            f"{phase}_layer{layer_index}_conv",
                            f"{phase}_layer{layer_index}_norm",
                        ))
                model_names.extend((f"{phase}_output_linear", f"{phase}_direct_product", f"{phase}_hblock"))
            profile.report(epoch, (
                "train_data", "train_forward", "train_loss", "train_backward", "train_optimizer",
                "train_metrics", "test_data", "test_forward", "test_loss", "test_metrics",
                "epoch_metrics", "scheduler", "checkpoint",
            ), self.para.rank, model_names)

            if self.para.rank == 0:
                info_global += f"  Time:{time.perf_counter() - epoch_start:.3f}s"
                print(info_global)
                sys.stdout.flush()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            info_global = ""
            if self.para.get("trainset") is not None:
                for data in self.trainloader:
                    data.to(self.device)
                    H_block = self.model(data)
                    train_loss_MSE, train_loss_MAE, num_ele = self.lossfunction.ham_loss(H_block, self.model.module.AtomType_OrbitalSum, data)
                    self.train_lossrecord.update(train_loss_MSE.item(), train_loss_MAE.item(), num_ele)

                mse_global, mae_global = self.train_lossrecord.global_loss(self.device, self.para.torch_float_dtype)
                mse_global_max, mae_global_max = self.train_lossrecord.global_max(self.device, self.para.torch_float_dtype)
                mse_global_min, mae_global_min = self.train_lossrecord.global_min(self.device, self.para.torch_float_dtype)
                if self.para.rank == 0:
                    info_global += (
                        f"Train:\n"
                        + f"  MSE(eV^2):{mse_global:.7f}  MAX:{mse_global_max:.7f}  MIN:{mse_global_min:.7f}\n"
                        + f"  MAE(eV):{mae_global:.7f}  MAX:{mae_global_max:.7f}  MIN:{mae_global_min:.7f}\n"
                    )

            if self.para.get("testset") is not None:
                for data in self.testsetloader:
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
                        + f"  MSE(eV^2):{mse_global:.7f}  MAX:{mse_global_max:.7f}  MIN:{mse_global_min:.7f}\n"
                        + f"  MAE(eV):{mae_global:.7f}  MAX:{mae_global_max:.7f}  MIN:{mae_global_min:.7f}\n"
                    )

            if not info_global and self.para.rank == 0:
                info_global = "No trainset or testset provided; evaluation skipped."

            if self.para.rank == 0:
                print(info_global)

    def topyatb(self):
        if self.para.rank != 0:
            return
        model = self.model.module if hasattr(self.model, "module") else self.model
        model.eval()
        write = dict.__getitem__(self.para, "write")
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

                def block_r(block, data, channel=0, spinor=False):
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
                    if spinor:
                        dim_matrix *= 2

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
                            mr = mr[:, channel].cpu().numpy() if mr.ndim == 4 else mr.cpu().numpy()
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
                            row_idx = ((2 * offset0 + (dim0 if channel >= 2 else 0)) if spinor else offset0)[:, None] + row_grid
                            col_idx = ((2 * offset1 + (dim1 if channel % 2 else 0)) if spinor else offset1)[:, None] + col_grid
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
                                dtype=self.para.numpy_complex_dtype if (self.para.soc or self.para.magnetism == "non-collinear") else self.para.numpy_float_dtype,
                            )
                        }, dim_matrix

                    return {
                        "index": np.concatenate(mr_index, axis=0),
                        "value": np.concatenate(mr_value, axis=0)
                    }, dim_matrix

                def write_h_csr(filename, matrix, dim_matrix):
                    with open(filename, "w") as f:
                        f.write("STEP: 0\n")
                        f.write(f"Matrix Dimension of H(R): {dim_matrix}\n")
                        f.write(f"Matrix number of H(R): {n_cell}\n")
                        for i_cell in range(n_cell):
                            cell_shift = unique_cell_shift[i_cell].tolist()
                            mask = matrix["index"][:, 0] == i_cell
                            indices = matrix["index"][mask][:, 1:]
                            values = matrix["value"][mask] / Rydberg
                            f.write(f"{cell_shift[0]} {cell_shift[1]} {cell_shift[2]} {len(values)}\n")
                            if len(values):
                                order = np.lexsort((indices[:, 1], indices[:, 0]))
                                indices, values = indices[order], values[order]
                                rows = indices[:, 0].astype(self.para.numpy_int_dtype, copy=False)
                                cols = indices[:, 1].astype(self.para.numpy_int_dtype, copy=False)
                                counts = np.bincount(rows, minlength=dim_matrix).astype(self.para.numpy_int_dtype, copy=False)
                                row_ptr = np.concatenate((np.zeros(1, dtype=self.para.numpy_int_dtype), np.cumsum(counts, dtype=self.para.numpy_int_dtype)))
                                f.write(" ".join(f"({x.real:.{csr_digits}e},{x.imag:.{csr_digits}e})" if np.iscomplexobj(x) else f"{x:.{csr_digits}e}" for x in values) + "\n")
                                f.write(" ".join(str(x) for x in cols) + "\n")
                                f.write(" ".join(str(x) for x in row_ptr) + "\n")

                def write_h_blocks(name, blocks):
                    spinor = self.para.soc or self.para.magnetism == "non-collinear"
                    if spinor:
                        matrices = [block_r(blocks, data, channel, True)[0] for channel in range(4)]
                        dim_matrix = 2 * int(sum(model.AtomType_OrbitalSum[t].item() for t in AtomType))
                        matrix = {"index": np.concatenate([m["index"] for m in matrices]),
                                  "value": np.concatenate([m["value"] for m in matrices]).astype(self.para.numpy_complex_dtype, copy=False)}
                        write_h_csr(f"{prefix}{name}_0.csr", matrix, dim_matrix)
                    else:
                        channels = 2 if self.para.magnetism == "collinear" else 1
                        for channel in range(channels):
                            matrix, dim_matrix = block_r(blocks, data, channel)
                            write_h_csr(f"{prefix}{name}_{channel}.csr", matrix, dim_matrix)

                csr_digits = 8 if self.para.numpy_float_dtype == np.float32 else 16
                prefix = "" if len(self.trainloader) == 1 else f"{batch_idx}_"

                if "h_ref" in write:
                    h_ref = data.HR
                    if hasattr(data, "iHR"):
                        h_ref = {a: {b: h_ref[a][b] + 1j * data.iHR[a][b]
                                     for b in h_ref[a]} for a in h_ref}
                    write_h_blocks("h_ref", rotate2abacus(h_ref, AtomType))

                if "olp" in write:
                    olp = rotate2abacus(data.SR, AtomType)
                    if self.para.soc or self.para.magnetism == "non-collinear":
                        olp_up, dim_matrix = block_r(olp, data, 0, True)
                        olp_down, _ = block_r(olp, data, 3, True)
                        olp = {
                            "index": np.concatenate((olp_up["index"], olp_down["index"])),
                            "value": np.concatenate((olp_up["value"], olp_down["value"])).astype(self.para.numpy_complex_dtype, copy=False),
                        }
                    else:
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

                if "rR" in write:
                    rR_x = rotate2abacus(data.rR["x"], AtomType)
                    rR_y = rotate2abacus(data.rR["y"], AtomType)
                    rR_z = rotate2abacus(data.rR["z"], AtomType)
                    if self.para.soc or self.para.magnetism == "non-collinear":
                        rR_x_up, dim_matrix = block_r(rR_x, data, 0, True)
                        rR_y_up, _ = block_r(rR_y, data, 0, True)
                        rR_z_up, _ = block_r(rR_z, data, 0, True)
                        rR_x_down, _ = block_r(rR_x, data, 3, True)
                        rR_y_down, _ = block_r(rR_y, data, 3, True)
                        rR_z_down, _ = block_r(rR_z, data, 3, True)
                        rR_x = {"index": np.concatenate((rR_x_up["index"], rR_x_down["index"])),
                                "value": np.concatenate((rR_x_up["value"], rR_x_down["value"]))}
                        rR_y = {"index": np.concatenate((rR_y_up["index"], rR_y_down["index"])),
                                "value": np.concatenate((rR_y_up["value"], rR_y_down["value"]))}
                        rR_z = {"index": np.concatenate((rR_z_up["index"], rR_z_down["index"])),
                                "value": np.concatenate((rR_z_up["value"], rR_z_down["value"]))}
                    else:
                        rR_x, dim_matrix = block_r(rR_x, data)
                        rR_y, _ = block_r(rR_y, data)
                        rR_z, _ = block_r(rR_z, data)

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

                if "h_pred" in write:
                    data.to(self.device)
                    h_pred = tensor2tensor(self.model(data), "cpu")
                    data.to("cpu")
                    write_h_blocks("h_pred", rotate2abacus(h_pred, AtomType))

    def save_checkpoint(self, epoch: int):
        checkpoint_dir = os.path.join(self.para.model_save_path, "checkpoint")
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_dir, f"cp{epoch+1}.pth")
        checkpoint = {
            "epoch": epoch + 1,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.lr_scheduler.state_dict(),
            "Train_MSE": self.train_lossrecord.mse_ave,
            "Train_MAE": self.train_lossrecord.mae_ave,
        }
        if self.para.get("testset") is not None:
            checkpoint.update(Test_MSE=self.test_lossrecord.mse_ave, Test_MAE=self.test_lossrecord.mae_ave)
        torch.save(checkpoint, checkpoint_file)

    def init_model(self):
        self.start_epoch = 0
        self.checkpoint_info = None
        checkpoint_path = self.para.get("init_from_checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False) if checkpoint_path else None
        if checkpoint is None:
            initial_lr = self.para.lr
        else:
            initial_lr = checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"]

        if self.para.lcaoversion == -1:
            from .model_lcao_legacy import Model
        else:
            # Versions 0 and 1 share the model; version 1 selects optimized TO2.
            from .model_lcao import Model

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
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.lr_scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

            if self.para.get("new_lr") is not None:
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = self.para.new_lr

            self.checkpoint_info = (
                f"Epoch:{checkpoint['epoch']}  lr:{self.optimizer.state_dict()['param_groups'][0]['lr']:.10f}  "
                f"Train_MSE:{checkpoint['Train_MSE']:.7f}  Train_MAE:{checkpoint['Train_MAE']:.7f}"
            )
            if self.para.get("testset") is not None and "Test_MSE" in checkpoint and "Test_MAE" in checkpoint:
                self.checkpoint_info += f"  Test_MSE:{checkpoint['Test_MSE']:.7f}  Test_MAE:{checkpoint['Test_MAE']:.7f}"
