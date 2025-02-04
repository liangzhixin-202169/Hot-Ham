import torch
import os
import sys
from entrypoints.Parameters import Parameters
from entrypoints.Lossfun import LossRecord, Lossfunction
from data.DatasetPreprocess import DatasetPrepocess


class Kernel(torch.nn.Module):
    def __init__(self, para: Parameters):
        super().__init__()
        self.para = para
        self.intdtype = self.para.intdtype
        self.floatdtype = self.para.floatdtype
        self.device = self.para.device

        # Read dataset
        self.datapreprocess = DatasetPrepocess(para)
        self.trainloader = self.datapreprocess.trainset_loader
        self.valsetloader = self.datapreprocess.valset_loader
        self.testsetloader = self.datapreprocess.testset_loader

        # Initilize model, optimizer and lr_scheduler
        self.start_epoch = 0
        self.checkpoint_info = None
        if self.para.init_from_model is not None:
            self.model = torch.load(self.para.model_init_path)
            self.model.to(device=self.device)
            self.optimizer = getattr(torch.optim, para.optimizer)(self.model.parameters(),
                                                                  lr=self.para.lr,
                                                                  weight_decay=self.para.lambda_2)
            self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=para.gamma)
        else:
            if para.prediction == 2:
                from entrypoints.model_profile import Model
            else:
                from entrypoints.model import Model

            self.model = Model(para=para)
            self.model.to(device=self.device)
            self.optimizer = getattr(torch.optim, para.optimizer)(self.model.parameters(),
                                                                  lr=self.para.lr,
                                                                  weight_decay=self.para.lambda_2)
            if self.para.lr_scheduler == "ExponentialLR":
                self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=para.gamma)
            elif self.para.lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode="min", factor=self.para.factor, patience=self.para.patience, threshold=para.threshold)

            if self.para.init_from_checkpoint is not None:
                checkpoint = torch.load(self.para.init_from_checkpoint)
                self.start_epoch = checkpoint['epoch']
                checkpoint_weights = self.Version_Convertion(checkpoint['model_state_dict'], self.model.state_dict())
                # init_state = {k: v for k, v in checkpoint['model_state_dict'].items() if k in self.model.state_dict()}
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

        # Define loss function
        self.lossfunction = Lossfunction(para)
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

    def train(self):
        if self.checkpoint_info is not None:
            print(self.checkpoint_info)

        for epoch in range(self.start_epoch, self.start_epoch+self.para.epoch):
            self.model.train()
            self.train_lossrecord.reset()
            self.val_lossrecord.reset()
            self.test_lossrecord.reset()

            for data in self.trainloader:
                self.optimizer.zero_grad()
                H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                mse, mae, num_ele = self.lossfunction.trainloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.AtomType_OrbitalSum, data)
                (torch.sqrt(mse)+mae).backward()
                self.optimizer.step()
                self.train_lossrecord.update(mse.item(), mae.item(), num_ele)

            if self.para.lr_scheduler == "ExponentialLR":
                current_lr = self.lr_scheduler.get_last_lr()[0]
            elif self.para.lr_scheduler == "ReduceLROnPlateau":
                current_lr = self.optimizer.state_dict()['param_groups'][0]['lr']
            info = f"Epoch:{epoch+1:>5}   lr: {current_lr:.6f}" + self.train_lossrecord.compute_info("Train")

            if ((epoch+1) % self.para.checkpoint_interval == 0):
                self.model.eval()
                with torch.no_grad():
                    for data in self.valsetloader:
                        H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                        val_loss_MSE, val_loss_MAE, num_ele = self.lossfunction.trainloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.AtomType_OrbitalSum, data)
                        self.val_lossrecord.update(val_loss_MSE.item(), val_loss_MAE.item(), num_ele)

                    for data in self.testsetloader:
                        H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                        test_loss_MSE, test_loss_MAE, num_ele = self.lossfunction.trainloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.AtomType_OrbitalSum, data)
                        self.test_lossrecord.update(test_loss_MSE.item(), test_loss_MAE.item(), num_ele)

                    self.save_checkpoint(epoch)
                    info += self.val_lossrecord.compute_info("Val")+self.test_lossrecord.compute_info("Test")

            print(info)
            if self.para.lr_scheduler == "ExponentialLR":
                self.lr_scheduler.step()
            elif self.para.lr_scheduler == "ReduceLROnPlateau":
                self.lr_scheduler.step(self.train_lossrecord.mae_ave)
            sys.stdout.flush()

    def eval(self):
        self.model.eval()
        with torch.no_grad():
            for data in self.trainloader:
                H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                train_loss_MSE, train_loss_MAE, num_ele = self.lossfunction.testloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.AtomType_OrbitalSum, data)
                self.train_lossrecord.update(train_loss_MSE.item(), train_loss_MAE.item(), num_ele)

            for data in self.valsetloader:
                H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                val_loss_MSE, val_loss_MAE, num_ele = self.lossfunction.testloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.AtomType_OrbitalSum, data)
                self.val_lossrecord.update(val_loss_MSE.item(), val_loss_MAE.item(), num_ele)

            for data in self.testsetloader:
                H_block, GraphEdgeIndex_to_BlockEdgeIndex = self.model(data)
                test_loss_MSE, test_loss_MAE, num_ele = self.lossfunction.testloss_ham(H_block, GraphEdgeIndex_to_BlockEdgeIndex, self.model.AtomType_OrbitalSum, data)
                self.test_lossrecord.update(test_loss_MSE.item(), test_loss_MAE.item(), num_ele)

            info = self.train_lossrecord.eval_info("Train") +\
                self.val_lossrecord.eval_info("Val") +\
                self.test_lossrecord.eval_info("Test")
            print(info)

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
                new_words = []
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
                current_version[key] = old_version["N_average"]

        return current_version
