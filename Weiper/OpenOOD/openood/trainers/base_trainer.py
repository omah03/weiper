import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

import openood.utils.comm as comm
from openood.utils import Config

class BaseTrainer:
    def __init__(
        self,
        net: nn.Module,
        train_loader: DataLoader,
        config: Config,
        freeze_body: bool = False,
        checkpoint_path: str = "/home/omar/weiper/Weiper/resultsFull/cifar10_resnet18_32x32_base_e300_lr0.1_pixmix/e175.pth",
    ) -> None:

        self.net = net
        self.train_loader = train_loader
        self.config = config
        self.freeze_body = freeze_body
        self.checkpoint_path = checkpoint_path

        # Default optimizer for the full model
        self.optimizer = torch.optim.SGD(
            net.parameters(),
            config.optimizer.lr,
            momentum=config.optimizer.momentum,
            weight_decay=config.optimizer.weight_decay,
            nesterov=True,
        )

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[50, 75, 90],
            gamma=0.2
        )

        self.start_epoch = 0
        if self.checkpoint_path is not None and os.path.isfile(self.checkpoint_path):
            checkpoint = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)

            if "state_dict" in checkpoint:
                self.net.load_state_dict(checkpoint["state_dict"], strict=False)
                print("[INFO] Loaded model weights from checkpoint.")
            else:
                self.net.load_state_dict(checkpoint, strict=False)
                print("[WARNING].")

            # Load optimizer + scheduler if present
            if "optimizer_state_dict" in checkpoint:
                self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
                print("[INFO] Loaded optimizer state from checkpoint.")
            if "scheduler_state_dict" in checkpoint:
                self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
                print("[INFO] Loaded scheduler state from checkpoint.")
            if "epoch" in checkpoint:
                self.start_epoch = checkpoint["epoch"]
                print(f"[INFO] Resuming from epoch {self.start_epoch}.")
        else:
            print("[WARNING")

        if self.freeze_body:
            for name, param in self.net.named_parameters():
                if not name.startswith("fc."):
                    param.requires_grad = False
            fc_params = [p for p in self.net.parameters() if p.requires_grad]
            self.optimizer = torch.optim.SGD(
                fc_params,
                config.optimizer.lr,
                momentum=config.optimizer.momentum,
                weight_decay=config.optimizer.weight_decay,
                nesterov=True,
            )
            print("[INFO] Body is frozen; only final layer (fc) is trainable.")

        plt.ion()
        self.fig_loss, self.ax_loss = plt.subplots()
        self.fig_acc, self.ax_acc = plt.subplots()
        self.epoch_losses = []
        self.epoch_accuracies = []

        self.update_w = []
        self.update_b = []

    def train_epoch(self, epoch_idx):
        
        self.update_w = []
        self.update_b = []
        
        self.net.train()

        loss_avg = 0.0
        acc_sum = 0.0
        batch_count = 0

        num_steps = len(self.train_loader)
        pbar = tqdm(self.train_loader, total=num_steps, desc=f"Epoch {epoch_idx}")

        for step, batch in enumerate(pbar, start=1):
            data = batch["data"].cuda()
            target = batch["label"].cuda()

            logits = self.net(data)
            loss = F.cross_entropy(logits, target)

            # Compute accuracy for logging
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                batch_acc = (preds == target).float().mean().item()

            # Exponential moving average for loss
            loss_avg = 0.8 * loss_avg + 0.2 * loss.item()
            acc_sum += batch_acc
            batch_count += 1

            self.optimizer.zero_grad()
            loss.backward()

            state_w = self.optimizer.state.get(self.net.fc.weight, {})
            state_b = self.optimizer.state.get(self.net.fc.bias, {})

            if "momentum_buffer" in state_w:
                update_w = state_w["momentum_buffer"].clone().detach().cpu()
            else:
                update_w = self.net.fc.weight.grad.clone().detach().cpu()

            if "momentum_buffer" in state_b:
                update_b = state_b["momentum_buffer"].clone().detach().cpu()
            else:
                update_b = self.net.fc.bias.grad.clone().detach().cpu()

            self.update_w.append(update_w)
            self.update_b.append(update_b)

            self.optimizer.step()

            pbar.set_postfix({"loss": loss_avg, "acc": batch_acc})

        self.scheduler.step()  
        epoch_accuracy = acc_sum / batch_count if batch_count > 0 else 0.0
        metrics = {
            "epoch_idx": epoch_idx,
            "loss": self.save_metrics(loss_avg),
            "accuracy": epoch_accuracy,
        }

        self.epoch_losses.append(metrics["loss"])
        self.epoch_accuracies.append(metrics["accuracy"])

        print(f"[Epoch {epoch_idx:03d}] Loss: {loss_avg:.4f}, Acc: {epoch_accuracy*100:.2f}%")

        updates_save_path = f"./final_layer_updates_epoch{epoch_idx}.pt"
        torch.save({
            "update_w": self.update_w,  
            "update_b": self.update_b,   
            "epoch": epoch_idx,
        }, updates_save_path)

        print(f"[INFO] Saved final-layer update vectors for epoch {epoch_idx} to {updates_save_path}.")

        return self.net, metrics  # so that train_pipeline can access them if needed

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])
        return total_losses_reduced
