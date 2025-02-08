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
    def __init__(self, net: nn.Module, train_loader: DataLoader, config: Config) -> None:
        self.net = net
        self.train_loader = train_loader
        self.config = config

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
            # self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            # self.optimizer,
            # lr_lambda=lambda step: cosine_annealing(
            #     step,
            #     config.optimizer.num_epochs * len(train_loader),
            #     1,

        plt.ion()
        self.fig_loss, self.ax_loss = plt.subplots()
        self.fig_acc, self.ax_acc = plt.subplots()

        self.epoch_losses = []
        self.epoch_accuracies = []

    def train_epoch(self, epoch_idx):
        self.net.train()
        loss_avg = 0.0
        acc_sum = 0.0
        batch_count = 0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(
            range(1, len(train_dataiter) + 1),
            desc=f"Epoch {epoch_idx:03d}: ",
            position=0,
            leave=True,
            disable=not comm.is_main_process(),
        ):
            batch = next(train_dataiter)
            data = batch["data"].cuda()
            target = batch["label"].cuda()

            # Forward pass
            logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier, target)

            # Accuracy calculation
            with torch.no_grad():
                preds = logits_classifier.argmax(dim=1)
                batch_acc = (preds == target).float().mean().item()

            # Exponential moving average of loss for logging
            loss_avg = loss_avg * 0.8 + float(loss) * 0.2

            acc_sum += batch_acc
            batch_count += 1

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Step the scheduler after each epoch
        self.scheduler.step()

        if epoch_idx >= 90:
            save_dir = (
                f"./results/"
                f"{self.config.dataset.name}_{self.net.__class__.__name__.lower()}_"
                f"base_e300_lr0.1_pixmix"
            )
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"e{epoch_idx}.pth")
            torch.save(
                {
                    "epoch": epoch_idx,
                    "state_dict": self.net.state_dict(),
                    "optimizer_state_dict": self.optimizer.state_dict(),
                    "scheduler_state_dict": self.scheduler.state_dict(),
                    "config": self.config,
                },
                save_path,
            )
            print(f"[INFO] Model and optimizer saved at: {save_path}")

        epoch_accuracy = acc_sum / batch_count if batch_count > 0 else 0.0

        metrics = {
            "epoch_idx": epoch_idx,
            "loss": self.save_metrics(loss_avg),
            "accuracy": epoch_accuracy,
        }

        self.epoch_losses.append(metrics["loss"])
        self.epoch_accuracies.append(metrics["accuracy"])
        self.ax_loss.clear()
        self.ax_loss.plot(self.epoch_losses, label="Training Loss")
        self.ax_loss.set_xlabel("Epoch")
        self.ax_loss.set_ylabel("Loss")
        self.ax_loss.set_title("Training Loss Over Epochs")
        self.ax_loss.legend()
        self.fig_loss.canvas.draw()
        self.fig_loss.canvas.flush_events()

        self.ax_acc.clear()
        self.ax_acc.plot(self.epoch_accuracies, label="Training Accuracy", color="green")
        self.ax_acc.set_xlabel("Epoch")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.set_title("Training Accuracy Over Epochs")
        self.ax_acc.legend()
        self.fig_acc.canvas.draw()
        self.fig_acc.canvas.flush_events()

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])
        return total_losses_reduced
