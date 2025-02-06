import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

import openood.utils.comm as comm
from openood.utils import Config

from .lr_scheduler import cosine_annealing


class BaseTrainer:
    def __init__(
        self, net: nn.Module, train_loader: DataLoader, config: Config
    ) -> None:
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
        # s = [
        #     torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=60, gamma=0.2),
        #     torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=40, gamma=0.2),
        # ]
        # self.scheduler = lambda step: (s[0] if step < 120 else s[1])
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: cosine_annealing(
                step,
                config.optimizer.num_epochs * len(train_loader),
                1,
                1e-6 / config.optimizer.lr,
            ),
        )

    def train_epoch(self, epoch_idx):
        self.net.train()

        loss_avg = 0.0
        train_dataiter = iter(self.train_loader)

        for train_step in tqdm(
            range(1, len(train_dataiter) + 1),
            desc="Epoch {:03d}: ".format(epoch_idx),
            position=0,
            leave=True,
            disable=not comm.is_main_process(),
        ):
            batch = next(train_dataiter)
            data = batch["data"].cuda()
            target = batch["label"].cuda()

            # forward
            logits_classifier = self.net(data)
            loss = F.cross_entropy(logits_classifier, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            # exponential moving average, show smooth values
            with torch.no_grad():
                loss_avg = loss_avg * 0.8 + float(loss) * 0.2


        # Save checkpoint every 10 epochs (here at epochs where epoch_idx % 10 == 5)
        if epoch_idx >= 90:
            save_dir = f"./results/{self.config.dataset.name}_{self.net.__class__.__name__.lower()}_base_e300_lr0.1_pixmix"
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"e{epoch_idx}.pth")
            
            torch.save({
                "epoch": epoch_idx,
                "state_dict": self.net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": None if not callable(self.scheduler) else self.scheduler(epoch_idx).state_dict(),
                "config": self.config
                            }, save_path)

            print(f"[INFO] Model and optimizer saved at: {save_path}")

        metrics = {}
        metrics["epoch_idx"] = epoch_idx
        metrics["loss"] = self.save_metrics(loss_avg)

        return self.net, metrics

    def save_metrics(self, loss_avg):
        all_loss = comm.gather(loss_avg)
        total_losses_reduced = np.mean([x for x in all_loss])
        return total_losses_reduced
