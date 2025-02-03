import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from openood.utils import Config

import sys
from OpenOOD.openood.networks.resnet18_32x32 import ResNet18_32x32

class FCTrainer:
    def __init__(self, config_path: str, checkpoint_path: str):
        self.config = Config(config_path)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = ResNet18_32x32(num_classes=10).to(self.device)

        if checkpoint_path:
            print(f"Loading pretrained checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint)
            body_state_dict = {
                k: v for k, v in self.model.state_dict().items()
                if not k.startswith('fc.')
            }
            body_save_path = '/media/dc-04-vol03/omar/Bachelor/weiper/Weiper/checkpoints/body/resnet18_body.pth'
            os.makedirs(os.path.dirname(body_save_path), exist_ok=True)
            torch.save(body_state_dict, body_save_path)
            print(f"Body saved at {body_save_path}")

        for name, param in self.model.named_parameters():
            if not name.startswith('fc.'):
                param.requires_grad = False

        self.optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=self.config.optimizer.lr,
            momentum=self.config.optimizer.momentum,
            weight_decay=self.config.optimizer.weight_decay
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=50, gamma=0.2)

        self.criterion = nn.CrossEntropyLoss()

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, padding=4),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=4)

    def train(self):
        self.model.train()
        for epoch in range(1, 101): 
            epoch_loss = 0
            for batch in tqdm(self.train_loader, desc=f"Epoch {epoch}"):
                inputs, targets = batch
                inputs, targets = inputs.to(self.device), targets.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

            self.scheduler.step()

            fc_save_dir = '/media/dc-04-vol03/omar/Bachelor/weiper/Weiper/checkpoints/fc_layers/'
            os.makedirs(fc_save_dir, exist_ok=True)
            fc_save_path = os.path.join(fc_save_dir, f'fc_epoch_{epoch}.pth')
            torch.save(self.model.fc.state_dict(), fc_save_path)
            print(f"FC layer saved at {fc_save_path}")

            acc = self.evaluate()
            print(f"Epoch {epoch}, Loss: {epoch_loss / len(self.train_loader):.4f}, Accuracy: {acc:.2f}%")

    def evaluate(self):
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return 100 * correct / total

if __name__ == '__main__':
    config_path = '/media/dc-04-vol03/omar/Bachelor/weiper/Weiper/OpenOOD/configs/pipelines/train/baseline.yml'  
    checkpoint_path = '/media/dc-04-vol03/omar/Bachelor/weiper/Weiper/OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best_epoch96_acc0.9470.ckpt'
    trainer = FCTrainer(config_path=config_path, checkpoint_path=checkpoint_path)
    trainer.train()
