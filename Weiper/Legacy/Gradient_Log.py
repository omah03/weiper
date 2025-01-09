import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.datasets as datasets
from torch.utils.data import DataLoader

###############################################
# Configuration
###############################################
DATA_ROOT = "./data"  # Directory to store/download CIFAR-10
CHECKPOINT_PATH = "./OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt"
OUTPUT_CSV = "gradient_stats.csv"
OUTPUT_PLOT = "gradient_noise_subplots.png"
BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_CLASSES = 10
EPOCHS = 1   # How many epochs to run to gather gradient logs
WINDOW_SIZE = 20  # Rolling window size for smoothing
FONT_SIZE = 14

###############################################
# Define CIFAR-10 Transforms and Dataset
###############################################
transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616])
])

train_set = datasets.CIFAR10(root=DATA_ROOT, train=True, download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

###############################################
# Load Pretrained ResNet18 Model
# Adjust import path as needed.
###############################################
from OpenOOD.openood.networks.resnet18_32x32_copy import ResNet18_32x32

model = ResNet18_32x32(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.cuda()

###############################################
# Setup Loss and Model
# No optimizer.step(), just forward/backward to log gradients.
###############################################
criterion = nn.CrossEntropyLoss()
model.train()

gradient_logs = []

###############################################
# Gradient Logging Loop
# Forward + backward per batch, log mean & std of final layer gradients.
# No weight updates to preserve the checkpoint state.
###############################################
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        model.zero_grad(set_to_none=True)
        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        final_layer_grad = model.fc.weight.grad.detach().cpu().numpy().flatten()

        mean_grad_norm = np.linalg.norm(final_layer_grad.mean())
        std_grad = final_layer_grad.std()

        gradient_logs.append({
            'epoch': epoch,
            'batch': batch_idx,
            'mean_grad_norm': mean_grad_norm,
            'std_grad': std_grad
        })

# Save logs to CSV
df = pd.DataFrame(gradient_logs)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Saved gradient stats to {OUTPUT_CSV}")

###############################################
# Smoothing and Plotting
# Create two subplots: top for mean_grad_norm, bottom for std_grad
###############################################
df['mean_grad_norm_smooth'] = df['mean_grad_norm'].rolling(window=WINDOW_SIZE, center=True).mean()
df['std_grad_smooth'] = df['std_grad'].rolling(window=WINDOW_SIZE, center=True).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top subplot: Mean Grad Norm (smoothed)
ax1.plot(df['batch'], df['mean_grad_norm_smooth'], color='blue', linewidth=2)
ax1.set_ylabel('Mean Grad Norm', fontsize=FONT_SIZE, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Use scientific notation for mean_grad_norm axis
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_title('Gradient Noise Metrics Over Batches', fontsize=FONT_SIZE+2)

# Bottom subplot: Std Grad (sigma, smoothed)
ax2.plot(df['batch'], df['std_grad_smooth'], color='orange', linewidth=2)
ax2.set_xlabel('Batch Index', fontsize=FONT_SIZE)
ax2.set_ylabel('Std Grad (sigma)', fontsize=FONT_SIZE, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()

print(f"[INFO] Plot saved to {OUTPUT_PLOT}")

###############################################
# Interpretation (for Thesis):
#
# By avoiding optimizer updates, this script records gradient statistics at a
# near-converged state. The top subplot shows that mean_grad_norm (even magnified
# with scientific notation) remains very small but varies slightly. The bottom subplot
# shows std_grad (sigma) on a separate scale, reflecting the residual noise level in
# the gradients. Smoothing both lines clarifies underlying trends, making the figure
# suitable for inclusion in a bachelor’s thesis.
###############################################
