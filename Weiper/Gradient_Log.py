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

##############################
# Additional Libraries for Normality Tests
##############################
from scipy.stats import shapiro, normaltest, kstest, probplot

##############################
# Configuration
##############################
DATA_ROOT = "./data"  # Directory to store/download CIFAR-10
CHECKPOINT_PATH = "./OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt"

OUTPUT_CSV = "gradient_stats.csv"            # CSV with batch-wise mean,std
OUTPUT_PLOT = "gradient_noise_subplots.png"  # Rolling stats figure
OUTPUT_QQ_PLOT = "gradient_noise_QQplot.png" # Q-Q plot for distribution check

BATCH_SIZE = 128
NUM_WORKERS = 4
NUM_CLASSES = 10
EPOCHS = 1               # We'll run 1 epoch just to gather data
WINDOW_SIZE = 20         # Rolling window size for smoothing
FONT_SIZE = 14

##############################
# Define CIFAR-10 Transforms
##############################
transform = T.Compose([
    T.RandomCrop(32, padding=4),
    T.RandomHorizontalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.4914, 0.4822, 0.4465],
                std=[0.2470, 0.2435, 0.2616])
])

train_set = datasets.CIFAR10(root=DATA_ROOT, train=True,
                             download=True, transform=transform)
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=NUM_WORKERS,
                          pin_memory=True)

##############################
# Load Pretrained Model
##############################
from OpenOOD.openood.networks.resnet18_32x32_copy import ResNet18_32x32
model = ResNet18_32x32(num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(CHECKPOINT_PATH))
model.cuda()

##############################
# Setup Loss
##############################
criterion = nn.CrossEntropyLoss()
model.train()

##############################
# For Logging
##############################
gradient_logs = []
# We'll accumulate ALL final-layer gradients (flattened).
all_final_grads = []

##############################
# Gradient Logging Loop
##############################
for epoch in range(EPOCHS):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        model.zero_grad(set_to_none=True)

        output = model(data)
        loss = criterion(output, target)
        loss.backward()

        # final-layer gradient => shape [num_classes, last_layer_dim]
        final_layer_grad = model.fc.weight.grad.detach().cpu().numpy().flatten()

        mean_grad_norm = np.linalg.norm(final_layer_grad.mean())
        std_grad = final_layer_grad.std()

        gradient_logs.append({
            'epoch': epoch,
            'batch': batch_idx,
            'mean_grad_norm': mean_grad_norm,
            'std_grad': std_grad
        })

        # Append entire gradient vector for global distribution testing
        all_final_grads.append(final_layer_grad)

##############################
# Save Logs to CSV
##############################
df = pd.DataFrame(gradient_logs)
df.to_csv(OUTPUT_CSV, index=False)
print(f"[INFO] Saved batch-wise gradient stats to {OUTPUT_CSV}")

##############################
# Convert 'all_final_grads' into ONE big array
##############################
all_final_grads = np.concatenate(all_final_grads, axis=0)  # shape [N_total_grad_elems]
print(f"[DEBUG] all_final_grads shape => {all_final_grads.shape}")

##############################
# Normality Tests
##############################
# 1) Shapiro–Wilk test
shapiro_stat, shapiro_p = shapiro(all_final_grads[:5000])  
# NOTE: Shapiro can be very slow with large arrays, so we often use a subsample (like 5k).
print(f"[Shapiro] W-stat={shapiro_stat:.4f}, p-value={shapiro_p:.4e}")

# 2) D’Agostino’s K^2 normality test
dagostino_stat, dagostino_p = normaltest(all_final_grads[:20000])  
# again possibly only a subsample
print(f"[D'Agostino] stat={dagostino_stat:.4f}, p={dagostino_p:.4e}")

# 3) Kolmogorov–Smirnov test comparing to normal(μ,σ)
mu_ = np.mean(all_final_grads)
sigma_ = np.std(all_final_grads)
standardized = (all_final_grads - mu_) / (sigma_ + 1e-12)
ks_stat, ks_p = kstest(standardized[:30000], 'norm')  d
print(f"[K-S Test] stat={ks_stat:.4f}, p={ks_p:.4e}")
print(f"[DEBUG] Mean={mu_:.6f}, StdDev={sigma_:.6f}")

##############################
# Q–Q Plot: Visual Check
##############################
plt.figure(figsize=(6, 6))
probplot(standardized[:30000], dist="norm", plot=plt)  # Q-Q plot of standardized data
plt.title("Q-Q Plot of Final-Layer Gradients (Standardized)\n(Subsample of 30k points)")
plt.savefig(OUTPUT_QQ_PLOT, dpi=300)
plt.close()
print(f"[INFO] Q-Q plot saved => {OUTPUT_QQ_PLOT}")

##############################
# Rolling/Smoothed Plot of (mean_grad_norm, std_grad)
##############################
df['mean_grad_norm_smooth'] = df['mean_grad_norm'].rolling(window=WINDOW_SIZE, center=True).mean()
df['std_grad_smooth'] = df['std_grad'].rolling(window=WINDOW_SIZE, center=True).mean()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top subplot: mean_grad_norm_smooth
ax1.plot(df['batch'], df['mean_grad_norm_smooth'],
         color='blue', linewidth=2)
ax1.set_ylabel('Mean Grad Norm', fontsize=FONT_SIZE, color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
ax1.grid(True, linestyle='--', alpha=0.7)
ax1.set_title('Gradient Noise Metrics Over Batches', fontsize=FONT_SIZE + 2)

# Bottom subplot: std_grad_smooth
ax2.plot(df['batch'], df['std_grad_smooth'],
         color='orange', linewidth=2)
ax2.set_xlabel('Batch Index', fontsize=FONT_SIZE)
ax2.set_ylabel('Std Grad (sigma)', fontsize=FONT_SIZE, color='orange')
ax2.tick_params(axis='y', labelcolor='orange')
ax2.grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.savefig(OUTPUT_PLOT, dpi=300)
plt.close()
print(f"[INFO] Rolling-stats figure saved => {OUTPUT_PLOT}")

##############################
# Interpretation
##############################
"""
This updated script:

1. Collects final-layer gradients for each batch in `all_final_grads`.
2. Performs Shapiro–Wilk, D'Agostino K², and Kolmogorov–Smirnov tests to assess normality.
3. Creates a Q–Q plot (saved as `gradient_noise_QQplot.png`) to visualize the distribution’s adherence to a Gaussian shape.

If the p-values from these tests are extremely small (e.g., < 1e-4), that suggests
the gradient distribution *deviates* significantly from normal. If p-values are
moderate or large, it indicates we *cannot* reject the null hypothesis of normality.

The Q–Q plot complements the numeric tests by showing if the tails are heavier
than a normal distribution would predict.
"""
