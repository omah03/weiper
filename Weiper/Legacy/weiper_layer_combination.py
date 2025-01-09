import torch
import torch.nn as nn
import numpy as np
from pprint import pprint

# ---------------------------
# Metrics Functions
# ---------------------------
import numpy as np
from sklearn import metrics

def acc(pred, label):
    ind_pred = pred[label != -1]
    ind_label = label[label != -1]
    num_tp = np.sum(ind_pred == ind_label)
    acc = num_tp / len(ind_label) if len(ind_label) > 0 else 0
    return acc

def auc_and_fpr_recall(conf, label, tpr_th=0.95):
    ood_indicator = np.zeros_like(label)
    ood_indicator[label == -1] = 1
    fpr_list, tpr_list, thresholds = metrics.roc_curve(ood_indicator, -conf)
    idx = np.argmax(tpr_list >= tpr_th)
    fpr = fpr_list[idx] if idx < len(fpr_list) else fpr_list[-1]

    precision_in, recall_in, _ = metrics.precision_recall_curve(ood_indicator, -conf)
    precision_out, recall_out, _ = metrics.precision_recall_curve(1 - ood_indicator, conf)

    auroc = metrics.auc(fpr_list, tpr_list)
    aupr_in = metrics.auc(recall_in, precision_in)
    aupr_out = metrics.auc(recall_out, precision_out)
    return auroc, aupr_in, aupr_out, fpr

def compute_all_metrics(conf, label, pred):
    # Compute standard OOD detection metrics
    recall = 0.95
    auroc, aupr_in, aupr_out, fpr = auc_and_fpr_recall(conf, label, recall)
    accuracy = acc(pred, label)
    return [fpr*100, auroc*100, aupr_in*100, aupr_out*100, accuracy*100]

# ---------------------------
# Main Code
# ---------------------------
from OpenOOD.openood.networks import ResNet18_32x32
from OpenOOD.openood.evaluation_api.datasets import get_id_ood_dataloader, get_default_preprocessor
from OpenOOD.openood.utils.config import Config
from OpenOOD.openood.postprocessors.weiper_kldiv_postprocessor import WeiPerKLDivPostprocessor
from OpenOOD.openood.postprocessors.weiper_kldiv.utils import batch_histogram, UniformKernel, calculate_uncertainty

########################################
# Step 1: Load CIFAR-10 Model
########################################
print("[DEBUG] Step 1: Initializing CIFAR-10 model and loading weights...")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

dataset = "cifar10"
model = ResNet18_32x32(num_classes=10).to(device)
model_ckpt = "./OpenOOD/results/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt"
model.load_state_dict(torch.load(model_ckpt, map_location=device))
model.eval()
print("[DEBUG] Model loaded and set to eval mode.")

########################################
# Step 2: Setup Data for CIFAR-10 (ID) and CIFAR-100 (OOD)
########################################
print("[DEBUG] Step 2: Setting up CIFAR-10 as ID and CIFAR-100 as OOD dataloaders...")
dataset_name = "cifar10"
data_root = "./data"
preprocessor = get_default_preprocessor(dataset_name)
batch_size = 200
id_ood_dls = get_id_ood_dataloader(
    dataset_name,     # ID = CIFAR-10
    data_root,
    preprocessor,
    postprocessor_name="weiper_kldiv",
    batch_size=batch_size,
    num_workers=0
)

id_loader_dict = id_ood_dls["id"]
test_dl = id_loader_dict["test"]

# CIFAR-100 as OOD. Assume OpenOOD can provide CIFAR-100 OOD loader under "ood".
# If not directly available, you might need to specify it in the config or dataset paths.
ood_loader_dict = id_ood_dls["ood"]
# Let's assume CIFAR-100 is listed under 'far' or 'near' sets. Check keys:
# We'll try to find cifar100 in the OOD loaders. Adjust as necessary.
cifar100_dl = None
for split_name, loader_dict in ood_loader_dict.items():
    for ood_name, ood_dl in loader_dict.items():
        if "cifar100" in ood_name.lower():
            cifar100_dl = ood_dl
            break
    if cifar100_dl is not None:
        break

if cifar100_dl is None:
    raise ValueError("CIFAR-100 OOD loader not found in ood_loader_dict. Please adjust code or config.")

print("[DEBUG] CIFAR-100 OOD loader found.")

########################################
# Step 3: Initialize WeiPerKLDivPostprocessor
########################################
print("[DEBUG] Step 3: Initializing WeiPerKLDivPostprocessor...")
config = Config("./OpenOOD/configs/postprocessors/weiper_kldiv.yml")
postprocessor = WeiPerKLDivPostprocessor(config)

postprocessor.setup(
    net=model,
    id_loader_dict=id_loader_dict,
    ood_loader_dict=None,
    use_cache=False
)
print("[DEBUG] WeiPer postprocessor setup done.")

########################################
# Helper functions to compute scores
########################################
def compute_scores_postprocessor(dataloader, is_ood=False):
    conf_list = []
    pred_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to(device)
            curr_label = torch.ones(len(data))*-1 if is_ood else batch["label"]
            pred, conf = postprocessor.postprocess(model, data)
            conf_list.append(conf.cpu())
            pred_list.append(pred.cpu())
            label_list.append(curr_label.cpu())
    conf_list = torch.cat(conf_list)
    pred_list = torch.cat(pred_list)
    label_list = torch.cat(label_list)
    return pred_list, conf_list, label_list

########################################
# Step 4: Compute WeiPer+KLD at penultimate layer for ID and CIFAR-100 OOD
########################################
print("[DEBUG] Step 4: Computing WeiPer+KLD at penultimate layer for CIFAR-10 (ID) & CIFAR-100 (OOD)...")

id_pred_pen, id_conf_pen, id_label_pen = compute_scores_postprocessor(test_dl, is_ood=False)
ood_pred_pen, ood_conf_pen, ood_label_pen = compute_scores_postprocessor(cifar100_dl, is_ood=True)

penultimate_conf = torch.cat([id_conf_pen, ood_conf_pen])
penultimate_pred = torch.cat([id_pred_pen, ood_pred_pen])
penultimate_label = torch.cat([id_label_pen, ood_label_pen])
print("[DEBUG] Penultimate combined conf shape:", penultimate_conf.shape)

########################################
# Step 5: Compute Identity projection KLD at intermediate layer (no WeiPer)
########################################
print("[DEBUG] Step 5: Computing identity projection KLD at intermediate layer...")
intermediate_layer_name = "layer2"
features_list = []

def hook_fn(module, input, output):
    features_list.append(output.detach())

handle = None
for name, module in model.named_modules():
    if name == intermediate_layer_name:
        handle = module.register_forward_hook(hook_fn)
        break
if handle is None:
    raise ValueError(f"Could not find layer {intermediate_layer_name} in model.")

# Extract training features for the intermediate layer
print("[DEBUG] Extracting intermediate layer distributions from training data...")
train_features = []
model.eval()
with torch.no_grad():
    for batch in id_loader_dict["train"]:
        data = batch["data"].to(device)
        features_list.clear()
        _ = model(data)
        intermediate_feats = features_list[0].clone().cpu()
        train_features.append(intermediate_feats)

train_features = torch.cat(train_features, dim=0)
print("[DEBUG] Train intermediate layer features shape:", train_features.shape)

def compute_identity_density_reference(features, n_bins=100, device=device):
    if features.dim() > 2:
        features = features.view(features.size(0), -1)
    density, bins = batch_histogram(features, n_bins=n_bins, device=device, probability=True)
    density_mean = density.mean(dim=-1, keepdim=True)
    return density_mean, (bins[0], bins[-1])

n_bins_identity = 100
identity_train_density, identity_train_minmax = compute_identity_density_reference(train_features, n_bins=n_bins_identity)
print("[DEBUG] Identity projection train reference density shape:", identity_train_density.shape)

s_identity = 2
kernel_identity = UniformKernel(n_bins_identity, kernel_size=s_identity).to(device)

def compute_identity_kld_scores(dataloader, is_ood=False):
    conf_list = []
    pred_list = []
    label_list = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to(device)
            curr_label = torch.ones(len(data))*-1 if is_ood else batch["label"]

            features_list.clear()
            _ = model(data)
            test_feats = features_list[0].clone().to(device)
            if test_feats.dim() > 2:
                test_feats = test_feats.view(test_feats.size(0), -1)

            test_density, _ = batch_histogram(test_feats, n_bins=n_bins_identity, device=device, probability=True)

            uncertainties = []
            split_size = 100
            for split_feats in test_density.split(split_size, dim=-1):
                u = calculate_uncertainty(
                    split_feats,
                    identity_train_density.to(device),
                    kernel_identity,
                    eps=0.01,
                    symmetric=True,
                    device=device
                )
                uncertainties.append(u)
            uncertainties = torch.cat(uncertainties, dim=-1)
            # Negative uncertainties as score (higher = more ID)
            conf = -uncertainties

            # Get predictions for ID accuracy calculation:
            handle.remove() # remove hook temporarily
            logits = model(data)
            preds = logits.argmax(dim=1).cpu()
            # re-add hook
            for name, module in model.named_modules():
                if name == intermediate_layer_name:
                    handle = module.register_forward_hook(hook_fn)
                    break

            conf_list.append(conf.cpu())
            pred_list.append(preds)
            label_list.append(curr_label.cpu())

    conf_list = torch.cat(conf_list)
    pred_list = torch.cat(pred_list)
    label_list = torch.cat(label_list)
    return pred_list, conf_list, label_list

id_pred_idKLD, id_conf_idKLD, id_label_idKLD = compute_identity_kld_scores(test_dl, is_ood=False)
ood_pred_idKLD, ood_conf_idKLD, ood_label_idKLD = compute_identity_kld_scores(cifar100_dl, is_ood=True)

handle.remove()

identity_conf = torch.cat([id_conf_idKLD, ood_conf_idKLD])
identity_pred = torch.cat([id_pred_idKLD, ood_pred_idKLD])
identity_label = torch.cat([id_label_idKLD, ood_label_idKLD])
print("[DEBUG] Identity projection combined conf shape:", identity_conf.shape)

########################################
# Step 6: Combine penultimate WeiPer+KLD and intermediate identity KLD scores
########################################
print("[DEBUG] Step 6: Combining scores...")
combined_conf = 0.5 * (penultimate_conf + identity_conf)
combined_pred = penultimate_pred
combined_label = penultimate_label
print("[DEBUG] Combined scores shape:", combined_conf.shape)

########################################
# Step 7: Evaluate using compute_all_metrics
########################################
print("[DEBUG] Step 7: Evaluating combined scores...")

c = combined_conf.numpy()
l = combined_label.numpy()
p = combined_pred.numpy()

results = compute_all_metrics(c, l, p)
fpr95, auroc, aupr_in, aupr_out, accuracy = results

print("Results for CIFAR-10 (ID) vs CIFAR-100 (OOD):")
print(f"FPR@95:   {fpr95:.2f}")
print(f"AUROC:     {auroc:.2f}")
print(f"AUPR_IN:   {aupr_in:.2f}")
print(f"AUPR_OUT:  {aupr_out:.2f}")
print(f"ACC(ID):   {accuracy:.2f}")

print("[DEBUG] Done.")
