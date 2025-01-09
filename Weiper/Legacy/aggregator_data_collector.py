# aggregator_data_collector.py
import torch
import numpy as np
from OpenOOD.openood.evaluation_api import Evaluator
from OpenOOD.openood.networks import ResNet18_32x32

###############################################
# Adjustable parameters
dataset = "cifar10"  # ID dataset
data_root = "./data"
config_root = "./OpenOOD/configs/"
postprocessor_name = "weiper_kldiv"
batch_size = 2000
verbose = True
APS_mode = False
num_workers = 0

# Model checkpoints: we assume you have at least one trained model checkpoint
# If you have multiple checkpoints, pick one. For simplicity, pick s0:
checkpoint_path = f'./OpenOOD/results/{dataset}_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'

################################################
# Load the model and weights
print("[DEBUG] Loading model and weights...")
num_classes = 100 if dataset == "cifar100" else 10
model = ResNet18_32x32(num_classes=num_classes)
model.load_state_dict(torch.load(checkpoint_path))
model.cuda()
model.eval()
print("[DEBUG] Model loaded successfully from:", checkpoint_path)

# Initialize evaluator with the loaded model
evaluator = Evaluator(
    model,
    id_name=dataset,
    data_root=data_root,
    config_root=config_root,
    preprocessor=None,
    postprocessor_name=postprocessor_name,
    batch_size=batch_size,
    verbose=verbose,
    APS_mode=APS_mode,
    num_workers=num_workers,
)

# Collect OOD scores on ID validation set
print("[DEBUG] Collecting OOD scores on ID validation set (id/val)...")
id_pred_val, id_conf_val, id_gt_val = evaluator.postprocessor.inference(
    evaluator.net, evaluator.dataloader_dict["id"]["val"], progress=verbose
)
# id_conf_val: OOD scores for ID validation samples
scores_val_id = id_conf_val  # shape [N_id_val]
print("[DEBUG] ID validation scores shape:", scores_val_id.shape)

# Collect OOD scores on OOD validation set
print("[DEBUG] Collecting OOD scores on OOD validation set (ood/val)...")
ood_pred_val, ood_conf_val, ood_gt_val = evaluator.postprocessor.inference(
    evaluator.net, evaluator.dataloader_dict["ood"]["val"], progress=verbose
)
scores_val_ood = ood_conf_val  # shape [N_ood_val]
print("[DEBUG] OOD validation scores shape:", scores_val_ood.shape)

# Save to .npz file
output_file = "val_scores.npz"
np.savez(output_file, scores_val_id=scores_val_id, scores_val_ood=scores_val_ood)
print("[DEBUG] Saved validation scores to", output_file)
