import torch
import openood
from openood.networks import ResNet18_32x32
from openood.networks import ResNet50, ViT_B_16
from torchvision.models import ResNet50_Weights, ViT_B_16_Weights
from torch.hub import load_state_dict_from_url
from openood.evaluators.metrics import auc_and_fpr_recall
from itertools import product
from tqdm import tqdm
import yaml
import numpy as np
from pprint import pprint
from openood.utils.config import Config
import argparse


def hyperparameter_search(dataset, resnet=True, batch_size=5000, device="cuda:0"):
    aurocs_max = 0
    if dataset in ["cifar10", "cifar100"]:
        models = [
            ResNet18_32x32(num_classes=100 if dataset == "cifar100" else 10)
            for _ in range(3)
        ]
        [
            model.load_state_dict(
                torch.load(
                    f"./OpenOOD/results/{dataset}_resnet18_32x32_base_e100_lr0.1_default/s{i}/best.ckpt"
                )
            )
            for i, model in enumerate(models)
        ]

    elif resnet:
        net = ResNet50()
        weights = ResNet50_Weights.IMAGENET1K_V1
        net.load_state_dict(load_state_dict_from_url(weights.url))

        models = [net]

    else:
        model = ViT_B_16()
        weights = ViT_B_16_Weights.IMAGENET1K_V1
        model.load_state_dict(load_state_dict_from_url(weights.url))
        models = [model]

    models = [model.to(device).eval() for model in models]
    with open("./OpenOOD/configs/postprocessors/weiper_kldiv.yml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    sweep = config["postprocessor"]["postprocessor_sweep"]
    config_weiper = Config("./OpenOOD/configs/postprocessors/weiper_kldiv.yml")
    weiper = [
        openood.postprocessors.WeiPerKLDivPostprocessor(config_weiper) for _ in models
    ]
    if resnet:
        latents = torch.load(f"./storage/{dataset}_latents.pth")
    else:
        latents = torch.load(f"./storage/{dataset}_vit_latents.pth")
    for perturbation_distance, n_bins in tqdm(
        product(sweep["perturbation_distance_list"], sweep["n_bins_list"]),
        total=len(sweep["perturbation_distance_list"]) * len(sweep["n_bins_list"]),
    ):
        for i, model in enumerate(models):
            weiper[i].n_bins = n_bins
            weiper[i].perturbation_distance = perturbation_distance
            weiper[i].setup(
                model,
                None,
                None,
                latents_loader=latents[i]["id"]["train"].split(batch_size),
            )
        for lambda_1, lambda_2, smoothing, smoothing_perturbed in product(
            sweep["lambda_1_list"],
            sweep["lambda_2_list"],
            sweep["smoothing_list"],
            sweep["smoothing_perturbed_list"],
        ):
            weiper[i].lambda_1 = lambda_1
            weiper[i].lambda_2 = lambda_2
            weiper[i].smoothing = smoothing
            weiper[i].smoothing_perturbed = smoothing_perturbed
            aurocs = []
            for i, model in enumerate(models):
                id_val_score = []
                for val_data in latents[i]["id"]["val"].split(batch_size):
                    _, conf = weiper[i].postprocess(model, None, latents=val_data)
                    id_val_score.append(conf.cpu())
                id_val_score = torch.cat(id_val_score)
                ood_val_score = []
                for val_data in latents[i]["ood"]["val"].split(batch_size):
                    _, conf = weiper[i].postprocess(model, None, latents=val_data)
                    ood_val_score.append(conf.cpu())
                ood_val_score = torch.cat(ood_val_score)
                conf = torch.cat([id_val_score, ood_val_score])
                label = torch.cat(
                    [torch.zeros_like(id_val_score), -torch.ones_like(ood_val_score)]
                )

                auroc = auc_and_fpr_recall(conf, label, 0.95)[0]
                aurocs.append(auroc)
            if aurocs_max < np.mean(aurocs):
                aurocs_max = np.mean(aurocs)
                params_optimal = {
                    "n_repeats": 100,
                    "perturbation_distance": perturbation_distance,
                    "n_bins": n_bins,
                    "lambda_1": lambda_1,
                    "lambda_2": lambda_2,
                    "smoothing": smoothing,
                    "smoothing_perturbed": smoothing_perturbed,
                    "exact_minmax": True,
                    "n_samples_for_setup": 300000,
                }
    return aurocs_max, params_optimal


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--resnet", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=5000)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
    dataset = args.dataset
    resnet = args.resnet
    batch_size = args.batch_size
    aurocs_max, params_optimal = hyperparameter_search(
        dataset, resnet, batch_size, device
    )
    with open("./OpenOOD/configs/postprocessors/weiper_kldiv.yml", "r") as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    with open("./OpenOOD/configs/postprocessors/weiper_kldiv.yml", "w") as f:
        config["postprocessor"]["postprocessor_args"] = params_optimal
        yaml.dump(config, f)
    print("Maximal AUROC score on Val set: ", aurocs_max)
    print("Optimal hyperparameters: ")
    pprint(params_optimal)
