from typing import Any
import torch
import torch.nn as nn
from .base_postprocessor import BasePostprocessor
from .weiper_kldiv.utils import (
    calculate_WeiPerKLDiv_score,
    calculate_weiper_space,
)
import numpy as np
from tqdm import tqdm

class WeiPerKLDivPostprocessor(BasePostprocessor):
    def __init__(self, config):
        super().__init__(config)
        self.args = self.config.postprocessor.postprocessor_args
        self.args_dict = self.config.postprocessor.postprocessor_sweep
        self.lambda_1 = self.args.lambda_1
        self.lambda_2 = self.args.lambda_2
        self.exact_minmax = self.args.exact_minmax
        self.smoothing = self.args.smoothing
        self.smoothing_perturbed = self.args.smoothing_perturbed
        self.n_bins = self.args.n_bins
        self.perturbation_distance = self.args.perturbation_distance
        self.n_repeats = self.args.n_repeats
        self.n_samples_for_setup = self.args.n_samples_for_setup
        self.train_dataset_list = None
        self.layer_names = None

        # Store training densities and related info for each layer
        self.layer_train_info = {}

    def flatten_features(self, feat: torch.Tensor) -> torch.Tensor:
        # Convert [N, C, H, W] or [N, D, 1, 1] into [N, D]
        return feat.view(feat.size(0), -1)

    @torch.no_grad()
    def setup(
        self,
        net: nn.Module,
        id_loader_dict,
        ood_loader_dict,
        id_name="imagenet",
        valid_num=None,
        layer_names=[3,4],
        aps=None,
        use_cache=False,
        hyperparameter_search=False,
        latents_loader=None,
        **kwargs,
    ):
        print("[DEBUG] Entering WeiPerKLDivPostprocessor.setup")
        print("[DEBUG] layer_names received:", layer_names)
        self.layer_names = layer_names

        net.eval()
        device = next(iter(net.parameters())).device
        train_dl = id_loader_dict["train"] if latents_loader is None else latents_loader

        # Determine which layers to process
        if self.layer_names is not None and len(self.layer_names) > 0:
            layers_to_process = list(self.layer_names)
            if 4 not in layers_to_process:
                layers_to_process.append(4)
        else:
            layers_to_process = [4]

        print("[DEBUG] layers_to_process:", layers_to_process)

        # If we need exact minmax for penultimate layer:
        if self.exact_minmax:
            for i, batch in enumerate(train_dl):
                data = batch["data"].to(device)
                output, feature_list = net(data, return_feature_list=True)
                # Penultimate layer is index 4
                penultimate_latents = self.flatten_features(feature_list[4])
                if i == 0:
                    self.W_tilde = None
                pen_weiper_latents, self.W_tilde = calculate_weiper_space(
                    net,
                    penultimate_latents,
                    self.W_tilde,
                    device,
                    self.perturbation_distance,
                    self.n_repeats,
                )

                latents_min = penultimate_latents.min().item()
                latents_max = penultimate_latents.max().item()
                weiper_latents_min = pen_weiper_latents.min().item()
                weiper_latents_max = pen_weiper_latents.max().item()

                if i == 0:
                    pen_minmax = ((latents_min, latents_max), (weiper_latents_min, weiper_latents_max))
                else:
                    old_minmax = pen_minmax
                    pen_minmax = (
                        (min(old_minmax[0][0], latents_min), max(old_minmax[0][1], latents_max)),
                        (min(old_minmax[1][0], weiper_latents_min), max(old_minmax[1][1], weiper_latents_max))
                    )
            self.train_min_max = pen_minmax
        else:
            self.train_min_max = None
            self.W_tilde = None

        def compute_train_densities_for_layer(latents, is_penultimate=False):
            if is_penultimate:
                return calculate_WeiPerKLDiv_score(
                    net,
                    latents,
                    lambda_1=self.lambda_1,
                    lambda_2=self.lambda_2,
                    n_bins=self.n_bins,
                    n_repeats=self.n_repeats,
                    smoothing=self.smoothing,
                    smoothing_perturbed=self.smoothing_perturbed,
                    perturbation_distance=self.perturbation_distance,
                    device=device,
                    W_tilde=self.W_tilde,
                    train_min_max=self.train_min_max
                )
            else:
                return calculate_WeiPerKLDiv_score(
                    net,
                    latents,
                    lambda_1=self.lambda_1,
                    lambda_2=self.lambda_2,
                    n_bins=self.n_bins,
                    n_repeats=1,  # minimal perturbation
                    smoothing=self.smoothing,
                    smoothing_perturbed=self.smoothing_perturbed,
                    perturbation_distance=0.0,  # no perturbation
                    device=device,
                    train_min_max=None,
                    W_tilde=None,
                )

        layer_accumulators = {}
        for layer_idx in layers_to_process:
            key = 'penultimate' if layer_idx == 4 else layer_idx
            layer_accumulators[key] = {
                'dens': None,
                'weiper_dens': None,
                'count': 0,
                'train_min_max': None,
                'W_tilde': None
            }

        for i, batch in enumerate(train_dl):
            data = batch["data"].to(device)
            output, feature_list = net(data, return_feature_list=True)

            for layer_idx in layers_to_process:
                key = 'penultimate' if layer_idx == 4 else layer_idx
                latents = self.flatten_features(feature_list[layer_idx])
                is_penultimate = (layer_idx == 4)

                (
                    train_densities,
                    train_densities_weiper,
                    updated_min_max,
                    updated_W_tilde,
                ) = compute_train_densities_for_layer(latents, is_penultimate)

                if layer_accumulators[key]['dens'] is None:
                    layer_accumulators[key]['dens'] = train_densities
                    layer_accumulators[key]['weiper_dens'] = train_densities_weiper
                    layer_accumulators[key]['train_min_max'] = updated_min_max
                    layer_accumulators[key]['W_tilde'] = updated_W_tilde
                else:
                    layer_accumulators[key]['dens'] += train_densities
                    layer_accumulators[key]['weiper_dens'] += train_densities_weiper
                layer_accumulators[key]['count'] += 1

        for key, vals in layer_accumulators.items():
            c = vals['count']
            vals['dens'] = (vals['dens'] / c).mean(dim=-1)[:, None]
            vals['weiper_dens'] = (vals['weiper_dens'] / c).mean(dim=-1)[:, None]
            self.layer_train_info[key] = {
                'train_densities': (vals['dens'], vals['weiper_dens']),
                'train_min_max': vals['train_min_max'],
                'W_tilde': vals['W_tilde']
            }

        print("[DEBUG] Finished setup without errors. Stored training densities for layers:", list(self.layer_train_info.keys()))

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, latents: torch.Tensor = None):
        net.eval()
        device = next(iter(net.parameters())).device
        print("[DEBUG] Entering WeiPerKLDivPostprocessor.postprocess")

        if latents is not None:
            # Direct latents provided
            feature_list_used = [None, None, None, None, latents]
            pred = None
            print("[DEBUG] latents provided directly, skipping forward pass.")
            # only penultimate if present
            layers_to_process = ['penultimate'] if 'penultimate' in self.layer_train_info else []
        else:
            if self.layer_names is not None and len(self.layer_names) > 0:
                output, feature_list = net(data.to(device), return_feature_list=True)
                pred = torch.max(torch.softmax(output, dim=1), dim=1)[1]

                layers_to_process = list(self.layer_names)
                if 4 not in layers_to_process:
                    layers_to_process.append(4)
                feature_list_used = feature_list
                print("[DEBUG] postprocess layers_to_process:", layers_to_process)
            else:
                # No layer_names specified, just penultimate
                output, feature = net(data.to(device), return_feature=True)
                pred = torch.max(torch.softmax(output, dim=1), dim=1)[1]
                layers_to_process = ['penultimate'] if 'penultimate' in self.layer_train_info else []
                feature_list_used = [None, None, None, None, feature]

        ood_scores = []

        for layer_idx in layers_to_process:
            if layer_idx == 4 or layer_idx == 'penultimate':
                key = 'penultimate'
                is_penultimate = True
                latents_current = self.flatten_features(feature_list_used[4])
            else:
                key = layer_idx
                is_penultimate = False
                latents_current = self.flatten_features(feature_list_used[layer_idx])

            print(f"[DEBUG] Computing OOD score for layer key: {key}, shape {latents_current.shape}")
            train_densities, train_densities_weiper = self.layer_train_info[key]['train_densities']
            saved_train_min_max = self.layer_train_info[key]['train_min_max']
            saved_W_tilde = self.layer_train_info[key]['W_tilde']

            if is_penultimate:
                conf = calculate_WeiPerKLDiv_score(
                    net,
                    latents_current,
                    lambda_1=self.lambda_1,
                    lambda_2=self.lambda_2,
                    n_bins=self.n_bins,
                    n_repeats=self.n_repeats,
                    smoothing=self.smoothing,
                    smoothing_perturbed=self.smoothing_perturbed,
                    perturbation_distance=self.perturbation_distance,
                    train_densities=(train_densities, train_densities_weiper),
                    train_min_max=saved_train_min_max,
                    device=device,
                    W_tilde=saved_W_tilde,
                )
            else:
                conf = calculate_WeiPerKLDiv_score(
                    net,
                    latents_current,
                    lambda_1=self.lambda_1,
                    lambda_2=self.lambda_2,
                    n_bins=self.n_bins,
                    n_repeats=1,
                    smoothing=self.smoothing,
                    smoothing_perturbed=self.smoothing_perturbed,
                    perturbation_distance=0.0,
                    train_densities=(train_densities, train_densities_weiper),
                    train_min_max=None,
                    device=device,
                    W_tilde=None,
                )

            ood_scores.append(conf)

        final_ood_score = torch.mean(torch.stack(ood_scores), dim=0)
        print("[DEBUG] Combined OOD score shape:", final_ood_score.shape)
        print("[DEBUG] postprocess completed without errors.")
        return pred, final_ood_score

    def set_hyperparam(self, hyperparam: list):
        print("[DEBUG] Setting hyperparameters:", hyperparam)
        self.lambda_1 = hyperparam[0]
        self.lambda_2 = hyperparam[1]
        self.smoothing = hyperparam[2]
        self.smoothing_perturbed = hyperparam[3]
        self.n_bins = hyperparam[4]
        self.perturbation_distance = hyperparam[5]

    def get_hyperparam(self):
        return (
            self.lambda_1,
            self.lambda_2,
            self.smoothing,
            self.smoothing_perturbed,
            self.n_bins,
            self.perturbation_distance,
        )
