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

    @torch.no_grad()
    def setup(
        self,
        net: nn.Module,
        id_loader_dict,
        ood_loader_dict,
        id_name="imagenet",
        valid_num=None,
        layer_names=None,
        aps=None,
        use_cache=False,
        hyperparameter_search=False,
        latents_loader=None,
        **kwargs,
    ):
        net.eval()
        device = next(iter(net.parameters())).device
        train_dl = id_loader_dict["train"] if latents_loader is None else latents_loader
        self.W_tilde = None
        (latents_min, latents_max) = (np.inf, -np.inf)
        (weiper_latents_min, weiper_latents_max) = (np.inf, -np.inf)
        if self.exact_minmax:
            for i, batch in enumerate(train_dl):
                if latents_loader is None:
                    data = batch["data"]
                    latents = net(data.to(device), return_feature=True)[1]
                else:
                    latents = batch
                weiper_latents, self.W_tilde = calculate_weiper_space(
                    net,
                    latents,
                    self.W_tilde,
                    device,
                    self.perturbation_distance,
                    self.n_repeats,
                )
                latents_min = (
                    latents.min() if latents.min() < latents_min else latents_min
                )
                latents_max = (
                    latents.max() if latents.max() > latents_max else latents_max
                )
                weiper_latents_min = (
                    weiper_latents.min()
                    if weiper_latents.min() < weiper_latents_min
                    else weiper_latents_min
                )
                weiper_latents_max = (
                    weiper_latents.max()
                    if weiper_latents.max() > weiper_latents_max
                    else weiper_latents_max
                )
                self.train_min_max = (latents_min, latents_max), (
                    weiper_latents_min,
                    weiper_latents_max,
                )
        else:
            self.train_min_max = None
        for i, batch in enumerate(train_dl):
            if latents_loader is None:
                data = batch["data"]
                latents = net(data.to(device), return_feature=True)[1]
            else:
                latents = batch
            if i == 0:
                self.train_densities = torch.zeros(self.n_bins, latents.shape[0])
                self.train_densities_weiper = torch.zeros(self.n_bins, latents.shape[0])

            (
                train_densities,
                train_densities_weiper,
                self.train_min_max,
                self.W_tilde,
            ) = calculate_WeiPerKLDiv_score(
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
                train_min_max=self.train_min_max,
            )
            self.train_densities += train_densities
            self.train_densities_weiper += train_densities_weiper

        self.train_densities /= len(train_dl)
        self.train_densities_weiper /= len(train_dl)
        self.train_densities = self.train_densities.mean(dim=-1)[:, None]
        self.train_densities_weiper = self.train_densities_weiper.mean(dim=-1)[:, None]
        self.train_densities = (
            self.train_densities,
            self.train_densities_weiper,
        )

    @torch.no_grad()
    def postprocess(self, net: nn.Module, data: Any, latents: torch.Tensor = None):
        net.eval()
        if latents is not None:
            feature = latents
            pred = None
        else:
            output, feature = net(data, return_feature=True)
            pred = torch.max(torch.softmax(output, dim=1), dim=1)[1]
        device = next(iter(net.parameters())).device
        conf = calculate_WeiPerKLDiv_score(
            net,
            feature,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
            n_bins=self.n_bins,
            n_repeats=self.n_repeats,
            smoothing=self.smoothing,
            smoothing_perturbed=self.smoothing_perturbed,
            perturbation_distance=self.perturbation_distance,
            train_densities=self.train_densities,
            train_min_max=self.train_min_max,
            device=device,
            W_tilde=self.W_tilde,
        )
        return pred, conf

    def set_hyperparam(self, hyperparam: list):
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
