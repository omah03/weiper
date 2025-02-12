import torch
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
import torch.nn.functional as F
from typing import Union
import torch.nn as nn
import os 


# @torch.jit.script
# def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
#     """
#     Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
#     Replicates but the multi-dimensional behaviour of numpy.linspace in PyTorch.
#     Source: https://github.com/pytorch/pytorch/issues/61292
#     """
#     # create a tensor of 'num' steps from 0 to 1
#     steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

#     for i in range(start.ndim):
#         steps = steps.unsqueeze(-1)

#     # the output starts at 'start' and increments until 'stop' in each dimension
#     out = start[None] + steps * (stop - start)[None]
#     return out


# @torch.no_grad()
# def batch_histogram(
#     samples: torch.Tensor,
#     n_bins: int = 100,
#     batch_size: int = 1000,
#     device: str = "cpu",
#     disable_tqdm: bool = False,
#     bins_min_max: tuple = None,
#     probability: bool = False,
# ) -> Union[torch.Tensor, torch.Tensor]:
#     """Generates a histogram (density, bins) for multiple dimensions for batch tensor shaped [n_samples, n_dims].

#     Args:
#         samples (torch.Tensor): Input samples.
#         n_bins (int, optional): Number of bins. Defaults to 100.
#         batch_size (int, optional): Batch size. Defaults to 200.
#         device (str, optional): Cuda device or CPU. Defaults to "cpu".
#         disable_tqdm (bool, optional): Whether to show progress bars (if True: disabled). Defaults to False.
#         bins_min_max (tuple, optional): Set min and max for the bins instead of calculating it from the samples. Defaults to None.
#         probability (bool, optional): If True, normalizes the densities to sum to one (outputs probabilities instead of densities). Defaults to False.

#     Returns:
#         Union[torch.Tensor, torch.Tensor]: density, bins (shaped [n_bins, n_dims], [n_bins + 1, n_dims])
#     """
#     eps = 1e-8
#     samples = samples.to(device)
#     if bins_min_max is None:
#         if not isinstance(samples, torch.Tensor):
#             min = next(iter(samples)).min(dim=0)[0]
#             max = next(iter(samples)).max(dim=0)[0]
#             for sample in samples:
#                 min = torch.where(sample.min(dim=0)[0] < min, sample.min(dim=0)[0], min)
#                 max = torch.where(sample.max(dim=0)[0] > max, sample.max(dim=0)[0], max)
#         else:
#             min = samples.min(dim=0)[0]
#             max = samples.max(dim=0)[0]
#         bins = linspace(min, max + eps, num=n_bins + 1)
#     else:
#         min = bins_min_max[0]
#         max = bins_min_max[1]
#         bins = linspace(min, max + eps, num=n_bins + 1)[:, None].repeat(
#             1, samples.shape[1]
#         )
#         bins = torch.cat(
#             [bins, torch.ones(1, bins.shape[-1]).to(bins.device) * np.inf], dim=0
#         )
#         bins = torch.cat(
#             [-torch.ones(1, bins.shape[-1]).to(bins.device) * np.inf, bins], dim=0
#         )
#     bin_length = bins[2:3] - bins[1:2] + 1e-12

#     bins = bins.to(device)
#     if not isinstance(samples, torch.Tensor) or samples.shape[0] > batch_size:
#         if isinstance(samples, torch.Tensor):
#             samples = samples.split(batch_size)
#         densities = torch.zeros(n_bins + 2, samples[0].shape[1])
#         for samples in tqdm(samples, disable=disable_tqdm):
#             samples = samples.to(device)

#             density = torch.logical_and(
#                 (samples[None] < bins[1:, None]),
#                 (samples[None] >= bins[:-1, None]),
#             )
#             density = density.to(torch.float32).mean(dim=1)
#             densities += density.cpu()
#         densities /= len(samples)
#         density = densities
#     else:
#         density = torch.logical_and(
#             (samples[None] < bins[1:, None]), (samples[None] >= bins[:-1, None])
#         )
#         density = density.to(torch.float32).mean(dim=1)
#     if not probability:
#         density /= bin_length.to(density.device)
#     else:
#         density = density[1:-1] / density.sum(dim=0)[None]
#     return density, bins


# class UniformKernel(torch.nn.Module):
#     """Uniform kernel for smoothing the density.

#     Args:
#         dim (int): Number of (latent/WeiPer) dimensions.
#         kernel_size (int, optional): Kernel size. Defaults to 10.
#     """

#     def __init__(
#         self,
#         dim,
#         kernel_size=10,
#     ):
#         super().__init__()
#         self.kernel = torch.nn.Conv1d(
#             dim, dim, kernel_size, padding="same", padding_mode="replicate"
#         )
#         self.kernel.weight.data = torch.zeros_like(self.kernel.weight.data)
#         self.kernel.weight.data[range(dim), range(dim)] = (
#             torch.ones(1, dim, kernel_size) / kernel_size
#         )
#         self.kernel.bias.data = torch.zeros_like(self.kernel.bias)
#         self.kernel.bias.requires_grad = False
#         self.kernel.weight.requires_grad = False

#     @torch.no_grad()
#     def forward(self, densities, bins=None):
#         # smoothing
#         smoothed_densities = self.kernel(densities.T).T
#         # correct densities to sum to 1.
#         if bins is not None:
#             smoothed_densities /= (smoothed_densities * (bins[1] - bins[0])).sum(dim=0)[
#                 None
#             ]
#         return smoothed_densities


# @torch.no_grad()
# def calculate_weiper_space(
#     model: nn.Module,
#     latents: torch.Tensor,
#     perturbed_fc: nn.Module = None,
#     device: str = "cpu",
#     fc_dir: str = "./checkpoints/fc_layers", # doesnt matter
#     start_epoch: int = 101, #doesnt matter
#     end_epoch: int = 200, #doesnt matter    
#     perturbation_distance: float = 2.1,
#     batch_size: int = 256,
# ) -> Union[torch.Tensor, nn.Module, int, int]:
#     """Creates the perturbed fully connected layer (curly H in the paper)
#     and calculates the perturbed logits from the penultimate latents.

#     Args:
#         model (torch.nn.Module): Instance of the neural network.
#         latents (torch.Tensor): The penultimate latents.
#         perturbed_fc (torch.nn.Module, optional): Precalculates perturbed fully connected layer. Defaults to None.
#         device (str, optional): Cuda device or CPU. Defaults to "cpu".
#         perturbation_distance (float, optional): (Relative) perturbation distance (delta in the paper). Defaults to 2.1.
#         n_repeats (int, optional): The number of repeats (r in the paper). Defaults to 50.
#         noise_proportional (bool, optional): Force a constant ratio between noise and class projections controlled by
#         perturbation_distance. Defaults to True.
#         constant_length (bool, optional): Normalize the noise to have length perturbation_distance. Defaults to True.
#         batch_size (int, optional): Batch size. Defaults to 200.
#         apply_softmax (bool, optional): Whether to apply softmax to the output. Defaults to False.
#         ablation_noise_only (bool, optional): Use weight independent random projections. Defaults to False.

#     Returns:
#         Union[torch.Tensor, torch.Tensor]: noise_logits, perturbed_fc
#     """

#     eps_norm = 1e-8 
#     snapshot_weights = []
#     snapshot_biases = []


#     ref_norm = model.fc.weight.norm(p=2, dim=1, keepdim=True) 
#     ref_norm = ref_norm.to(device) 

#     for ep in range(start_epoch, end_epoch + 1):
#         fc_path = os.path.join(fc_dir, f"fc_epoch_{ep}.pth")
#         if not os.path.exists(fc_path):
#             raise FileNotFoundError(f"[ERROR] Missing snapshot => {fc_path}")
#         fc_state = torch.load(fc_path, map_location=device)
#         if "fc_state_dict" in fc_state:
#             fc_state = fc_state["fc_state_dict"]
#         if "weight" not in fc_state and "fc.weight" in fc_state:
#             fc_state["weight"] = fc_state.pop("fc.weight")
#         if "bias" not in fc_state and "fc.bias" in fc_state:
#             fc_state["bias"] = fc_state.pop("fc.bias")
#         w = fc_state["weight"]
#         b = fc_state["bias"]  
#         norm = w.norm(p=2, dim=1, keepdim=True) + eps_norm
#         w_normalized = w / norm
#         w_scaled = w_normalized * (ref_norm * perturbation_distance)
#         w_scaled = w
#         snapshot_weights.append(w_scaled)
#         snapshot_biases.append(b)

#     stacked_weight = torch.cat(snapshot_weights, dim=0)  
#     stacked_bias   = torch.cat(snapshot_biases, dim=0)    

#     M = end_epoch - start_epoch + 1    
#     C = snapshot_weights[0].shape[0]       
#     D = snapshot_weights[0].shape[1]

#     def build_lin():
#         big_fc = nn.Linear(D, M * C, bias=True)
#         with torch.no_grad():
#             big_fc.weight.copy_(stacked_weight)
#             big_fc.bias.copy_(stacked_bias)
#         big_fc.to(device)
#         return big_fc

#     if perturbed_fc is None:
#         perturbed_fc = build_lin()
#     else:
#         if perturbed_fc.weight.shape != (M * C, D):
#             perturbed_fc = build_lin()
#         else:
#             with torch.no_grad():
#                 perturbed_fc.weight.copy_(stacked_weight)
#                 perturbed_fc.bias.copy_(stacked_bias)

#     print(f"[DEBUG] => Built big_fc with shape: {perturbed_fc.weight.shape}, bias: {perturbed_fc.bias.shape}")

#     output_list = []
#     for chunk in latents.split(batch_size):
#         out = perturbed_fc(chunk.to(device))
#         output_list.append(out.cpu())
#     weiper_logits = torch.cat(output_list, dim=0)

#     return weiper_logits, perturbed_fc, M, C

@torch.no_grad()
def calculate_weiper_space(
    model: nn.Module,
    latents: torch.Tensor,
    perturbed_fc: nn.Module = None,
    device: str = "cpu",
    fc_dir: str = "./checkpoints/fc_layers",  # directory with fc_epoch_{ep}.pth
    start_epoch: int = 101,
    end_epoch: int = 200,
    perturbation_distance: float = 2.1,
    batch_size: int = 256,
) -> Union[torch.Tensor, nn.Module, int, int]:
    """
    For each snapshot from [start_epoch..end_epoch], we load the fc layer
    and add a small random perturbation (scaled by 'perturbation_distance').
    Then we stack them all => big_fc => apply to latents.

    Returns:
        weiper_logits (torch.Tensor): shape [B, M*C], where M = #snapshots
        perturbed_fc (nn.Module): the big fc layer
        M, C (int): number of snapshots, number of classes
    """
    eps_norm = 1e-8
    snapshot_weights = []
    snapshot_biases = []

    ref_norm = model.fc.weight.norm(p=2, dim=1, keepdim=True) + eps_norm
    ref_norm = ref_norm.to(device)

    for ep in range(start_epoch, end_epoch + 1):
        fc_path = os.path.join(fc_dir, f"fc_epoch_{ep}.pth")
        if not os.path.exists(fc_path):
            raise FileNotFoundError(f"[ERROR] Missing snapshot => {fc_path}")

        fc_state = torch.load(fc_path, map_location=device)
        if "fc_state_dict" in fc_state:
            fc_state = fc_state["fc_state_dict"]
        if "weight" not in fc_state and "fc.weight" in fc_state:
            fc_state["weight"] = fc_state.pop("fc.weight")
        if "bias" not in fc_state and "fc.bias" in fc_state:
            fc_state["bias"] = fc_state.pop("fc.bias")

        w = fc_state["weight"]  # shape [C, D]
        b = fc_state["bias"]    # shape [C]

        noise_vec = F.normalize(torch.randn_like(w), dim=1)
        noise_scaled = noise_vec * (ref_norm * perturbation_distance)

        w_perturbed = w + noise_scaled

        snapshot_weights.append(w_perturbed)
        snapshot_biases.append(b)

    stacked_weight = torch.cat(snapshot_weights, dim=0) 
    stacked_bias   = torch.cat(snapshot_biases, dim=0)  

    M = end_epoch - start_epoch + 1
    C = snapshot_weights[0].shape[0] 
    D = snapshot_weights[0].shape[1]  

    def build_lin():
        big_fc = nn.Linear(D, M * C, bias=True)
        with torch.no_grad():
            big_fc.weight.copy_(stacked_weight)
            big_fc.bias.copy_(stacked_bias)
        big_fc.to(device)
        return big_fc

    if perturbed_fc is None:
        perturbed_fc = build_lin()
    else:
        if perturbed_fc.weight.shape != (M * C, D):
            perturbed_fc = build_lin()
        else:
            with torch.no_grad():
                perturbed_fc.weight.copy_(stacked_weight)
                perturbed_fc.bias.copy_(stacked_bias)

    print(f"[DEBUG] => Built big_fc with shape: {perturbed_fc.weight.shape}, bias: {perturbed_fc.bias.shape}")

    output_list = []
    for chunk in latents.split(batch_size):
        out = perturbed_fc(chunk.to(device))
        output_list.append(out.cpu())
    weiper_logits = torch.cat(output_list, dim=0)

    return weiper_logits, perturbed_fc, M, C


# def calculate_density(
#     latents, min_train, max_train, n_bins=100, eps=1e-8, device="cpu", verbose=False
# ):

#     density = []

#     for l in tqdm(latents.split(100, dim=0), disable=not verbose):
#         density_t = batch_histogram(
#             l.T,
#             n_bins,
#             bins_min_max=(min_train, max_train),
#             disable_tqdm=True,
#             probability=True,
#             device=device,
#         )[0]

#         density.append(density_t.cpu())
#     density = torch.cat(density, dim=-1)
#     return density


# def kldiv(
#     p: torch.Tensor,
#     q: torch.Tensor,
#     eps: float = 1e-8,
#     kernel: torch.nn.Module = None,
#     symmetric: bool = True,
#     device: str = "cpu",
# ) -> torch.Tensor:
#     """Calculate the Kullback-Leibler divergence after smoothing and normalizing the densities.
#     Mapping of shapes: (n_bins, n_samples_0),(n_bins, n_samples_1) -> (n_samples_0,n_samples_1)

#     Args:
#         p (torch.Tensor): p density tensor.
#         q (torch.Tensor): q density tensor.
#         eps (float, optional): Epsilon added to q to prevent zero entries. Defaults to 1e-8.
#         kernel (torch.nn.Module, optional): Smoothing kernel. Defaults to None.
#         symmetric (bool, optional): Calculate KLD(p,q) + KLD(q,p) if True. Defaults to True.
#         device (str, optional): Cuda device or CPU. Defaults to "cpu".

#     Returns:
#         torch.Tensor: uncertainty
#     """
#     kernel.to(device)
#     p = p.to(device)
#     if kernel is not None:
#         q_ = kernel(q.to(device))
#     q_ += eps
#     p_ = (p + eps).clone()
#     p_ /= p_.sum(dim=0)
#     q_ /= q_.sum(dim=0)
#     if symmetric:
#         q_p = q_.log()[:, None] - p_.log()[..., None]
#         return -(p_[..., None] * q_p).sum(dim=0) - (q_[:, None] * (-q_p)).sum(dim=0)
#     else:
#         return -(p_[..., None] * (q_.log()[:, None] - p_.log()[..., None])).sum(dim=0)


# def calculate_uncertainty(
#     density: torch.Tensor,
#     density_train_mean: torch.Tensor,
#     kernel: torch.nn.Module,
#     eps: float = 1e-8,
#     symmetric: bool = True,
#     device: str = "cpu",
# ) -> torch.Tensor:
#     """Calculate the score (uncertainties) for given densities and the training set mean density.

#     Args:
#         density (torch.Tensor): Given density tensor.
#         density_train_mean (torch.Tensor): Density mean of the training set.
#         kernel (torch.nn.Module): Smoothing kernel.
#         eps (float, optional): Epsilon added to q to prevent zero entries. Defaults to 1e-8.
#         symmetric (bool, optional): Calculate KLD(p,q) + KLD(q,p) if True. Defaults to True.
#         device (str, optional): Cuda device or CPU. Defaults to "cpu".

#     Returns:
#         torch.Tensor: uncertainty
#     """
#     uncertainty = []
#     for dens_t in tqdm(density.split(100, dim=-1), disable=True):
#         s = 100
#         if dens_t.shape[-1] < 100:
#             s = dens_t.shape[-1]
#             dens_t = torch.cat(
#                 [dens_t, torch.ones(dens_t.shape[0], 100 - s)], dim=-1
#             ).clone()

#         uncertainty.append(
#             kldiv(
#                 density_train_mean,
#                 dens_t,
#                 kernel=kernel,
#                 eps=eps,
#                 symmetric=symmetric,
#                 device=device,
#             )[:, :s]
#         )
#     return torch.cat(uncertainty, dim=-1)[0]


@torch.no_grad()
def calculate_WeiPerKLDiv_score(
    model: nn.Module,
    latents: torch.Tensor,
    n_bins: int = 100,
    perturbation_distance: float = 2.1,  
    n_repeats: int = 100,                
    smoothing: int = 20,
    smoothing_perturbed: int = 20,
    epsilon: float = 0.01,
    epsilon_noise: float = 0.025,
    lambda_1: float = 1,
    lambda_2: float = 1,
    symmetric: bool = True,
    device: str = "cpu",
    verbose: bool = True,
    ablation_noise_only: bool = False,
    train_min_max: tuple = None,
    train_densities: Iterable[torch.Tensor] = None,
    perturbed_fc: nn.Module = None,
    return_msp_only: bool = True,
    fc_dir: str = "./checkpoints/fc_layers",
    start_epoch: int = 101,
    end_epoch: int = 200,
    **params,
) -> Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply the WeiPerKLDiv score to the given latents.

    Args:
        model (torch.nn.Module): Instance of the neural network.
        latents (Iterable[torch.Tensor]): Train latents, test latents, ood latents.
        densities (torch.Tensor, optional): If the densities are already calculated, they can be referenced here. Defaults to None.
        n_bins (int, optional): Number of bins. Defaults to 100.
        perturbation_distance (float, optional): (Relative) perturbation distance (delta in the paper). Defaults to 2.1.
        n_repeats (int, optional): The number of repeats (r in the paper). Defaults to 100.
        smoothing (int, optional): Smoothing size (s_1 in the paper) for the density. Defaults to 20.
        smoothing_perturbed (int, optional): Smoothing size (s_2 in the paper) for the perturbed density. Defaults to 20.
        epsilon (float, optional): Epsilon added to q to prevent zero entries. Defaults to 0.025.
        lambda_1 (float, optional): Lambda_1 (as in the paper). Defaults to 1.
        lambda_2 (float, optional): Lambda_2 (as in the paper). Defaults to 1.
        symmetric (bool, optional): Calculate KLD(p,q) + KLD(q,p) if True. Defaults to True.
        device (str, optional): Cuda device or CPU. Defaults to "cpu".
        verbose (bool, optional): Whether to show progress bars. Defaults to False.
        ablation_noise_only (bool, optional): Use weight independent random projections. Defaults to False.
        train_min_max (tuple, optional): If the min and max of the train set are already calculated, they can be referenced here. Defaults to None.
        train_densities (Iterable[torch.Tensor], optional): If the densities of the train set are already calculated,
        they can be referenced here. Defaults to None.

        Snapshot-based WeiPer approach:
      - Loads snapshots from epochs [start_epoch, end_epoch],
      - Stacks them using the scaling in calculate_weiper_space,
      - Computes logits on the given latents,
      - Reshapes the logits to (B, M, C), averages over snapshots, applies softmax,
      - Returns the Maximum Softmax Probability (MSP) for each sample


    Returns:
        Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: uncertainties, densities, latents_weiper, train_densities, W_tilde
    """
    model.eval()

    if verbose:
        print("Calculate perturbed logits...")

    latents_weiper, W_tilde, M, C = calculate_weiper_space(
        model,
        latents,
        perturbed_fc=perturbed_fc,
        device=device,
        fc_dir=fc_dir,
        start_epoch=start_epoch,
        end_epoch=end_epoch,
        perturbation_distance=perturbation_distance,
        batch_size=200,
    )

    if verbose:
        print("Evaluate density...")
    # if train_min_max is None:
    #     # calculate min max estimates
    #     def minmax_plus_avg_gap(min_, max_):
    #         gap = (max_ - min_) / latents.shape[0]
    #         return min_ - gap, max_ + gap

    #     min_train, max_train = minmax_plus_avg_gap(min_train, max_train)
    #     min_weiper_train, max_weiper_train = minmax_plus_avg_gap(
    #         min_weiper_train, max_weiper_train
    #     )
    #     train_min_max = (min_train, max_train), (min_weiper_train, max_weiper_train)
    # else:
    #     (min_train, max_train), (min_weiper_train, max_weiper_train) = train_min_max
    # densities = calculate_density(
    #     latents,
    #     min_train,
    #     max_train,
    #     n_bins=n_bins,
    #     device=device,
    # )
    # densities_weiper = calculate_density(
    #     latents_weiper,
    #     min_weiper_train,
    #     max_weiper_train,
    #     n_bins=n_bins,
    #     device=device,
    # )
    # if train_densities is None:
    #     densities_train_mean = densities.mean(dim=-1)[:, None]
    #     densities_weiper_train_mean = densities_weiper.mean(dim=-1)[:, None]
    #     return (
    #         densities_train_mean,
    #         densities_weiper_train_mean,
    #         ((min_train, max_train), (min_weiper_train, max_weiper_train)),
    #         W_tilde,
    #     )
    # else:
    #     densities_train_mean, densities_weiper_train_mean = train_densities

    def calculate_weiper_pred(logits: torch.Tensor) -> torch.Tensor:
        B, total_dim = logits.shape
        if total_dim != M * C:
            raise ValueError(f"Dim mismatch: got {total_dim}, expected {M * C}")
        logits_3d = logits.view(B, M, C)
        prob = torch.softmax(logits_3d, dim=2)
        max_prob = prob.max(dim=2)[0]  
        avg_max_prob = max_prob.mean(dim=1) 
        return avg_max_prob


    msp_weiper = calculate_weiper_pred(latents_weiper)

    if return_msp_only:
        if verbose:
            print("[DEBUG] => Returning MSP only from snapshot-based aggregator with scaling.")
        return msp_weiper
    
    # else:
    #     kernel = UniformKernel(100, smoothing).to(device)
    #     kernel_noise = UniformKernel(100, smoothing_perturbed).to(device)
    #     uncertainties = calculate_uncertainty(
    #         densities,
    #         densities_train_mean,
    #         kernel,
    #         eps=epsilon,
    #         device=device,
    #         symmetric=symmetric,
    #     )
    #     uncertainties_weiper = calculate_uncertainty(
    #         densities_weiper,
    #         densities_weiper_train_mean,
    #         kernel_noise,
    #         eps=epsilon_noise,
    #         device=device,
    #         symmetric=symmetric,
    #     )

    #     uncertainties = -(
    #         uncertainties + lambda_1 * uncertainties_weiper - lambda_2 * msp_weiper
    #     )

    #     return uncertainties
