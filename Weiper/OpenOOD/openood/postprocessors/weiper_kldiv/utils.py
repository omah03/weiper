import torch
import numpy as np
from tqdm import tqdm
from collections.abc import Iterable
import torch.nn.functional as F
from typing import Union


@torch.jit.script
def linspace(start: torch.Tensor, stop: torch.Tensor, num: int):
    """
    Creates a tensor of shape [num, *start.shape] whose values are evenly spaced from start to end, inclusive.
    Replicates but the multi-dimensional behaviour of numpy.linspace in PyTorch.
    Source: https://github.com/pytorch/pytorch/issues/61292
    """
    # create a tensor of 'num' steps from 0 to 1
    steps = torch.arange(num, dtype=torch.float32, device=start.device) / (num - 1)

    for i in range(start.ndim):
        steps = steps.unsqueeze(-1)

    # the output starts at 'start' and increments until 'stop' in each dimension
    out = start[None] + steps * (stop - start)[None]
    return out


@torch.no_grad()
def batch_histogram(
    samples: torch.Tensor,
    n_bins: int = 100,
    batch_size: int = 1000,
    device: str = "cpu",
    disable_tqdm: bool = False,
    bins_min_max: tuple = None,
    probability: bool = False,
) -> Union[torch.Tensor, torch.Tensor]:
    """Generates a histogram (density, bins) for multiple dimensions for batch tensor shaped [n_samples, n_dims].

    Args:
        samples (torch.Tensor): Input samples.
        n_bins (int, optional): Number of bins. Defaults to 100.
        batch_size (int, optional): Batch size. Defaults to 200.
        device (str, optional): Cuda device or CPU. Defaults to "cpu".
        disable_tqdm (bool, optional): Whether to show progress bars (if True: disabled). Defaults to False.
        bins_min_max (tuple, optional): Set min and max for the bins instead of calculating it from the samples. Defaults to None.
        probability (bool, optional): If True, normalizes the densities to sum to one (outputs probabilities instead of densities). Defaults to False.

    Returns:
        Union[torch.Tensor, torch.Tensor]: density, bins (shaped [n_bins, n_dims], [n_bins + 1, n_dims])
    """
    eps = 1e-8
    samples = samples.to(device)
    if bins_min_max is None:
        if not isinstance(samples, torch.Tensor):
            min = next(iter(samples)).min(dim=0)[0]
            max = next(iter(samples)).max(dim=0)[0]
            for sample in samples:
                min = torch.where(sample.min(dim=0)[0] < min, sample.min(dim=0)[0], min)
                max = torch.where(sample.max(dim=0)[0] > max, sample.max(dim=0)[0], max)
        else:
            min = samples.min(dim=0)[0]
            max = samples.max(dim=0)[0]
        bins = linspace(min, max + eps, num=n_bins + 1)
    else:
        min = bins_min_max[0]
        max = bins_min_max[1]
        bins = linspace(min, max + eps, num=n_bins + 1)[:, None].repeat(
            1, samples.shape[1]
        )
        bins = torch.cat(
            [bins, torch.ones(1, bins.shape[-1]).to(bins.device) * np.inf], dim=0
        )
        bins = torch.cat(
            [-torch.ones(1, bins.shape[-1]).to(bins.device) * np.inf, bins], dim=0
        )
    bin_length = bins[2:3] - bins[1:2] + 1e-12

    bins = bins.to(device)
    if not isinstance(samples, torch.Tensor) or samples.shape[0] > batch_size:
        if isinstance(samples, torch.Tensor):
            samples = samples.split(batch_size)
        densities = torch.zeros(n_bins + 2, samples[0].shape[1])
        for samples in tqdm(samples, disable=disable_tqdm):
            samples = samples.to(device)

            density = torch.logical_and(
                (samples[None] < bins[1:, None]),
                (samples[None] >= bins[:-1, None]),
            )
            density = density.to(torch.float32).mean(dim=1)
            densities += density.cpu()
        densities /= len(samples)
        density = densities
    else:
        density = torch.logical_and(
            (samples[None] < bins[1:, None]), (samples[None] >= bins[:-1, None])
        )
        density = density.to(torch.float32).mean(dim=1)
    if not probability:
        density /= bin_length.to(density.device)
    else:
        density = density[1:-1] / density.sum(dim=0)[None]
    return density, bins


class UniformKernel(torch.nn.Module):
    """Uniform kernel for smoothing the density.

    Args:
        dim (int): Number of (latent/WeiPer) dimensions.
        kernel_size (int, optional): Kernel size. Defaults to 10.
    """

    def __init__(
        self,
        dim,
        kernel_size=10,
    ):
        super().__init__()
        self.kernel = torch.nn.Conv1d(
            dim, dim, kernel_size, padding="same", padding_mode="replicate"
        )
        self.kernel.weight.data = torch.zeros_like(self.kernel.weight.data)
        self.kernel.weight.data[range(dim), range(dim)] = (
            torch.ones(1, dim, kernel_size) / kernel_size
        )
        self.kernel.bias.data = torch.zeros_like(self.kernel.bias)
        self.kernel.bias.requires_grad = False
        self.kernel.weight.requires_grad = False

    @torch.no_grad()
    def forward(self, densities, bins=None):
        # smoothing
        smoothed_densities = self.kernel(densities.T).T
        # correct densities to sum to 1.
        if bins is not None:
            smoothed_densities /= (smoothed_densities * (bins[1] - bins[0])).sum(dim=0)[
                None
            ]
        return smoothed_densities


@torch.no_grad()
def calculate_weiper_space(
    model: torch.nn.Module,
    latents: torch.Tensor,
    perturbed_fc: torch.nn.Module = None,
    device: str = "cpu",
    perturbation_distance: float = 2.1,
    n_repeats: int = 50,
    noise_proportional: bool = True,
    constant_length: bool = True,
    batch_size: int = 256,
    ablation_noise_only: bool = False,
) -> Union[torch.Tensor, torch.Tensor]:
    """Creates the perturbed fully connected layer (curly H in the paper)
    and calculates the perturbed logits from the penultimate latents.

    Args:
        model (torch.nn.Module): Instance of the neural network.
        latents (torch.Tensor): The penultimate latents.
        perturbed_fc (torch.nn.Module, optional): Precalculates perturbed fully connected layer. Defaults to None.
        device (str, optional): Cuda device or CPU. Defaults to "cpu".
        perturbation_distance (float, optional): (Relative) perturbation distance (delta in the paper). Defaults to 2.1.
        n_repeats (int, optional): The number of repeats (r in the paper). Defaults to 50.
        noise_proportional (bool, optional): Force a constant ratio between noise and class projections controlled by
        perturbation_distance. Defaults to True.
        constant_length (bool, optional): Normalize the noise to have length perturbation_distance. Defaults to True.
        batch_size (int, optional): Batch size. Defaults to 200.
        apply_softmax (bool, optional): Whether to apply softmax to the output. Defaults to False.
        ablation_noise_only (bool, optional): Use weight independent random projections. Defaults to False.

    Returns:
        Union[torch.Tensor, torch.Tensor]: noise_logits, perturbed_fc
    """
    if perturbed_fc is not None:
        perturbed_fc = perturbed_fc.to(device)

    random_perturbs = [
        perturbation_distance
        * (
            F.normalize(torch.randn_like(model.fc.weight.data), dim=1)
            if constant_length
            else torch.randn_like(model.fc.weight.data)
        )
        for _ in range(n_repeats)
    ]
    if noise_proportional:
        random_perturbs = [
            pert * model.fc.weight.norm(dim=1)[:, None] for pert in random_perturbs
        ]

    def build_lin():
        weight = model.fc.weight.data
        bias = model.fc.bias.data
        if ablation_noise_only:
            weight = 0 * weight
            bias = 0 * bias
        weight_ = torch.cat([weight + pert for pert in random_perturbs], dim=0)
        bias_ = torch.cat([bias for _ in range(n_repeats)], dim=0)

        perturbed_fc = torch.nn.Linear(
            latents[0].shape[-1], n_repeats * weight.shape[0]
        )
        perturbed_fc.weight.data = weight_
        perturbed_fc.bias.data = bias_
        perturbed_fc.to(device)
        return perturbed_fc

    if perturbed_fc is None:
        perturbed_fc = build_lin()

    weiper_logits = torch.cat(
        [perturbed_fc(x.to(device)).cpu() for x in latents.split(batch_size)], dim=0
    )

    return weiper_logits, perturbed_fc


def calculate_density(
    latents, min_train, max_train, n_bins=100, eps=1e-8, device="cpu", verbose=False
):

    density = []

    for l in tqdm(latents.split(100, dim=0), disable=not verbose):
        density_t = batch_histogram(
            l.T,
            n_bins,
            bins_min_max=(min_train, max_train),
            disable_tqdm=True,
            probability=True,
            device=device,
        )[0]

        density.append(density_t.cpu())
    density = torch.cat(density, dim=-1)
    return density


def kldiv(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
    kernel: torch.nn.Module = None,
    symmetric: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Calculate the Kullback-Leibler divergence after smoothing and normalizing the densities.
    Mapping of shapes: (n_bins, n_samples_0),(n_bins, n_samples_1) -> (n_samples_0,n_samples_1)

    Args:
        p (torch.Tensor): p density tensor.
        q (torch.Tensor): q density tensor.
        eps (float, optional): Epsilon added to q to prevent zero entries. Defaults to 1e-8.
        kernel (torch.nn.Module, optional): Smoothing kernel. Defaults to None.
        symmetric (bool, optional): Calculate KLD(p,q) + KLD(q,p) if True. Defaults to True.
        device (str, optional): Cuda device or CPU. Defaults to "cpu".

    Returns:
        torch.Tensor: uncertainty
    """
    kernel.to(device)
    p = p.to(device)
    if kernel is not None:
        q_ = kernel(q.to(device))
    q_ += eps
    p_ = (p + eps).clone()
    p_ /= p_.sum(dim=0)
    q_ /= q_.sum(dim=0)
    if symmetric:
        q_p = q_.log()[:, None] - p_.log()[..., None]
        return -(p_[..., None] * q_p).sum(dim=0) - (q_[:, None] * (-q_p)).sum(dim=0)
    else:
        return -(p_[..., None] * (q_.log()[:, None] - p_.log()[..., None])).sum(dim=0)


def calculate_uncertainty(
    density: torch.Tensor,
    density_train_mean: torch.Tensor,
    kernel: torch.nn.Module,
    eps: float = 1e-8,
    symmetric: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Calculate the score (uncertainties) for given densities and the training set mean density.

    Args:
        density (torch.Tensor): Given density tensor.
        density_train_mean (torch.Tensor): Density mean of the training set.
        kernel (torch.nn.Module): Smoothing kernel.
        eps (float, optional): Epsilon added to q to prevent zero entries. Defaults to 1e-8.
        symmetric (bool, optional): Calculate KLD(p,q) + KLD(q,p) if True. Defaults to True.
        device (str, optional): Cuda device or CPU. Defaults to "cpu".

    Returns:
        torch.Tensor: uncertainty
    """
    uncertainty = []
    for dens_t in tqdm(density.split(100, dim=-1), disable=True):
        s = 100
        if dens_t.shape[-1] < 100:
            s = dens_t.shape[-1]
            dens_t = torch.cat(
                [dens_t, torch.ones(dens_t.shape[0], 100 - s)], dim=-1
            ).clone()

        uncertainty.append(
            kldiv(
                density_train_mean,
                dens_t,
                kernel=kernel,
                eps=eps,
                symmetric=symmetric,
                device=device,
            )[:, :s]
        )
    return torch.cat(uncertainty, dim=-1)[0]


@torch.no_grad()
def calculate_WeiPerKLDiv_score(
    model: torch.nn.Module,
    latents: Iterable[torch.Tensor],
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
    verbose: bool = False,
    ablation_noise_only: bool = False,
    train_min_max: tuple = None,
    train_densities: Iterable[torch.Tensor] = None,
    perturbed_fc: torch.nn.Module = None,
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

    Returns:
        Union[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: uncertainties, densities, latents_weiper, train_densities, W_tilde
    """
    model.eval()

    if verbose:
        print("Calculate perturbed logits...")
    latents_weiper, W_tilde = calculate_weiper_space(
        model,
        latents,
        perturbed_fc=perturbed_fc,
        device="cpu",
        perturbation_distance=perturbation_distance,
        n_repeats=n_repeats,
        noise_proportional=True,
        constant_length=True,
        batch_size=200,
        ablation_noise_only=ablation_noise_only,
    )

    if verbose:
        print("Evaluate density...")
    if train_min_max is None:
        # calculate min max estimates
        def minmax_plus_avg_gap(min_, max_):
            gap = (max_ - min_) / latents.shape[0]
            return min_ - gap, max_ + gap

        min_train, max_train = minmax_plus_avg_gap(min_train, max_train)
        min_weiper_train, max_weiper_train = minmax_plus_avg_gap(
            min_weiper_train, max_weiper_train
        )
        train_min_max = (min_train, max_train), (min_weiper_train, max_weiper_train)
    else:
        (min_train, max_train), (min_weiper_train, max_weiper_train) = train_min_max
    densities = calculate_density(
        latents,
        min_train,
        max_train,
        n_bins=n_bins,
        device=device,
    )
    densities_weiper = calculate_density(
        latents_weiper,
        min_weiper_train,
        max_weiper_train,
        n_bins=n_bins,
        device=device,
    )
    if train_densities is None:
        densities_train_mean = densities.mean(dim=-1)[:, None]
        densities_weiper_train_mean = densities_weiper.mean(dim=-1)[:, None]
        return (
            densities_train_mean,
            densities_weiper_train_mean,
            ((min_train, max_train), (min_weiper_train, max_weiper_train)),
            W_tilde,
        )
    else:
        densities_train_mean, densities_weiper_train_mean = train_densities

    def calculate_weiper_pred(logits):
        n_classes = int(logits.shape[-1] / n_repeats)
        logits = logits.to(device)
        weiper_pred = []
        # calculate WeiPer predictions for each perturbed space
        for i in range(n_repeats):
            weiper_pred.append(
                logits[:, i * n_classes : (i + 1) * n_classes]
                .softmax(dim=-1)
                .max(dim=-1)[0][:, None]
            )
        return torch.cat(weiper_pred, dim=-1).mean(dim=-1)

    msp_weiper = calculate_weiper_pred(latents_weiper)

    kernel = UniformKernel(100, smoothing).to(device)
    kernel_noise = UniformKernel(100, smoothing_perturbed).to(device)
    uncertainties = calculate_uncertainty(
        densities,
        densities_train_mean,
        kernel,
        eps=epsilon,
        device=device,
        symmetric=symmetric,
    )
    uncertainties_weiper = calculate_uncertainty(
        densities_weiper,
        densities_weiper_train_mean,
        kernel_noise,
        eps=epsilon_noise,
        device=device,
        symmetric=symmetric,
    )

    uncertainties = -(
        uncertainties + lambda_1 * uncertainties_weiper - lambda_2 * msp_weiper
    )

    return uncertainties
