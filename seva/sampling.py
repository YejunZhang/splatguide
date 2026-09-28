import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from seva.geometry import get_camera_dist


def append_dims(x: torch.Tensor, target_dims: int) -> torch.Tensor:
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(
            f"input has {x.ndim} dims but target_dims is {target_dims}, which is less"
        )
    return x[(...,) + (None,) * dims_to_append]


def append_zero(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, x.new_zeros([1])])


def to_d(x: torch.Tensor, sigma: torch.Tensor, denoised: torch.Tensor) -> torch.Tensor:
    return (x - denoised) / append_dims(sigma, x.ndim)


def make_betas(
    num_timesteps: int, linear_start: float = 1e-4, linear_end: float = 2e-2
) -> np.ndarray:
    betas = (
        torch.linspace(
            linear_start**0.5, linear_end**0.5, num_timesteps, dtype=torch.float64
        )
        ** 2
    )
    return betas.numpy()


def generate_roughly_equally_spaced_steps(
    num_substeps: int, max_step: int
) -> np.ndarray:
    return np.linspace(max_step - 1, 0, num_substeps, endpoint=False).astype(int)[::-1]


class DDPMDiscretization(object):
    def __init__(
        self,
        linear_start: float = 5e-06,
        linear_end: float = 0.012,
        num_timesteps: int = 1000,
        log_snr_shift: float | None = 2.4,
    ):
        self.num_timesteps = num_timesteps

        betas = make_betas(
            num_timesteps,
            linear_start=linear_start,
            linear_end=linear_end,
        )
        self.log_snr_shift = log_snr_shift

        alphas = 1.0 - betas  # first alpha here is on data side
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

    def get_sigmas(self, n: int, device: str | torch.device = "cpu") -> torch.Tensor:
        if n < self.num_timesteps:
            timesteps = generate_roughly_equally_spaced_steps(n, self.num_timesteps)
            alphas_cumprod = self.alphas_cumprod[timesteps]
        elif n == self.num_timesteps:
            alphas_cumprod = self.alphas_cumprod
        else:
            raise ValueError(f"Expected n <= {self.num_timesteps}, but got n = {n}.")

        sigmas = ((1 - alphas_cumprod) / alphas_cumprod) ** 0.5
        if self.log_snr_shift is not None:
            sigmas = sigmas * np.exp(self.log_snr_shift)
        return torch.flip(
            torch.tensor(sigmas, dtype=torch.float32, device=device), (0,)
        )

    def __call__(
        self,
        n: int,
        do_append_zero: bool = True,
        flip: bool = False,
        device: str | torch.device = "cpu",
    ) -> torch.Tensor:
        sigmas = self.get_sigmas(n, device=device)
        sigmas = append_zero(sigmas) if do_append_zero else sigmas
        return sigmas if not flip else torch.flip(sigmas, (0,))


class DiscreteDenoiser(object):
    """Eps-parameterised denoiser on the discrete DDPM sigma grid (used for training and sampling).

    Inputs are frame-flattened [N, C, H, W]; `cond["replace"]` = [clean latent, mask] puts the
    clean latents of the reference frames back into the input.
    """

    def __init__(
        self,
        discretization: DDPMDiscretization,
        num_idx: int = 1000,
        device: str | torch.device = "cpu",
    ):
        self.sigmas = discretization(num_idx, do_append_zero=False, flip=True, device=device)

    def __call__(
        self,
        network: nn.Module,
        input: torch.Tensor,
        sigma: torch.Tensor,
        cond: dict,
        **additional_model_inputs,
    ) -> torch.Tensor:
        idx = (sigma - self.sigmas[:, None]).abs().argmin(dim=0).view(sigma.shape)
        sigma = append_dims(self.sigmas[idx], input.ndim)
        x, mask = cond.pop("replace").split((input.shape[1], 1), dim=1)
        input = input * (1 - mask) + x * mask
        c_in = 1 / (sigma**2 + 1.0) ** 0.5
        return network(input * c_in, idx, cond, **additional_model_inputs) * -sigma + input


class MultiviewCFG(object):
    """Classifier-free guidance; frames that coincide with a reference camera use `cfg_min`."""

    def __init__(self, cfg_min: float = 1.0):
        self.min_scale = cfg_min

    def prepare_inputs(
        self, x: torch.Tensor, s: torch.Tensor, c: dict, uc: dict
    ) -> tuple[torch.Tensor, torch.Tensor, dict]:
        return torch.cat([x] * 2), torch.cat([s] * 2), {k: torch.cat((uc[k], c[k]), 0) for k in c}

    def __call__(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        scale: float,
        c2w: torch.Tensor,
        K: torch.Tensor,
        input_frame_mask: torch.Tensor,
    ) -> torch.Tensor:
        x_u, x_c = x.chunk(2)
        c2w_input = c2w[input_frame_mask]
        rotation_diff = get_camera_dist(c2w, c2w_input, mode="rotation").min(-1).values
        translation_diff = get_camera_dist(c2w, c2w_input, mode="translation").min(-1).values
        K_diff = ((K[:, None] - K[input_frame_mask][None]).flatten(-2) == 0).all(-1).any(-1)
        close_frame = (rotation_diff < 10.0) & (translation_diff < 1e-5) & K_diff
        scale = torch.where(close_frame, self.min_scale, scale)
        return x_u + append_dims(scale, x.ndim) * (x_c - x_u)


class EulerEDMSampler(object):
    """SEVA's Euler sampler without churn. The (vanishing) noise term of each step is kept: it
    draws from the RNG, and sigma_hat = sigma + 1e-6 makes it non-zero."""

    def __init__(
        self,
        discretization: DDPMDiscretization,
        guider: MultiviewCFG,
        num_steps: int,
        device: str | torch.device = "cuda",
    ):
        self.num_steps = num_steps
        self.discretization = discretization
        self.guider = guider
        self.device = device

    def __call__(
        self, denoiser, x: torch.Tensor, scale: float, cond: dict, uc: dict, **guider_kwargs
    ) -> torch.Tensor:
        sigmas = self.discretization(self.num_steps, device=self.device)
        x *= torch.sqrt(1.0 + sigmas[0] ** 2.0)
        s_in = x.new_ones([x.shape[0]])
        for i in tqdm(range(len(sigmas) - 1), desc="Sampling", leave=False):
            sigma = s_in * sigmas[i]
            sigma_hat = sigma + 1e-6
            eps = torch.randn_like(x)
            x = x + eps * append_dims(sigma_hat**2 - sigma**2, x.ndim) ** 0.5
            denoised = denoiser(*self.guider.prepare_inputs(x, sigma_hat, cond, uc))
            denoised = self.guider(denoised, sigma_hat, scale, **guider_kwargs)
            d = to_d(x, sigma_hat, denoised)
            dt = append_dims(s_in * sigmas[i + 1] - sigma_hat, x.ndim)
            x = x + dt * d
        return x
