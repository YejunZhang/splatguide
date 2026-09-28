import torch

from seva.sampling import DiscreteDenoiser, append_dims


def pose_distance(c2w: torch.Tensor) -> torch.Tensor:
    """Pairwise camera distance [T, T]: sqrt(||t_rel||^2 + 2 (1 - tr(R_rel) / 3))."""
    rel = torch.einsum("nij,mjk->nmik", torch.linalg.inv(c2w), c2w)
    rot = torch.sqrt(2 * (1 - torch.einsum("...ii", rel[..., :3, :3]).clip(max=3.0) / 3))
    return torch.sqrt(torch.norm(rel[..., :3, 3], dim=-1) ** 2 + rot**2)


def seva_weights(mask: torch.Tensor, c2w: torch.Tensor, max_weight: float = 5.0, eps: float = 1e-6) -> torch.Tensor:
    """SEVA per-frame loss weights [T]: 0 for references, otherwise the pose distance to the
    nearest reference, scaled so that the farthest target gets max_weight."""
    dist = torch.where(mask, 0.0, pose_distance(c2w)[:, mask].min(dim=-1).values)
    return dist / (dist.max() + eps) * max_weight


def diffusion_loss(
    network, denoiser: DiscreteDenoiser, cond: dict, latents: torch.Tensor, mask: torch.Tensor, c2w: torch.Tensor
) -> torch.Tensor:
    """L2 between the denoised and the clean latents [T, C, H, W], weighted per frame by
    `seva_weights` (as in SEVA / GeoNVS training); one noise level per sample."""
    sigma = denoiser.sigmas[torch.randint(0, len(denoiser.sigmas), (1,))].expand(len(latents))
    noised = latents + torch.randn_like(latents) * append_dims(sigma, latents.ndim)
    denoised = denoiser(network, noised, sigma, cond, num_frames=len(latents))
    w = append_dims(seva_weights(mask, c2w), latents.ndim)
    return torch.mean(w * (denoised.float() - latents.float()) ** 2)
