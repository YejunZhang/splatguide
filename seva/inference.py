import torch
from einops import repeat

from seva.sampling import DDPMDiscretization, EulerEDMSampler, MultiviewCFG


def create_sampler(num_steps: int = 50, cfg_min: float = 1.2, device: str | torch.device = "cuda") -> EulerEDMSampler:
    return EulerEDMSampler(DDPMDiscretization(), MultiviewCFG(cfg_min), num_steps, device)


def do_sample(model, ae, denoiser, sampler: EulerEDMSampler, value_dict: dict, cfg: float) -> torch.Tensor:
    """Generate all frames of one group (SEVA `do_sample` + SplatGuide conditions).

    value_dict (all on the GPU):
        cond_frames        [T, 3, H, W] in [-1, 1]; only reference frames are used
        cond_frames_mask   [T] bool, True for reference frames
        plucker_coordinate [T, 6, H/8, W/8]
        c2w, K             normalised cameras (for the multiview CFG rule)
        clip_features      [1024] CLIP embedding (mean over references)
        rendered_latents   [T, 4, H/8, W/8] VAE latents of the 3DGS renders, or None
        tokens             [1, V, 5, D] reconstruction tokens of the V references, or None
    Returns decoded frames [T, 3, H, W] in [-1, 1] (CPU).
    """
    imgs = value_dict["cond_frames"]
    input_masks = value_dict["cond_frames_mask"]
    pluckers = value_dict["plucker_coordinate"]
    T = len(imgs)

    with torch.inference_mode(), torch.autocast("cuda"):
        latents = torch.nn.functional.pad(ae.encode(imgs[input_masks], 1), (0, 0, 0, 0, 0, 1), value=1.0)
        c_crossattn = repeat(value_dict["clip_features"], "d -> n 1 d", n=T)
        c_replace = latents.new_zeros(T, *latents.shape[1:])
        c_replace[input_masks] = latents

        mask = repeat(input_masks, "n -> n 1 h w", h=pluckers.shape[2], w=pluckers.shape[3])
        c_concat = [mask, pluckers]
        uc_concat = [pluckers.new_zeros(T, 1, *pluckers.shape[-2:]), pluckers]
        # Rendered latents are kept in the unconditional branch as well.
        rendered_latents = value_dict.get("rendered_latents")
        if rendered_latents is not None:
            c_concat.append(rendered_latents)
            uc_concat.append(rendered_latents)

        c = {
            "crossattn": c_crossattn,
            "replace": c_replace,
            "concat": torch.cat(c_concat, 1),
            "dense_vector": pluckers,
        }
        uc = {
            "crossattn": torch.zeros_like(c_crossattn),
            "replace": torch.zeros_like(c_replace),
            "concat": torch.cat(uc_concat, 1),
            "dense_vector": pluckers,
        }
        tokens = value_dict.get("tokens")
        if tokens is not None:
            c["tokens"] = uc["tokens"] = tokens.flatten(1, 2)

        randn = torch.randn(T, 4, *pluckers.shape[-2:]).to("cuda")
        samples_z = sampler(
            lambda input, sigma, c: denoiser(model, input, sigma, c, num_frames=T),
            randn,
            scale=cfg,
            cond=c,
            uc=uc,
            c2w=value_dict["c2w"],
            K=value_dict["K"],
            input_frame_mask=input_masks,
        )
        samples = ae.decode(samples_z, 1)
    return samples.detach().cpu()
