"""Frozen scene encoder, run online in training and evaluation: WorldMirror reconstruction,
3DGS renders, SEVA VAE latents and CLIP features.

Stage 1 predicts cameras for all views (references + targets) without priors. Stage 2 re-runs the
references with the Stage-1 reference cameras as priors; its cameras, 3D Gaussians and camera/register
tokens are kept. The Stage-1 target poses are aligned to the Stage-2 frame and every view is rendered
from the Stage-2 Gaussians.
"""
import os
import sys

import torch
from gsplat.rendering import rasterization

from recon.scenes import OUT_SIZE, WM_SIZE
from seva.modules.autoencoder import AutoEncoder
from seva.modules.conditioner import CLIPConditioner

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "third_party", "HunyuanWorld-Mirror"))
from src.models.models.worldmirror import WorldMirror  # noqa: E402

WORLDMIRROR_REVISION = "5574b7b0d5ac9d80e8a92976222370a0d20a57a0"  # tencent/HunyuanWorld-Mirror weights
INTR_IDX = (slice(None), [0, 1, 0, 1], [0, 1, 2, 2])  # fx, fy, cx, cy of [N, 3, 3]


class SceneEncoder:
    """Not an nn.Module on purpose: it stays out of the diffusion model's state dict and optimizer."""

    def __init__(self, device: str | torch.device = "cuda"):
        self.device = device
        self.worldmirror = WorldMirror.from_pretrained(
            "tencent/HunyuanWorld-Mirror", revision=WORLDMIRROR_REVISION
        ).to(device).eval()
        self.ae = AutoEncoder(chunk_size=4).to(device).eval()
        self.clip = CLIPConditioner().to(device).eval()

    def run_worldmirror(self, views: dict, cond_flags: list[int], gaussians: bool):
        """Forward pass with only the heads we use (cameras, and Gaussians if asked); also returns
        the camera + 4 register tokens of the last aggregator layer, [S, 5, 2048]."""
        wm = self.worldmirror
        wm.enable_depth = wm.enable_pts = wm.enable_norm = False
        wm.enable_gs = gaussians
        captured = {}
        hook = wm.visual_geometry_transformer.register_forward_hook(
            lambda module, inputs, outputs: captured.update(tokens=outputs[0][-1])
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            preds = wm(views=views, cond_flags=cond_flags)
        hook.remove()
        tokens = captured["tokens"]
        return preds, tokens.reshape(views["img"].shape[1], -1, tokens.shape[-1])[:, :5].float()

    def render(self, splats: dict, c2w: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """Stage-2 Gaussians on a white background; [V, 3, OUT_SIZE, OUT_SIZE] in [-1, 1]."""
        colors, _, _ = rasterization(
            means=splats["means"][0], quats=splats["quats"][0], scales=splats["scales"][0],
            opacities=splats["opacities"][0], colors=splats["sh"][0], sh_degree=0,
            viewmats=torch.linalg.inv(c2w), Ks=K.repeat(len(c2w), 1, 1), width=OUT_SIZE, height=OUT_SIZE,
            render_mode="RGB+ED", backgrounds=torch.ones(len(c2w), 3, device=c2w.device),
        )
        return colors[..., :3].permute(0, 3, 1, 2).clamp(0, 1) * 2.0 - 1.0

    @torch.no_grad()
    def __call__(self, images_wm: torch.Tensor, images: torch.Tensor, n_ref: int, align_fn) -> dict:
        """images_wm [N, 3, WM_SIZE, WM_SIZE], images [N, 3, OUT_SIZE, OUT_SIZE] in [0, 1], references first.

        Returns c2w [N, 4, 4], intrinsics [N, 3, 3] normalised by image size (per-view Stage-2 for the
        references, their mean for the targets), renders [N, 3, OUT_SIZE, OUT_SIZE] in [-1, 1] and
        their VAE latents, the CLIP embedding averaged over the references and the reference tokens
        [n_ref, 5, 2048].
        """
        imgs = images_wm.to(self.device)[None]
        preds1, _ = self.run_worldmirror({"img": imgs}, cond_flags=[0, 0, 0], gaussians=False)

        # Stage 2: references with Stage-1 pose + intrinsics priors (cond_flags = [pose, depth, rays]).
        views2 = {
            "img": imgs[:, :n_ref],
            "camera_pose": preds1["camera_poses"][:, :n_ref],
            "camera_intrinsics": preds1["camera_intrs"][:, :n_ref],
        }
        preds2, tokens = self.run_worldmirror(views2, cond_flags=[1, 0, 1], gaussians=True)
        poses1 = preds1["camera_poses"][0].cpu().numpy()
        tgt_c2w = align_fn(preds2["camera_poses"][0].cpu().numpy(), poses1[:n_ref], poses1[n_ref:])
        c2w = torch.cat([preds2["camera_poses"][0].float(), torch.from_numpy(tgt_c2w).to(self.device)])

        intrs = preds2["camera_intrs"][0].float()
        mean_K = intrs.mean(dim=0, keepdim=True)
        render_K = mean_K.clone()
        render_K[INTR_IDX] *= OUT_SIZE / WM_SIZE
        K = torch.cat([intrs, mean_K.expand(len(c2w) - n_ref, 3, 3)])
        K[INTR_IDX] /= WM_SIZE

        rendered = self.render(preds2["splats"], c2w, render_K)
        refs = images[:n_ref].to(self.device) * 2.0 - 1.0
        return {
            "c2w": c2w,
            "K": K,
            "rendered": rendered,
            "rendered_latents": self.ae.encode(rendered),
            "clip_features": self.clip(refs).mean(dim=0),
            "tokens": tokens,
        }
