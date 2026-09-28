#!/usr/bin/env python3
"""Evaluate SplatGuide (or the SEVA baseline) on raw multi-view scenes with train/test split files.

    python eval.py --data_root RE10K_EVAL/images --split_dir RE10K_EVAL/re10k_split --split_num 3 \
        --model_path WEIGHTS --output_dir OUT

WorldMirror runs on the unposed images of each scene; nothing is precomputed. The architecture is
read from the weights: a 15-channel model is conditioned on the rendered 3DGS latents, a model with
attn3 layers on the reconstruction tokens.
"""
import os

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

import json

import fire
import numpy as np
import torch
from PIL import Image
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from tqdm import tqdm

from recon.align import align_median_scale_shift
from recon.scenes import OUT_SIZE, eval_views, find_scenes, frame_paths, load_images, split_file
from recon.worldmirror import SceneEncoder
from seva.geometry import normalize_cameras, plucker_rays
from seva.inference import create_sampler, do_sample
from seva.model import SGMWrapper
from seva.sampling import DDPMDiscretization, DiscreteDenoiser
from seva.utils import load_seva

DEVICE = "cuda"
MAX_FRAMES = 21  # context window of the diffusion model
CFG, CFG_MIN, NUM_STEPS, CAMERA_SCALE, SEED = 2.0, 1.2, 50, 2.0, 2026


def load_gt_cameras(gt_pose_dir: str, scene_id: str, order: list[int]):
    """RealEstate10K GT cameras for the frames of an eval split (rebuttal
    experiment): relative to frame 0, trajectory scaled so that max ||t|| = 1."""
    num_frames = len(order)
    with open(os.path.join(gt_pose_dir, "camera", f"{scene_id}.txt")) as f:
        lines = f.readlines()[1:]
    c2w = np.zeros((num_frames, 4, 4), dtype=np.float64)
    K = np.zeros((num_frames, 3, 3), dtype=np.float32)
    for i, idx in enumerate(order):
        v = np.array(lines[idx].split(), dtype=np.float64)
        w2c = np.eye(4)
        w2c[:3] = v[7:19].reshape(3, 4)
        c2w[i] = np.linalg.inv(w2c)
        K[i] = np.array([[v[1], 0, v[3]], [0, v[2], v[4]], [0, 0, 1]], dtype=np.float32)
    c2w = np.einsum("ij,njk->nik", np.linalg.inv(c2w[0]), c2w)
    max_t = float(np.linalg.norm(c2w[:, :3, 3], axis=-1).max())
    c2w[:, :3, 3] *= 1.0 / max_t if max_t > 1e-8 else 1.0
    return torch.from_numpy(c2w.astype(np.float32)), torch.from_numpy(K)


def save_images(images: torch.Tensor, indices: list[int], out_dir: str, prefix: str):
    """images: [N, 3, H, W] in [0, 1]."""
    os.makedirs(out_dir, exist_ok=True)
    arrays = (images.cpu().permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8)
    for array, idx in zip(arrays, indices):
        Image.fromarray(array).save(os.path.join(out_dir, f"{prefix}_{idx:03d}.png"), compress_level=1)


class Evaluator:
    def __init__(self, model_path: str, gt_pose_dir: str | None = None):
        seva = load_seva(model_path, DEVICE)
        self.use_render = seva.params.in_channels == 15
        self.use_tokens = seva.params.token_dim is not None
        self.model = SGMWrapper(seva)
        self.gt_pose_dir = gt_pose_dir

        self.encoder = SceneEncoder(DEVICE)
        self.denoiser = DiscreteDenoiser(DDPMDiscretization(), num_idx=1000, device=DEVICE)
        self.sampler = create_sampler(NUM_STEPS, CFG_MIN)
        self.lpips = LearnedPerceptualImagePatchSimilarity(net_type="alex").to(DEVICE)
        self.psnr = PeakSignalNoiseRatio(data_range=1.0).to(DEVICE)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)

    def reconstruct(self, scene: str, split_dir: str, split_num: int) -> dict:
        """Run the scene encoder on the split's views (references first)."""
        ref_ids, tgt_ids = eval_views(split_dir, os.path.basename(scene), split_num)
        frames = frame_paths(scene)
        images_wm, images = load_images([frames[i] for i in ref_ids + tgt_ids])
        data = self.encoder(images_wm, images, len(ref_ids), align_median_scale_shift)
        data["gt_images"] = images.to(DEVICE)
        data["num_refs"] = len(ref_ids)
        if self.gt_pose_dir is not None:
            c2w, K = load_gt_cameras(self.gt_pose_dir, os.path.basename(scene), ref_ids + tgt_ids)
            data["c2w"], data["K"] = c2w.to(DEVICE), K.to(DEVICE)
        return data

    def generate(self, data: dict, frames: list[int]):
        """Run the diffusion model on `frames` (references first, then targets; <= MAX_FRAMES)."""
        T = len(frames)
        c2w = data["c2w"][frames]
        K = data["K"][frames]
        plucker = plucker_rays(c2w, K)  # also rescales K, which the CFG rule reuses

        torch.manual_seed(SEED)
        imgs = data["gt_images"][frames] * 2.0 - 1.0
        value_dict = {
            # Zero-scaled noise as in SEVA (also keeps the CUDA RNG stream of the paper runs).
            "cond_frames": imgs + 0.0 * torch.randn_like(imgs),
            "cond_frames_mask": torch.arange(T, device=DEVICE) < data["num_refs"],
            "plucker_coordinate": plucker,
            "c2w": c2w,
            "K": K,
            "clip_features": data["clip_features"],
            "rendered_latents": data["rendered_latents"][frames] if self.use_render else None,
            "tokens": data["tokens"][None] if self.use_tokens else None,
        }
        return do_sample(self.model, self.encoder.ae, self.denoiser, self.sampler, value_dict, CFG)

    def compute_metrics(self, gt: torch.Tensor, pred: torch.Tensor) -> dict:
        """gt, pred: [N, 3, H, W] in [0, 1]."""
        scores = {"lpips": [], "psnr": [], "ssim": []}
        for g, p in zip(gt[:, None], pred[:, None]):
            scores["lpips"].append(self.lpips(p * 2.0 - 1.0, g * 2.0 - 1.0).item())
            scores["psnr"].append(self.psnr(p, g).item())
            scores["ssim"].append(self.ssim(p, g).item())
        metrics = {}
        for k, v in scores.items():
            metrics[f"{k}_mean"] = float(np.mean(v))
            metrics[f"{k}_std"] = float(np.std(v))
        metrics.update({f"{k}_scores": v for k, v in scores.items()})
        metrics["num_images"] = len(gt)
        return metrics

    def evaluate_scene(self, scene: str, split_dir: str, split_num: int, output_dir: str) -> dict:
        data = self.reconstruct(scene, split_dir, split_num)
        num_frames = len(data["gt_images"])
        refs = list(range(data["num_refs"]))
        targets = list(range(data["num_refs"], num_frames))

        # Cameras are normalised once over the whole scene (as in SEVA), so that all
        # groups share the same frame; the first reference sets the scale.
        data["c2w"] = normalize_cameras(data["c2w"], CAMERA_SCALE)
        if num_frames <= MAX_FRAMES:
            generated = self.generate(data, list(range(num_frames)))
        else:
            # Too many targets for one pass: split them into groups sharing all references.
            generated = torch.zeros(num_frames, 3, OUT_SIZE, OUT_SIZE)
            group_size = MAX_FRAMES - len(refs)
            for start in range(0, len(targets), group_size):
                frames = refs + targets[start : start + group_size]
                generated[frames] = self.generate(data, frames).float()

        pred = ((generated[targets].to(DEVICE) + 1.0) / 2.0).clamp(0.0, 1.0)
        gt = data["gt_images"][targets]
        metrics = self.compute_metrics(gt, pred)

        scene_dir = os.path.join(output_dir, os.path.basename(scene))
        save_images(gt, targets, os.path.join(scene_dir, "gt_images"), "target")
        save_images(pred, targets, os.path.join(scene_dir, "generated_images"), "target")
        save_images((data["rendered"] + 1.0) / 2.0, range(num_frames), os.path.join(scene_dir, "rendered_images"), "rendered")
        with open(os.path.join(scene_dir, "evaluation_metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(
            f"{os.path.basename(scene)}: LPIPS {metrics['lpips_mean']:.4f}  "
            f"PSNR {metrics['psnr_mean']:.2f}  SSIM {metrics['ssim_mean']:.4f}  ({len(targets)} targets)"
        )
        return metrics


def main(
    data_root: str,
    split_dir: str,
    split_num: int,
    model_path: str,
    output_dir: str = "eval_results",
    gt_pose_dir: str | None = None,
):
    """
    Args:
        data_root: directory of scenes (images/ or image files inside each scene folder).
        split_dir: directory with <scene>/train_test_split_<split_num>.json (train_ids = references).
        split_num: number of reference views of the split.
        model_path: SplatGuide weights (`.safetensors` or Lightning `.ckpt`), or SEVA `.safetensors`.
        output_dir: where images and metrics are written.
        gt_pose_dir: RealEstate10K root with `camera/<scene>.txt`; if given, the WorldMirror
            cameras are replaced by GT cameras.
    """
    scenes = [
        s for s in find_scenes(data_root)
        if os.path.exists(split_file(split_dir, os.path.basename(s), split_num))
    ]
    print(f"Found {len(scenes)} scenes with a {split_num}-view split in {data_root}")
    torch.set_grad_enabled(False)
    # Deterministic sampling; WorldMirror's scatter_add has no deterministic kernel, hence warn_only.
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    evaluator = Evaluator(model_path, gt_pose_dir)
    os.makedirs(output_dir, exist_ok=True)

    all_metrics = [
        evaluator.evaluate_scene(s, split_dir, split_num, output_dir) for s in tqdm(scenes, desc="Scenes")
    ]

    overall = {}
    for k in ("lpips", "psnr", "ssim"):
        overall[f"{k}_mean"] = float(np.mean([m[f"{k}_mean"] for m in all_metrics]))
        overall[f"{k}_std"] = float(np.std([m[f"{k}_mean"] for m in all_metrics]))
    overall["num_scenes"] = len(all_metrics)
    with open(os.path.join(output_dir, "overall_metrics.json"), "w") as f:
        json.dump(overall, f, indent=2)
    print(
        f"\n{len(all_metrics)} scenes: LPIPS {overall['lpips_mean']:.4f}  "
        f"PSNR {overall['psnr_mean']:.2f}  SSIM {overall['ssim_mean']:.4f}\nSaved to {output_dir}"
    )


if __name__ == "__main__":
    fire.Fire(main)
