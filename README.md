# SplatGuide

Code for **SplatGuide: Geometric Priors from 3D Gaussians for Pose-Free Novel View Synthesis**.

A feed-forward reconstruction model (WorldMirror) turns the unposed input views into
camera poses and a 3DGS scene. The scene conditions a multi-view diffusion model (SEVA) in two ways:

- **Rendered images**: 3DGS renders of all reference and target views, VAE-encoded and
  concatenated to the U-Net input (11 → 15 input channels, new weights zero-initialised).
- **Reconstruction tokens**: per reference view, the camera token and the 4 register tokens
  of WorldMirror's last layer (dim 2048), injected through an extra cross-attention
  (`attn3`/`norm4`) that runs in parallel to SEVA's CLIP cross-attention.

Only the diffusion U-Net is trained. WorldMirror, the VAE and CLIP are frozen and run on the
fly, in training and in evaluation, directly on the raw images; nothing is precomputed or stored.

## Layout

```
train.py                  training entry point (PyTorch Lightning)
eval.py                   evaluation on raw benchmark scenes (PSNR / SSIM / LPIPS)
configs/splatguide.yaml   model and training setting of the paper
scripts/                  SLURM launchers (train, eval)
recon/                    frozen scene encoder, shared by training and evaluation
  worldmirror.py            SceneEncoder: two-stage WorldMirror -> cameras, tokens, 3DGS renders; VAE, CLIP
  scenes.py                 raw datasets: scene discovery, image loading, view sampling, eval splits
  align.py                  Stage-1 -> Stage-2 target pose alignment
seva/                     SEVA U-Net with the SplatGuide changes, denoiser, sampler
  model.py                  Seva U-Net, weight loading/init from SEVA, condition wrapper
  modules/transformer.py    multi-view transformer with token cross-attention (attn3)
  sampling.py               discrete denoiser (training + sampling), multiview CFG, Euler sampler
  inference.py              one sampling pass (do_sample)
  geometry.py               Plücker rays, camera normalisation
training/                 Lightning engine (conditions -> SEVA-weighted loss), raw-image data module
third_party/HunyuanWorld-Mirror   WorldMirror, git submodule pinned to ca63f738
```

## Setup

One environment serves training and evaluation (WorldMirror needs gsplat >= 1.5 and NumPy < 2):

```bash
git clone --recursive <this repo> && cd splatguide   # or: git submodule update --init
conda create -n splatguide python=3.10 && conda activate splatguide
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124   # WorldMirror needs gsplat >= 1.5
pip install -e .
# SEVA v1.1 weights (https://huggingface.co/stabilityai/stable-virtual-camera)
mkdir -p checkpoints && huggingface-cli download stabilityai/stable-virtual-camera modelv1.1.safetensors --local-dir checkpoints
```
WorldMirror (`tencent/HunyuanWorld-Mirror`), the VAE (`stabilityai/stable-diffusion-2-1-base`)
and OpenCLIP ViT-H-14 are downloaded on first use. WorldMirror is used unmodified at commit
`ca63f738` (2025-10-23); newer versions rename the prior keys (`camera_pose` → `camera_poses`),
which silently disables the Stage-2 priors; the weights are pinned to a fixed Hub revision.
On the Triton cluster the environment is the `hunyuanworld-mirror` conda env (torch 2.4,
gsplat 1.5.3, NumPy 1.24) with pytorch-lightning and omegaconf added.

## How WorldMirror is used

For every sample (references first, then targets; see `recon/worldmirror.py`):

1. **Stage 1**: WorldMirror on all views at 448×448, without priors → cameras of every view.
2. **Stage 2**: WorldMirror on the references only, with the Stage-1 reference cameras as
   priors → refined reference cameras, 3D Gaussians, and the camera + 4 register tokens of the
   last aggregator layer (read with a forward hook).
3. **Alignment**: the Stage-1 target poses are mapped into the Stage-2 frame
   (training: mean of `P2 @ inv(P1)` over references; evaluation: median scale + translation).
4. **Rendering**: the Stage-2 Gaussians are rendered with gsplat at 576×576 on a white
   background for all views, with the mean Stage-2 reference intrinsics.

The renders are then VAE-encoded, the references CLIP-encoded, and the cameras turned into
Plücker rays for the diffusion model.

## Data

Raw multi-view images; images are center-cropped to a square and resized (LANCZOS).

| data | layout | views |
|---|---|---|
| DL3DV (train) | `<root>[/<category>]/<scene>/{transforms.json, images_4/}` | 21 random frames (order of `transforms.json`); the first + 1–11 random others are references |
| RealEstate10K (train) | `<root>[/<category>]/<scene>/*.png` | clips with 65–280 frames; 21 strided or random frames; 2–5 references, one in the first and one in the last 30% |
| benchmarks (eval) | `<root>/<scene>/{images*/ or *.png}`, `<split_dir>/<scene>/train_test_split_<N>.json` | `train_ids` → references, `test_ids` → targets |

Training views are drawn afresh every time a scene is visited.

## Training

```bash
python train.py --base configs/splatguide.yaml -n splatguide \
    "data.params.datasets=[{name: dl3dv, root: /path/to/DL3DV-10K, repeat: 3}, {name: re10k, root: /path/to/re10k/train}]" \
    model.network.ckpt_path=checkpoints/modelv1.1.safetensors
# resume: python train.py --resume logs/splatguide
sbatch scripts/train.sh     # 8 x H200, the paper setting
```
Any config value can be overridden with `key=value`. Each GPU processes one 21-frame sample per
step; the effective learning rate is `base_learning_rate x #GPUs x grad accumulation`. The frozen
scene encoder (WorldMirror, VAE, CLIP image tower) sits on every GPU next to the U-Net;
checkpoints contain only the U-Net.

The loss follows SEVA (as implemented in [GeoNVS](https://github.com/MinJunKang/GeoNVS)): L2
between the denoised and the clean latents, weighted per frame by the pose distance to the
nearest reference view (0 for references, up to 5 for the farthest target), without a
sigma-dependent factor.

## Evaluation

```bash
python eval.py --data_root /path/to/real10K_eval/images --split_dir /path/to/real10K_eval/re10k_split \
    --split_num 3 --model_path checkpoints/splatguide.safetensors --output_dir eval_results/re10k_3view
MODEL=... sbatch scripts/eval.sh    # all benchmarks of the paper
```
All scenes under `--data_root` that have a split file are evaluated. The model type is read
from the checkpoint, so the same command also evaluates the SEVA baseline
(`--model_path checkpoints/modelv1.1.safetensors`, i.e. SEVA with WorldMirror poses).
`--gt_pose_dir` swaps in RealEstate10K ground-truth cameras.

Per scene, `eval.py` writes the generated / GT target views, the 3DGS renders and
`evaluation_metrics.json`; `overall_metrics.json` averages over scenes. Scenes with more than
21 frames are generated in groups (all references + up to 21 − #refs targets per pass).
As in SEVA, the cameras of a scene are normalised once (centred on the mean of the non-outlier
cameras, scaled so that the first reference is at distance 2) and shared by all groups.

`--model_path` takes a Lightning `.ckpt` or U-Net weights in `.safetensors`;
`scripts/export_weights.py` converts the former into the latter (without optimizer state).

On the Triton cluster the benchmark images are under `/scratch/cs/gen3r/yejun/dataset`
(see `scripts/eval.sh`). The paper model is `checkpoints/splatguide.safetensors`, exported from
`logs/final_test_yejun_access/epoch=012.ckpt` of
`/scratch/cs/gen3r/yejun/vggt_seva_mirror/vggt_seva_register_token/stable-virtual-camera`
(that `.ckpt` was pickled with NumPy 2 and does not load in the NumPy 1 environment WorldMirror needs).

## Differences from the research code

This repository is a cleaned-up version of that working directory. Changes that affect results:

- **No intermediate files.** The research code ran WorldMirror/VAE/CLIP once and trained and
  evaluated from stored H5 files (for training, 3 fixed view samples per DL3DV scene and 1 per
  RealEstate10K scene). Everything now runs on the fly, and training views are resampled.
- **Training loss.** The research code used an eps-style loss (weight sigma^-2, target frames
  only). It now uses the SEVA weighting described above. The paper checkpoint was trained
  with the old loss.
- **Eval camera normalisation.** The research code scaled single-pass scenes by the largest
  camera coordinate and centred grouped scenes on each group's cameras only. It now
  normalises once per scene as in SEVA (metrics change by less than 0.02 dB).

The DL3DV training data of the paper used 3 / 12 / 9 references for its 3 samples per scene
(the sampler of that run was not kept), and the Mip-NeRF 360 images on the cluster were
replaced after the paper's evaluation, so neither is reproduced exactly.

| research code | here |
|---|---|
| `main_finetune.py`, `eval_h5_data_dl3dv.py` | `train.py`, `eval.py` |
| `configs/example_training/yihao_test.yaml` | `configs/splatguide.yaml` |
| `use_vggt` / `vggt_context_dim` / `vggt_features` | `token_dim` / `tokens` (`attn3` weights unchanged) |
| `use_anysplat` (rendered latents) | always on in training; inferred from the weights in `eval.py` |
| `sgm/` training engine, `seva/sgm_compatible.py` | `training/`, `seva/sampling.py` (one denoiser for training and sampling) |
| `HunyuanWorld-Mirror/preprocess_worldmirror_*.py` | `recon/` |

Checkpoints trained with the research code load unchanged.

The visibility-aware view selection (paper Sec. 3.2) is not part of this repository.

## License

Derived from [Stable Virtual Camera](https://github.com/Stability-AI/stable-virtual-camera)
(Stability AI Non-Commercial License, see `LICENSE`) and
[generative-models](https://github.com/Stability-AI/generative-models). WorldMirror is used
under its own license (`third_party/HunyuanWorld-Mirror/License.txt`).
