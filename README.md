# SplatGuide

[![arXiv](https://img.shields.io/badge/arXiv-2608.16863-b31b1b.svg)](https://arxiv.org/abs/2608.16863)

Code for **SplatGuide: Geometric Priors from 3D Gaussians for Pose-Free Novel View Synthesis**.

## Setup

```bash
git clone --recursive https://github.com/YejunZhang/splatguide.git && cd splatguide
conda create -n splatguide python=3.10 && conda activate splatguide
pip install torch==2.4.0 torchvision==0.19.0 --index-url https://download.pytorch.org/whl/cu124
pip install gsplat==1.5.3 --index-url https://docs.gsplat.studio/whl/pt24cu124
pip install -e .
mkdir -p checkpoints && huggingface-cli download stabilityai/stable-virtual-camera modelv1.1.safetensors --local-dir checkpoints
```

## Data

- Training: DL3DV (`<root>/<category>/<scene>/{transforms.json, images_4/}`) and RealEstate10K (`<root>/<scene>/*.png`).
- Evaluation: `<root>/<scene>/images/` with `<split_dir>/<scene>/train_test_split_<N>.json`.

## Training

```bash
python train.py --base configs/splatguide.yaml -n splatguide \
    "data.datasets=[{name: dl3dv, root: /path/to/DL3DV-10K, repeat: 3}, {name: re10k, root: /path/to/re10k/train}]"
```

## Evaluation

```bash
python eval.py --data_root /path/to/re10k/images --split_dir /path/to/re10k/splits --split_num 3 \
    --model_path /path/to/splatguide.safetensors --output_dir eval_results/re10k_3view
```

## Citation

```bibtex
@article{zhang2026splatguide,
  title   = {SplatGuide: Geometric Priors from 3D Gaussians for Pose-Free Novel View Synthesis},
  author  = {Zhang, Yejun and Wang, Zihan and Ji, Xu and Wang, Yihao and Hou, Yuxin and Fang, Junyuan and Kilpel{\"a}inen, Juho-Matti and Solin, Arno and Rezazadegan Tavakoli, Hamed and Rahtu, Esa and Kannala, Juho},
  journal = {arXiv preprint arXiv:2608.16863},
  year    = {2026}
}
```

## License

Built on [Stable Virtual Camera](https://github.com/Stability-AI/stable-virtual-camera) (see `LICENSE`) and [HunyuanWorld-Mirror](https://github.com/Tencent-Hunyuan/HunyuanWorld-Mirror).
