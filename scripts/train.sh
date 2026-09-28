#!/bin/bash -l
#SBATCH --job-name=splatguide_train
#SBATCH --time=30:00:00
#SBATCH --gres=gpu:8
#SBATCH --account=ellis_users
#SBATCH --partition=gpu-h200-141g-ellis
#SBATCH --mem=1200G
#SBATCH --cpus-per-task=24
#SBATCH --output=job_output_%j.txt
# Usage: sbatch scripts/train.sh [extra train.py args, e.g. key=value overrides]
set -euo pipefail
cd "$(dirname "$0")/.."
export HF_HOME=${HF_HOME:-/scratch/cs/gen3r/yejun/huggingface}
PYTHON=${PYTHON:-/scratch/cs/gen3r/yejun/mamba_env/envs/hunyuanworld-mirror/bin/python}  # the single environment (see README)

# Dataset roots are set in configs/splatguide.yaml (data.params.datasets).
$PYTHON train.py --base configs/splatguide.yaml "$@" -n "${RUN_NAME:-splatguide}" \
    model.network.ckpt_path="${SEVA_CKPT:-checkpoints/modelv1.1.safetensors}"
