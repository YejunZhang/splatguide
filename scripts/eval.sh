#!/bin/bash -l
#SBATCH --job-name=splatguide_eval
#SBATCH --time=12:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-h200-141g-ellis,gpu-h100-80g
#SBATCH --mem=100G
#SBATCH --cpus-per-task=8
#SBATCH --output=job_output_%j.txt
# Usage: MODEL=path/to/model.{safetensors,ckpt} sbatch scripts/eval.sh
# MODEL may also be the SEVA .safetensors (baseline with WorldMirror poses).
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHON=${PYTHON:-python}

MODEL=${MODEL:-checkpoints/splatguide.safetensors}
OUT=${OUT:-eval_results/$(basename "${MODEL%.*}")}
D=${DATA:-data/benchmarks}   # <benchmark>/images/<scene>/..., <benchmark>/splits/<scene>/train_test_split_<N>.json

run() {  # name data_root split_dir split_num
    "$PYTHON" eval.py --data_root "$2" --split_dir "$3" --split_num "$4" --model_path "$MODEL" --output_dir "$OUT/$1_$4view"
}
for N in 3 6 9; do
    run re10k "$D/re10k/images" "$D/re10k/splits" $N
    run dl3dv "$D/dl3dv/images" "$D/dl3dv/splits" $N
    run mipnerf360 "$D/mipnerf360/images" "$D/mipnerf360/splits" $N
done
for N in 3 6; do
    run tnt "$D/tnt/images" "$D/tnt/images" $N
done
