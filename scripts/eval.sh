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
export HF_HOME=${HF_HOME:-/scratch/cs/gen3r/yejun/huggingface}
PYTHON=${PYTHON:-/scratch/cs/gen3r/yejun/mamba_env/envs/hunyuanworld-mirror/bin/python}  # the single environment (see README)

MODEL=${MODEL:-checkpoints/splatguide.safetensors}
OUT=${OUT:-eval_results/$(basename "${MODEL%.*}")}
D=${DATA:-/scratch/cs/gen3r/yejun/dataset}   # raw benchmark images and split files

run() {  # name data_root split_dir split_num
    "$PYTHON" eval.py --data_root "$2" --split_dir "$3" --split_num "$4" --model_path "$MODEL" --output_dir "$OUT/$1_$4view"
}
for N in 3 6 9; do
    run re10k "$D/final_final_test/real10K_eval/images" "$D/final_final_test/real10K_eval/re10k_split" $N
    run dl3dv "$D/final_final_test/yejun_dl3dv140" "$D/data_split/dl3dv140" $N
    run mipnerf360 "$D/final_final_test/mipnerf360_zihan" "$D/data_split/mipnerf360" $N
done
for N in 3 6; do
    run tnt "$D/tnt-viewcrafter" "$D/tnt-viewcrafter" $N
done
# GT-pose variant (RealEstate10K only): add --gt_pose_dir "$D/final_final_test/real10K_eval"
