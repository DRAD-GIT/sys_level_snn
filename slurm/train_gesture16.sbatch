#!/bin/bash
# Slurm job: train the 16x16, 10-class DVS Gesture SNN (tools/train_gesture16.py).
#
#   sbatch slurm/train_gesture16.sbatch                   # from the repository folder
#   EPOCHS=150 LOSS=ce sbatch slurm/train_gesture16.sbatch
#
# Resubmitting continues from the last finished epoch (logs/gesture16_train/last.pth).
# The first run reads the dataset once and caches it (logs/gesture16_train/data.pt).
#SBATCH --job-name=gesture16-snn
#SBATCH --partition=general          # GPU nodes: 4x RTX 2080 Ti, 1-day limit
#SBATCH --gres=gpu:RTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_gesture16_%j.out

set -eo pipefail      # no -u: module and venv scripts read unset variables
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python/3.12
source "${VENV:-/shares/bulk/yashbiyani/c3cim_sys_cluster/c3cim}/bin/activate"

DATA=${DATA:-/shares/bulk/yashbiyani/c3cim_sys_cluster/datasets}   # folder holding the Gesture dataset
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-64}
LR=${LR:-0.001}
LOSS=${LOSS:-mse}
WORK=${WORK:-logs/gesture16_train}

RESUME=""
if [ -f "$WORK/last.pth" ]; then
    RESUME="--resume"
fi

echo "job $SLURM_JOB_ID on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}, $(date)"
python -u tools/train_gesture16.py --data "$DATA" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" \
    --loss "$LOSS" --work "$WORK" --workers 8 $RESUME
echo "finished $(date)"
