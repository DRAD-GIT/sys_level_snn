#!/bin/bash
# Slurm job: train a small downscaled event SNN (tools/train_event_snn.py):
# MODEL=gesture16 (DVS Gesture 16x16, default) or MODEL=nmnist17 (N-MNIST 17x17).
#
#   sbatch slurm/train_event_snn.sbatch                   # from the repository folder
#   MODEL=nmnist17 EPOCHS=50 sbatch slurm/train_event_snn.sbatch
#   # a variant without editing the YAML: its own NAME (work folder and checkpoint)
#   NAME=s30_n80 STEP=30 WINDOW=2400 sbatch slurm/train_event_snn.sbatch
#
# Resubmitting continues from the last finished epoch (logs/<MODEL>_train/last.pth).
# The first run reads the dataset once and caches it (logs/<MODEL>_train/data.pt).
#SBATCH --job-name=event-snn
#SBATCH --partition=general          # GPU nodes: 4x RTX 2080 Ti, 1-day limit
#SBATCH --gres=gpu:RTX2080Ti:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm_event_snn_%j.out

set -eo pipefail      # no -u: module and venv scripts read unset variables
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
module load python/3.12
source "${VENV:-/shares/bulk/yashbiyani/c3cim_sys_cluster/c3cim}/bin/activate"

MODEL=${MODEL:-gesture16}
DATA=${DATA:-/shares/bulk/yashbiyani/c3cim_sys_cluster/datasets}   # folder holding the Gesture dataset
EPOCHS=${EPOCHS:-100}
BATCH=${BATCH:-64}
LR=${LR:-0.001}
LOSS=${LOSS:-mse}
WORK=${WORK:-logs/${MODEL}_train}
EXTRA=""
if [ -n "$NAME" ]; then          # a variant: own work folder and checkpoint, shared data cache
    WORK=logs/${MODEL}_$NAME
    EXTRA="--out pretrained/try/${MODEL}_$NAME.pth --data-cache logs/${MODEL}_train/data.pt"
fi
[ -n "$WINDOW" ] && EXTRA="$EXTRA --window-ms $WINDOW"
[ -n "$STEP" ] && EXTRA="$EXTRA --step-ms $STEP"
[ -n "$HIDDEN" ] && EXTRA="$EXTRA --hidden $HIDDEN"

RESUME=""
if [ -f "$WORK/last.pth" ]; then
    RESUME="--resume"
fi

echo "job $SLURM_JOB_ID on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}, $(date)"
python -u tools/train_event_snn.py --model "$MODEL" --data "$DATA" --epochs "$EPOCHS" --batch "$BATCH" --lr "$LR" \
    --loss "$LOSS" --work "$WORK" --workers 8 $RESUME $EXTRA
echo "finished $(date)"
