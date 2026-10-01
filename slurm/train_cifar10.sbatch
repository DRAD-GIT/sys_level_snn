#!/bin/bash
# Slurm job: train the CIFAR-10 VGG-11 SNN (tools/train_cifar10.py).
#
#   sbatch slurm/train_cifar10.sbatch                     # from the repository folder
#   DATA=/path/to/datasets sbatch slurm/train_cifar10.sbatch
#
# Resubmitting continues from the last finished epoch (logs/cifar10_train/last.pth),
# so a job stopped by its time limit is resumed by submitting it again.
# Adapt the #SBATCH lines to the cluster (sinfo -o "%P %G %l" lists partitions,
# GPUs and time limits).
#SBATCH --job-name=cifar10-snn
#SBATCH --partition=general          # GPU nodes: 4x RTX 2080 Ti, 1-day limit
#SBATCH --gres=gpu:RTX2080Ti:2       # GPUs (match GPUS below)
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00              # ~5 h expected; within the partition's 1-day limit
#SBATCH --output=logs/slurm_cifar10_%j.out

set -eo pipefail      # no -u: module and venv scripts read unset variables
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

# Python environment: the python module and a virtual environment holding
# torch, numpy and pyyaml (VENV; see README).
module load python/3.12
source "${VENV:-/shares/bulk/yashbiyani/c3cim_sys_cluster/c3cim}/bin/activate"

DATA=${DATA:-/shares/bulk/yashbiyani/c3cim_sys_cluster/datasets}   # folder holding cifar-10-batches-py
GPUS=${GPUS:-2}
BATCH=${BATCH:-256}
LR=${LR:-0.1}
EPOCHS=${EPOCHS:-200}
WORK=logs/cifar10_train

RESUME=""
if [ -f "$WORK/last.pth" ]; then
    RESUME="--resume"
fi

echo "job $SLURM_JOB_ID on $(hostname), GPUs: ${CUDA_VISIBLE_DEVICES:-none}, $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv
python -u tools/train_cifar10.py --amp --data "$DATA" --gpus "$GPUS" --batch "$BATCH" \
    --lr "$LR" --epochs "$EPOCHS" --work "$WORK" $RESUME
echo "finished $(date)"
