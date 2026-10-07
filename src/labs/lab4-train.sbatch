#!/usr/bin/env bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-node=2
#SBATCH --time=0:30:00
#SBATCH --output=/work/u9216411/cerulean-labs/study-group-online-judge/slurm-out/%j-%x.out

set -euo pipefail
export LAB4_JOB_START_TIME=$(date +%s)
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT=gpt2-training
export WANDB_MODE=online
export EXPERIMENT=prologue

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

module purge
module load cuda/13.0

PROJECT_ROOT="/work/u9216411/cerulean-labs/study-group-online-judge"
cd "$PROJECT_ROOT"

uv run --no-sync torchrun --standalone --nnodes=1 --nproc-per-node=2 \
    src/labs/lab4-train.py --train

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"
