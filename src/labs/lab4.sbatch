#!/usr/bin/env bash
#SBATCH --account=ACD115198
#SBATCH --partition=8gpus
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=2
#SBATCH --time=0:30:00

echo "========================================"
echo "Start Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "========================================"
START_TIME=$(date +%s)

module purge
module load cuda/13.0

cd "$JUDGE_SUBMISSION_DIR"
export PYTHONPATH="$JUDGE_TRUSTED_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"

uv sync --no-sources --no-dev
uv run --no-sync python -u -m judge.run_task \
  --task "$JUDGE_TASK_ID" \
  --submission "$JUDGE_SUBMISSION_DIR" \
  --output "$JUDGE_OUTPUT_DIR/result.json"

END_TIME=$(date +%s)
echo "========================================"
echo "End Time: $(date '+%Y-%m-%d %H:%M:%S')  "
echo "Total Cost: $((END_TIME - START_TIME)) s"
echo "========================================"
set -euo pipefail
