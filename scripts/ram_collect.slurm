#!/usr/bin/env bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

set -uo pipefail

: "${RAM_RUN_DIR:?Missing RAM_RUN_DIR}"
: "${RAM_PARENT_JOB_ID:?Missing RAM_PARENT_JOB_ID}"
: "${RAM_SLURM_LOG_DIR:?Missing RAM_SLURM_LOG_DIR}"

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
eval "$(mamba shell hook --shell bash)"
mamba activate torch_gpu

parent_state="unknown"
if command -v sacct >/dev/null 2>&1; then
    parent_state="$(sacct -X -n -P -j "$RAM_PARENT_JOB_ID" --format=State 2>/dev/null | head -n 1 || true)"
    parent_state="${parent_state:-unknown}"
fi

RAM_PARENT_STATE="$parent_state" \
RAM_COLLECTOR_JOB_ID="${SLURM_JOB_ID:-unknown}" \
RAM_COLLECTOR_STATE="RUNNING" \
    python scripts/collect_ram_results.py
