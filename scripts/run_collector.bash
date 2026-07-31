#!/usr/bin/env bash
# Manually run the RAM collector locally for a completed/staged compute run.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Accept parameters from positional arguments or environment variables:
# Usage: ./scripts/run_collector.bash [RUN_DIR] [PARENT_JOB_ID]
RUN_DIR="${1:-${RAM_RUN_DIR:-}}"
PARENT_JOB_ID="${2:-${RAM_PARENT_JOB_ID:-}}"

if [[ -z "${RUN_DIR}" ]]; then
    echo "Usage: $0 <RUN_DIR> [PARENT_JOB_ID]" >&2
    echo "Example: $0 results/2026-07/307898__ram-random" >&2
    exit 1
fi

if [[ ! -d "${RUN_DIR}" ]]; then
    echo "Error: Run directory '${RUN_DIR}' does not exist." >&2
    exit 1
fi

# Resolve absolute path for RUN_DIR
RAM_RUN_DIR="$(cd "${RUN_DIR}" && pwd)"

# Extract leading digits from directory basename if PARENT_JOB_ID not explicitly passed
if [[ -z "${PARENT_JOB_ID}" ]]; then
    DIR_NAME="$(basename "${RAM_RUN_DIR}")"
    PARENT_JOB_ID="${DIR_NAME%%_*}"
fi

if [[ ! "${PARENT_JOB_ID}" =~ ^[0-9]+$ ]]; then
    echo "Error: Could not extract numeric PARENT_JOB_ID from directory name '${RAM_RUN_DIR}'." >&2
    exit 1
fi

RAM_PARENT_JOB_ID="${PARENT_JOB_ID}"
RAM_SLURM_LOG_DIR="$(dirname "${RAM_RUN_DIR}")/.slurm-staging"

# Python interpreter setup (prefer gpython environment)
PYTHON_CMD="python3"
if [[ -x "${HOME}/Projects/01-Python/gp/.venv/bin/python" ]]; then
    PYTHON_CMD="${HOME}/Projects/01-Python/gp/.venv/bin/python"
fi

echo "=== Running RAM Collector (Local Execution) ==="
echo "  Run Directory:       ${RAM_RUN_DIR}"
echo "  Parent Job ID:       ${RAM_PARENT_JOB_ID}"
echo "  Slurm Log Directory: ${RAM_SLURM_LOG_DIR}"
echo "  Python Binary:       ${PYTHON_CMD}"
echo "=============================================="

export RAM_RUN_DIR
export RAM_PARENT_JOB_ID="${RAM_PARENT_JOB_ID}"
export RAM_SLURM_LOG_DIR
export RAM_PARENT_STATE="COMPLETED"
export RAM_COLLECTOR_JOB_ID="${RAM_COLLECTOR_JOB_ID:-${RAM_PARENT_JOB_ID}_collect}"
export RAM_COLLECTOR_STATE="RUNNING"

cd "${PROJECT_ROOT}"
exec "${PYTHON_CMD}" scripts/collect_ram_results.py
