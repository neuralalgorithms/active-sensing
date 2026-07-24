#!/usr/bin/env bash
#SBATCH --nodes=1

set -uo pipefail

: "${RAM_RESULTS_ROOT:?}" "${RAM_RESULTS_MONTH:?}" "${RAM_MODEL:?}" "${RAM_SEED_COUNT:?}" "${RAM_GLIMPSES:?}" "${RAM_PATCH_SIZE:?}" "${RAM_WORKERS:?}" "${RAM_CPUS:?}" "${RAM_TOTAL_MODELS:?}" "${RAM_DATA_SRC:?}"
: "${SLURM_ARRAY_JOB_ID:?}" "${SLURM_ARRAY_TASK_ID:?}"

export OMP_NUM_THREADS="${RAM_CPUS}"
export MKL_NUM_THREADS="${RAM_CPUS}"
export TQDM_DISABLE=1

RUN_DIR="${RAM_RUN_DIR:-${RAM_RESULTS_ROOT}/${RAM_RESULTS_MONTH}/${SLURM_ARRAY_JOB_ID}__ram-${RAM_MODEL}}"
PARTS_DIR="${RUN_DIR}/.staging/parts"
LOGS_DIR="${RUN_DIR}/.staging/worker-logs"
STATUS_DIR="${RUN_DIR}/.staging/worker-status"
SCRATCH="${SLURM_TMPDIR:-/tmp}/ram-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"

run_rank() {
    : "${SLURM_PROCID:?}"
    local global_idx=$((SLURM_ARRAY_TASK_ID * RAM_WORKERS + SLURM_PROCID))
    ((global_idx < RAM_TOTAL_MODELS)) || return 0

    local -a glimpses extra_args=()
    local seed glimpse label exit_code
    IFS=',' read -r -a glimpses <<< "$RAM_GLIMPSES"
    seed=$((global_idx % RAM_SEED_COUNT))
    glimpse="${glimpses[$((global_idx / RAM_SEED_COUNT))]}"
    label=$(printf 'p%02d_g%02d_seed%03d' "$RAM_PATCH_SIZE" "$glimpse" "$seed")
    [[ "$RAM_MODEL" == "random" ]] && extra_args=(--random_baseline)
    mkdir -p "${PARTS_DIR}/${label}"

    if python -u train_ram.py \
        --seed "$seed" \
        --glimpses "$glimpse" \
        --patch_size "$RAM_PATCH_SIZE" \
        --data_dir "${SCRATCH}/data" \
        --output_dir "${PARTS_DIR}/${label}" \
        "${extra_args[@]}" \
        > "${LOGS_DIR}/${label}.out" \
        2> "${LOGS_DIR}/${label}.err"; then
        exit_code=0
    else
        exit_code=$?
    fi
    printf '%s\n' "$exit_code" > "${STATUS_DIR}/${label}.status"
    return "$exit_code"
}

if [[ "${RAM_RANK_WORKER:-0}" == 1 ]]; then
    run_rank
    exit $?
fi

source "${HOME}/miniforge3/etc/profile.d/conda.sh"
eval "$(mamba shell hook --shell bash)"
mamba activate torch_gpu

[[ -f "${RUN_DIR}/submission.json" ]] || {
    printf 'ERROR: Missing initialized run metadata: %s/submission.json\n' "$RUN_DIR" >&2
    exit 64
}
[[ -d "$RAM_DATA_SRC" ]] || {
    printf 'ERROR: Data directory not found: %s\n' "$RAM_DATA_SRC" >&2
    exit 64
}

mkdir -p "$PARTS_DIR" "$LOGS_DIR" "$STATUS_DIR" "${SCRATCH}/data"
cp -a "${RAM_DATA_SRC}/." "${SCRATCH}/data/" || exit 64

export RAM_RANK_WORKER=1
if srun --ntasks="${RAM_WORKERS}" --cpus-per-task="${RAM_CPUS}" --kill-on-bad-exit=0 \
    "${BASH_SOURCE[0]}"; then
    step_exit=0
else
    step_exit=$?
fi
rm -rf -- "$SCRATCH"
exit "$step_exit"
