#!/usr/bin/env python3
"""Edit the control panel below, then run this file to submit a RAM array."""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# ---- Control panel: configure today's run here. ----
MODEL = "random"                         # "policy" or "random"
SEED_COUNT = 64                            # Seeds are 0 through SEED_COUNT - 1
GLIMPSES = (1,2,3,4,5,6,7)
PATCH_SIZE = 8
WORKERS_PER_ARRAY_TASK = 8
CPUS_PER_WORKER = 2
MAX_ACTIVE_ARRAY_TASKS = 4
MEMORY_PER_ARRAY_TASK = "8G"		 # --mem
WALL_TIME = "01:00:00"
PARTITION = "normal"
RESULTS_ROOT = Path.home() / "work" / "active-sensing" / "results"
DRY_RUN = True                           # If True, print mapping only; do not call Slurm
VERBOSE_DRY_RUN = False                  # If True, print line-by-line task mapping during dry run

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
COMPUTE_SCRIPT = SCRIPTS_DIR / "ram_compute.slurm"
COLLECT_SCRIPT = SCRIPTS_DIR / "ram_collect.slurm"


def validate() -> None:
    if MODEL not in {"policy", "random"}:
        raise RuntimeError("MODEL must be 'policy' or 'random'")
    if not GLIMPSES or len(set(GLIMPSES)) != len(GLIMPSES) or min(GLIMPSES) <= 0:
        raise RuntimeError("GLIMPSES must be unique positive integers")
    for name, value in {
        "SEED_COUNT": SEED_COUNT,
        "PATCH_SIZE": PATCH_SIZE,
        "WORKERS_PER_ARRAY_TASK": WORKERS_PER_ARRAY_TASK,
        "CPUS_PER_WORKER": CPUS_PER_WORKER,
        "MAX_ACTIVE_ARRAY_TASKS": MAX_ACTIVE_ARRAY_TASKS,
    }.items():
        if not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"{name} must be a positive integer")
    if not all((MEMORY_PER_ARRAY_TASK, WALL_TIME, PARTITION, str(RESULTS_ROOT))):
        raise RuntimeError("memory, time, partition, and results root must not be empty")


def run_sbatch(command: list[str], environment: dict[str, str], description: str) -> str:
    try:
        completed = subprocess.run(
            command, env=environment, check=True, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
    except FileNotFoundError as error:
        raise RuntimeError("sbatch is not available") from error
    except subprocess.CalledProcessError as error:
        detail = error.stderr.strip() or error.stdout.strip() or str(error)
        raise RuntimeError(f"Could not submit {description}: {detail}") from error
    job_id = completed.stdout.strip().split(";", maxsplit=1)[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Unexpected {description} sbatch response: {completed.stdout!r}")
    return job_id


def git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "unknown"


def atomic_json(destination: Path, payload: dict) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def sbatch_base() -> list[str]:
    return [
        "sbatch", "--parsable", f"--partition={PARTITION}",
        f"--chdir={PROJECT_ROOT}", "--nodes=1", "--export=ALL",
    ]


def derive_array() -> tuple[int, int, str, int]:
    # ---- Derived array calculation (what actually gets passed to Slurm) ----
    # total models = seeds * glimpses (e.g. 32 * 7 = 224)
    total_models = SEED_COUNT * len(GLIMPSES)
    # array elements = ceiling of (total models / workers per task)
    array_elements = math.ceil(total_models / WORKERS_PER_ARRAY_TASK)
    # This generates the string passed to Slurm, e.g. "0-27%4"
    array_spec = f"0-{array_elements - 1}%{MAX_ACTIVE_ARRAY_TASKS}"
    # How many padded slots at the tail end do nothing
    idle_slots = array_elements * WORKERS_PER_ARRAY_TASK - total_models
    return total_models, array_elements, array_spec, idle_slots


def print_mapping(total_models: int, array_elements: int, array_spec: str, idle_slots: int) -> None:
    print(f"Model: {MODEL}")
    print(f"Seeds: 0-{SEED_COUNT - 1} ({SEED_COUNT} seeds)")
    print(f"Glimpses: {','.join(map(str, GLIMPSES))} ({len(GLIMPSES)} values)")
    print(f"Total models: {total_models}")
    print(f"Array elements: {array_elements}")
    print(f"Array: --array={array_spec}")
    print(f"Workers per array element: {WORKERS_PER_ARRAY_TASK}")
    print(f"CPUs per worker: {CPUS_PER_WORKER}")
    print(f"Maximum active elements: {MAX_ACTIVE_ARRAY_TASKS}")
    print(f"Idle tail slots: {idle_slots}")

    print("\nDistribution Summary:")
    print(f"{'Array Tasks':<12} | {'Glimpse':<7} | {'Seeds':<10} | {'Models':<6}")
    print("-" * 44)
    for i, glimpse in enumerate(GLIMPSES):
        start_idx = i * SEED_COUNT
        end_idx = start_idx + SEED_COUNT - 1
        start_task = start_idx // WORKERS_PER_ARRAY_TASK
        end_task = end_idx // WORKERS_PER_ARRAY_TASK
        
        task_str = f"{start_task}..{end_task}" if start_task != end_task else str(start_task)
        seed_str = f"0..{SEED_COUNT - 1}"
        
        print(f"{task_str:<12} | {glimpse:<7} | {seed_str:<10} | {SEED_COUNT:<6}")

    if VERBOSE_DRY_RUN:
        print("\nTask mapping (glimpse-major):")
        for global_index in range(total_models):
            task_id, slot = divmod(global_index, WORKERS_PER_ARRAY_TASK)
            glimpse_index, seed = divmod(global_index, SEED_COUNT)
            print(f"  task={task_id} slot={slot} global_index={global_index} glimpse={GLIMPSES[glimpse_index]} seed={seed}")


def submission_payload(array_job_id: str, total_models: int, array_elements: int, array_spec: str, idle_slots: int) -> dict:
    return {
        "schema_version": 3,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "experiment": f"ram-{MODEL}",
        "model": {"type": MODEL, "class": "RecurrentAttentionModelClassic", "random_baseline": MODEL == "random"},
        "configuration": {
            "seed_count": SEED_COUNT,
            "glimpses": list(GLIMPSES),
            "patch_size": PATCH_SIZE,
        },
        "task_mapping": {
            "ordering": "glimpse-major",
            "workers_per_array_task": WORKERS_PER_ARRAY_TASK,
            "total_models": total_models,
            "array_elements": array_elements,
            "idle_tail_slots": idle_slots,
        },
        "resources": {
            "cpus_per_worker": CPUS_PER_WORKER,
            "memory_per_array_task": MEMORY_PER_ARRAY_TASK,
            "wall_time": WALL_TIME,
            "partition": PARTITION,
        },
        "slurm": {
            "array_job_id": int(array_job_id),
            "array_spec": array_spec,
            "array_state_at_submission": "HELD",
            "collector_job_id": None,
        },
        "git_revision": git_revision(),
    }


def main() -> int:
    validate()
    total_models, array_elements, array_spec, idle_slots = derive_array()
    if DRY_RUN:
        print_mapping(total_models, array_elements, array_spec, idle_slots)
        return 0
    if not COMPUTE_SCRIPT.is_file() or not COLLECT_SCRIPT.is_file():
        raise RuntimeError("Missing RAM compute or collector runner script")

    month = datetime.now(timezone.utc).strftime("%Y-%m")
    results_dir = RESULTS_ROOT / month
    log_dir = results_dir / ".slurm-staging"
    log_dir.mkdir(parents=True, exist_ok=True)
    compute_environment = os.environ | {
        "RAM_RESULTS_ROOT": str(RESULTS_ROOT),
        "RAM_RESULTS_MONTH": month,
        "RAM_MODEL": MODEL,
        "RAM_SEED_COUNT": str(SEED_COUNT),
        "RAM_GLIMPSES": ",".join(map(str, GLIMPSES)),
        "RAM_PATCH_SIZE": str(PATCH_SIZE),
        "RAM_WORKERS": str(WORKERS_PER_ARRAY_TASK),
        "RAM_CPUS": str(CPUS_PER_WORKER),
        "RAM_TOTAL_MODELS": str(total_models),
        "RAM_DATA_SRC": str(PROJECT_ROOT / "data"),
    }
    compute_command = sbatch_base() + [
        "--hold", f"--job-name=ram-{MODEL}",
        f"--ntasks={WORKERS_PER_ARRAY_TASK}", f"--cpus-per-task={CPUS_PER_WORKER}",
        f"--mem={MEMORY_PER_ARRAY_TASK}", f"--time={WALL_TIME}", f"--array={array_spec}",
        f"--output={log_dir}/compute_%A_%a.out", f"--error={log_dir}/compute_%A_%a.err",
        str(COMPUTE_SCRIPT),
    ]
    array_job_id = run_sbatch(compute_command, compute_environment, "held compute array")

    run_dir = results_dir / f"{array_job_id}__ram-{MODEL}"
    run_dir.mkdir(parents=True, exist_ok=True)
    collector_environment = os.environ | {
        "RAM_RUN_DIR": str(run_dir),
        "RAM_PARENT_JOB_ID": array_job_id,
        "RAM_SLURM_LOG_DIR": str(log_dir),
    }
    collector_command = sbatch_base() + [
        f"--job-name=ram-{MODEL}-collect", "--ntasks=1", "--cpus-per-task=1",
        "--mem=2G", "--time=00:10:00", f"--dependency=afterany:{array_job_id}",
        f"--output={log_dir}/collector_%j.out", f"--error={log_dir}/collector_%j.err",
        str(COLLECT_SCRIPT),
    ]
    try:
        collector_job_id = run_sbatch(collector_command, collector_environment, "afterany collector")
    except RuntimeError as error:
        raise RuntimeError(f"{error}; compute array {array_job_id} remains held") from error

    payload = submission_payload(array_job_id, total_models, array_elements, array_spec, idle_slots)
    payload["slurm"]["collector_job_id"] = int(collector_job_id)
    atomic_json(run_dir / "submission.json", payload)
    try:
        subprocess.run(["scontrol", "release", array_job_id], check=True)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            f"Could not release array {array_job_id}; it and collector {collector_job_id} remain submitted"
        ) from error

    print(f"Compute array: {array_job_id} ({array_spec})")
    print(f"Collector job: {collector_job_id} (afterany:{array_job_id})")
    print(f"Run directory: {run_dir}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(64)
