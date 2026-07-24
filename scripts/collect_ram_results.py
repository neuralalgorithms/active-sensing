#!/usr/bin/env python3
"""Collect staged RAM workers into atomic condition-level result artifacts."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tarfile
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


EXIT_COMPLETE = 0
EXIT_PARTIAL = 10
EXIT_ABORTED = 20
EXIT_FAILED = 30
EXIT_COLLECTION_ERROR = 40
REQUIRED_COLUMNS = {"model_type", "patch_size", "glimpses", "seed", "epoch"}


def environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def safe_collector_id(value: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in value)


def atomic_json(destination: Path, payload: dict, collector_id: str) -> None:
    temporary = destination.with_name(f".{destination.name}.tmp.{collector_id}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, destination)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_details(path: Path) -> dict:
    return {"bytes": path.stat().st_size, "sha256": sha256(path)}


def read_submission(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"Missing initialized submission metadata: {path}")
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Invalid submission metadata: {path}") from error


def read_statuses(status_directory: Path) -> tuple[dict[str, dict], list[dict]]:
    statuses: dict[str, dict] = {}
    errors: list[dict] = []
    if not status_directory.is_dir():
        return statuses, errors
    for path in sorted(status_directory.glob("*.status")):
        try:
            worker = path.stem
            if worker in statuses:
                raise ValueError(f"duplicate status for {worker}")
            statuses[worker] = {"worker": worker, "exit_code": int(path.read_text().strip())}
        except (OSError, ValueError) as error:
            errors.append({"path": str(path), "error": str(error)})
    return statuses, errors


def staged_csv_paths(parts_directory: Path) -> list[Path]:
    if not parts_directory.is_dir():
        return []
    return sorted(set(parts_directory.rglob("*.csv")) | set(parts_directory.rglob("*.csv.gz")))


def collect_frames(
    parts_directory: Path,
    submission: dict,
) -> tuple[dict[tuple[int, int], list[pd.DataFrame]], list[dict]]:
    groups: dict[tuple[int, int], list[pd.DataFrame]] = defaultdict(list)
    invalid_sources: list[dict] = []
    configuration = submission["configuration"]
    expected_model = submission["model"]["type"]
    expected_patch = int(configuration["patch_size"])
    expected_glimpses = {int(value) for value in configuration["glimpses"]}
    expected_seeds = set(range(int(configuration["seed_count"])))
    numeric_columns = ["patch_size", "glimpses", "seed", "epoch"]

    for path in staged_csv_paths(parts_directory):
        try:
            frame = pd.read_csv(path)
            missing = REQUIRED_COLUMNS - set(frame.columns)
            if missing or frame.empty:
                raise ValueError(f"missing columns {sorted(missing)} or empty CSV")
            frame = frame.copy()
            numeric = frame[numeric_columns].apply(pd.to_numeric, errors="raise")
            if numeric.isna().any().any() or ((numeric % 1) != 0).any().any():
                raise ValueError("condition columns must contain integers")
            frame[numeric_columns] = numeric.astype("int64")

            conditions = list(frame.groupby(["patch_size", "glimpses"], sort=False).groups)
            models = frame["model_type"].dropna().unique()
            seeds = frame["seed"].unique()
            if len(conditions) != 1 or len(models) != 1 or len(seeds) != 1:
                raise ValueError("each worker CSV must contain one model, condition, and seed")
            patch_size, glimpse = (int(value) for value in conditions[0])
            seed = int(seeds[0])
            if str(models[0]) != expected_model or patch_size != expected_patch:
                raise ValueError("model type or patch size does not match submission metadata")
            if glimpse not in expected_glimpses or seed not in expected_seeds:
                raise ValueError("glimpse or seed does not match submission metadata")

            expected_worker = f"p{patch_size:02d}_g{glimpse:02d}_seed{seed:03d}"
            staged_worker = path.relative_to(parts_directory).parts[0]
            if staged_worker != expected_worker:
                raise ValueError(f"staged worker {staged_worker!r} does not match {expected_worker!r}")
            groups[(patch_size, glimpse)].append(frame)
        except Exception as error:
            invalid_sources.append({"path": str(path), "error": str(error)})

    return groups, invalid_sources


def publish_conditions(
    run_directory: Path,
    groups: dict[tuple[int, int], list[pd.DataFrame]],
    invalid_sources: list[dict],
    collector_id: str,
) -> dict[str, dict]:
    published: dict[str, dict] = {}
    identity = ["model_type", "patch_size", "glimpses", "seed", "epoch"]

    for (patch_size, glimpse), frames in sorted(groups.items()):
        combined = pd.concat(frames, ignore_index=True)
        duplicate_rows = combined.duplicated(identity, keep=False)
        if duplicate_rows.any():
            duplicate_count = int(duplicate_rows.sum())
            invalid_sources.append(
                {
                    "condition": f"p{patch_size:02d}_g{glimpse:02d}",
                    "error": f"{duplicate_count} duplicate model/patch/glimpse/seed/epoch rows",
                }
            )
            continue

        combined = combined.sort_values(["seed", "epoch"], kind="stable")
        filename = f"p{patch_size:02d}_g{glimpse:02d}.csv.gz"
        destination = run_directory / filename
        temporary = destination.with_name(f".{filename}.tmp.{collector_id}")
        combined.to_csv(
            temporary,
            index=False,
            compression={"method": "gzip", "mtime": 0},
        )
        verified = pd.read_csv(temporary, compression="gzip")
        if len(verified) != len(combined) or list(verified.columns) != list(combined.columns):
            temporary.unlink(missing_ok=True)
            raise RuntimeError(f"verification failed for {filename}")
        os.replace(temporary, destination)
        details = artifact_details(destination)
        published[filename] = {
            "rows": int(len(combined)),
            "seeds": sorted(int(seed) for seed in combined["seed"].unique()),
            **details,
        }
    return published


def atomic_archive(
    destination: Path,
    members: Iterable[tuple[Path, str]],
    collector_id: str,
) -> dict:
    temporary = destination.with_name(f".{destination.name}.tmp.{collector_id}")
    with tarfile.open(temporary, "w:gz") as archive:
        for path, archive_name in sorted(members, key=lambda item: item[1]):
            if path.is_file():
                archive.add(path, arcname=archive_name)
    os.replace(temporary, destination)
    return artifact_details(destination)


def worker_log_members(directory: Path) -> list[tuple[Path, str]]:
    if not directory.is_dir():
        return []
    return [
        (path, str(path.relative_to(directory)))
        for path in directory.rglob("*")
        if path.is_file()
    ]


def slurm_log_members(directory: Path, parent_job_id: int) -> list[tuple[Path, str]]:
    if not directory.is_dir():
        return []
    paths = list(directory.glob(f"compute_{parent_job_id}_*.out"))
    paths.extend(directory.glob(f"compute_{parent_job_id}_*.err"))
    return [(path, path.name) for path in paths if path.is_file()]


def worker_labels(configuration: dict) -> set[str]:
    patch_size = int(configuration["patch_size"])
    return {
        f"p{patch_size:02d}_g{int(glimpse):02d}_seed{seed:03d}"
        for glimpse in configuration["glimpses"]
        for seed in range(int(configuration["seed_count"]))
    }

def expected_condition_names(configuration: dict) -> set[str]:
    patch_size = configuration.get("patch_size")
    if patch_size is None:
        return set()
    return {
        f"p{int(patch_size):02d}_g{int(glimpse):02d}.csv.gz"
        for glimpse in configuration.get("glimpses", [])
    }


def coverage_details(configuration: dict, published: dict[str, dict]) -> dict:
    expected_seeds = list(range(int(configuration["seed_count"])))
    expected_seed_set = set(expected_seeds)
    patch_size = configuration.get("patch_size")
    conditions: dict[str, dict] = {}

    if patch_size is not None:
        for glimpse in configuration.get("glimpses", []):
            filename = f"p{int(patch_size):02d}_g{int(glimpse):02d}.csv.gz"
            observed = set(published.get(filename, {}).get("seeds", []))
            conditions[filename] = {
                "expected_seeds": expected_seeds,
                "observed_seeds": sorted(observed),
                "missing_seeds": sorted(expected_seed_set - observed),
                "unexpected_seeds": sorted(observed - expected_seed_set),
                "complete": observed == expected_seed_set,
            }

    return {
        "expected_models": len(expected_seeds) * len(configuration.get("glimpses", [])),
        "published_seed_conditions": sum(
            len(details.get("seeds", [])) for details in published.values()
        ),
        "conditions": conditions,
        "complete": bool(conditions) and all(item["complete"] for item in conditions.values()),
    }


def collection_error_manifest(
    submission: dict,
    parent_job_id: int,
    parent_state: str,
    collector_id: str,
    collector_state: str,
    error: Exception,
) -> dict:
    return {
        "schema_version": 2,
        "collected_at": timestamp(),
        "status": "collection_error",
        "collector_exit_code": EXIT_COLLECTION_ERROR,
        "error": str(error),
        "experiment": submission.get("experiment", "unknown"),
        "model": submission.get("model", {}),
        "slurm": {
            **submission.get("slurm", {}),
            "array_job_id": parent_job_id,
            "array_state": parent_state,
            "collector_job_id": collector_id,
            "collector_state": collector_state,
        },
    }


def main() -> int:
    run_directory = Path(environment("RAM_RUN_DIR"))
    parent_job_id = int(environment("RAM_PARENT_JOB_ID"))
    parent_state = os.environ.get("RAM_PARENT_STATE", "unknown")
    collector_id = safe_collector_id(os.environ.get("RAM_COLLECTOR_JOB_ID", "unknown"))
    collector_state = os.environ.get("RAM_COLLECTOR_STATE", "RUNNING")
    slurm_log_directory = Path(environment("RAM_SLURM_LOG_DIR"))
    run_directory.mkdir(parents=True, exist_ok=True)

    staging_directory = run_directory / ".staging"
    submission = read_submission(run_directory / "submission.json")
    try:
        statuses, status_errors = read_statuses(staging_directory / "worker-status")
        groups, invalid_sources = collect_frames(staging_directory / "parts", submission)
        published = publish_conditions(run_directory, groups, invalid_sources, collector_id)

        worker_archive = run_directory / "worker-logs.tar.gz"
        worker_archive_details = atomic_archive(
            worker_archive,
            worker_log_members(staging_directory / "worker-logs"),
            collector_id,
        )
        slurm_members = slurm_log_members(slurm_log_directory, parent_job_id)
        slurm_archive = run_directory / "slurm-logs.tar.gz"
        slurm_archive_details = atomic_archive(
            slurm_archive,
            slurm_members,
            collector_id,
        )
    except Exception as exc:
        manifest = collection_error_manifest(
            submission,
            parent_job_id,
            parent_state,
            collector_id,
            collector_state,
            exc,
        )
        atomic_json(run_directory / "manifest.json", manifest, collector_id)
        return EXIT_COLLECTION_ERROR

    configuration = submission.get("configuration", {})
    expected_workers = worker_labels(configuration)
    observed_workers = set(statuses)
    successful_workers = sorted(
        worker for worker, status in statuses.items() if status.get("exit_code") == 0
    )
    failed_workers = sorted(
        worker for worker, status in statuses.items() if status.get("exit_code") != 0
    )
    missing_workers = sorted(expected_workers - observed_workers)
    unexpected_workers = sorted(observed_workers - expected_workers)
    expected_conditions = expected_condition_names(configuration)
    missing_conditions = sorted(expected_conditions - set(published))
    coverage = coverage_details(configuration, published)
    incomplete_conditions = sorted(
        filename
        for filename, details in coverage["conditions"].items()
        if not details["complete"] and filename in published
    )

    cancelled = parent_state.upper().startswith("CANCELLED")
    all_workers_complete = bool(expected_workers) and not (
        missing_workers or failed_workers or unexpected_workers or status_errors
    )
    if cancelled:
        status, exit_code = "aborted", EXIT_ABORTED
    elif (
        all_workers_complete
        and coverage["complete"]
        and not invalid_sources
        and not missing_conditions
        and not incomplete_conditions
    ):
        status, exit_code = "complete", EXIT_COMPLETE
    elif published:
        status, exit_code = "partial", EXIT_PARTIAL
    else:
        status, exit_code = "failed", EXIT_FAILED

    manifest = {
        "schema_version": 2,
        "collected_at": timestamp(),
        "status": status,
        "collector_exit_code": exit_code,
        "experiment": submission.get("experiment", "unknown"),
        "model": submission.get("model", {}),
        "configuration": configuration,
        "task_mapping": submission.get("task_mapping", {}),
        "resources": submission.get("resources", {}),
        "git_revision": submission.get("git_revision", "unknown"),
        "slurm": {
            **submission.get("slurm", {}),
            "array_job_id": parent_job_id,
            "array_state": parent_state,
            "collector_job_id": int(collector_id) if collector_id.isdigit() else collector_id,
            "collector_state": collector_state,
        },
        "workers": {
            "expected": sorted(expected_workers),
            "successful": successful_workers,
            "failed": failed_workers,
            "missing": missing_workers,
            "unexpected": unexpected_workers,
            "statuses": statuses,
            "status_errors": status_errors,
        },
        "coverage": coverage,
        "results": published,
        "artifacts": {
            "slurm-logs.tar.gz": slurm_archive_details,
            "worker-logs.tar.gz": worker_archive_details,
        },
        "missing_conditions": missing_conditions,
        "incomplete_conditions": incomplete_conditions,
        "invalid_sources": invalid_sources,
    }
    atomic_json(run_directory / "manifest.json", manifest, collector_id)

    if status == "complete":
        shutil.rmtree(staging_directory, ignore_errors=True)
        for path, _ in slurm_members:
            path.unlink(missing_ok=True)
    return exit_code


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"collector fatal error: {error}", file=sys.stderr)
        raise SystemExit(EXIT_COLLECTION_ERROR)
