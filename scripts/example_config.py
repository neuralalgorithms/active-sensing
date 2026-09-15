"""slpilot example configuration file.

This file demonstrates all available configuration options for slpilot, explaining
how each setting maps to Slurm arguments, worker processes, file structures,
and command-line flags.

A sweep configuration is a standard Python module (.py) defining uppercase constants.
To validate and submit this sweep, run:
    slpilot submit example_config.py
"""

from pathlib import Path

# ==============================================================================
# 1. EXPERIMENT IDENTITY & METADATA
# ==============================================================================

# Unique name for the experiment.
# Allowed characters: letters (A-Z, a-z), numbers (0-9), dots (.), underscores (_), and dashes (-).
# Used to name Slurm jobs ("s-pilot-<EXPERIMENT_NAME>") and the results subdirectory
# ("<RESULTS_ROOT>/<YYYY-MM>/<JOB_ID>__<EXPERIMENT_NAME>").
EXPERIMENT_NAME = "example_sweep"

# Optional human-readable description or note describing the experiment.
# Stored in submission.json and manifest.json for provenance and tracking.
NOTE = "Comprehensive example showing all slpilot configuration options."


# ==============================================================================
# 2. ENTRYPOINT, PYTHON ENVIRONMENT & WORKING DIRECTORY
# ==============================================================================

# Path to the Python training or benchmark script to execute for each task.
# Can be absolute or relative (relative paths resolve against this config file's directory).
# Must be an existing file.
ENTRYPOINT = "examples/train_example.py"

# Optional path to the Python interpreter executable used to run both the slpilot
# worker wrapper and the entrypoint script.
# Default: sys.executable (the Python environment from which `slpilot submit` is called).
# Useful when submitting jobs from a login node while targeting a specific virtualenv/conda env.
# PYTHON = "/path/to/conda/envs/myenv/bin/python"

# Optional working directory (CWD) from which the entrypoint script is executed.
# Default: directory containing this config file.
# Relative paths resolve against this config file's directory.
# CWD = "."

# Optional list or tuple of fixed string arguments passed to the entrypoint script
# before any sweep parameters.
# Example: FIXED_ARGS = ["--verbose", "--disable-telemetry"]
FIXED_ARGS = []


# ==============================================================================
# 3. REPRODUCIBILITY & SEEDS
# ==============================================================================

# Seeds to run for every condition (hyperparameter combination).
# Must be a non-empty range, list, or tuple of unique, non-negative integers.
# Seeds form the innermost (minor) axis of the Cartesian product: for each condition,
# all seeds are executed (e.g. condition 0 with seeds 0,1,2; condition 1 with seeds 0,1,2).
#
# Examples:
#   SEEDS = range(5)        # Runs seeds 0, 1, 2, 3, 4
#   SEEDS = [101, 202, 303] # Runs custom seed values
SEEDS = range(3)


# ==============================================================================
# 4. SLURM CLUSTER RESOURCE SPECIFICATIONS
# ==============================================================================

# Slurm partition (queue) to submit array and collector jobs to.
# Equivalent to: #SBATCH --partition=<PARTITION> (or -p <PARTITION>)
PARTITION = "normal"

# Slurm walltime limit per array task.
# Formatted as standard Slurm time (e.g., "HH:MM:SS", "D-HH:MM", or "MM").
# Equivalent to: #SBATCH --time=<TIME> (or -t <TIME>)
# Note: The lightweight post-run collector job uses a separate hardcoded limit of 10 minutes.
TIME = "01:00:00"

# ------------------------------------------------------------------------------
# MEMORY EXPLANATION & SLURM BEHAVIOR:
#
# What is MEMORY?
# `MEMORY` specifies the TOTAL memory allocated per Slurm array task (i.e., per
# allocated node / step).
#
# Is it per task or per worker?
# In slpilot, each Slurm array task runs one compute node hosting WORKERS_PER_TASK
# worker processes. `MEMORY` is the total memory budget for the entire array task.
# Therefore, all WORKERS_PER_TASK running concurrently on that node share this pool.
# For example, if MEMORY = "16G" and WORKERS_PER_TASK = 4, each worker has on average
# 4GB of RAM available.
#
# Standard Slurm Script Equivalent:
# In a standard sbatch script, this corresponds directly to:
#     #SBATCH --mem=<MEMORY>
#
# Other Slurm memory options NOT currently available in slpilot .py configs:
# - `--mem-per-cpu=<size>`: Allocates memory per allocated CPU core instead of per node.
#   (slpilot currently uses `--mem` for the entire node allocation).
# - `--mem-per-gpu=<size>`: Allocates memory per allocated GPU.
# - `--mem-bind=...`: Controls NUMA / memory affinity bindings across CPU sockets.
# ------------------------------------------------------------------------------
MEMORY = "16G"

# Number of concurrent worker processes packed into a single Slurm array task (node).
# Must be a positive integer (> 0).
# In Slurm, this sets `#SBATCH --ntasks=<WORKERS_PER_TASK>`.
# Inside the array task, `srun --ntasks=<WORKERS_PER_TASK>` spawns these workers in parallel.
# If your sweep has 60 total tasks (e.g., 20 conditions * 3 seeds) and WORKERS_PER_TASK = 4,
# slpilot will create ceil(60 / 4) = 15 Slurm array tasks.
WORKERS_PER_TASK = 2

# Number of CPU cores allocated to each individual worker process.
# Must be a positive integer (> 0).
# In Slurm, this sets `#SBATCH --cpus-per-task=<CPUS_PER_WORKER>` and passes
# `--cpus-per-task` to `srun`.
# Additionally, slpilot automatically exports:
#     OMP_NUM_THREADS = CPUS_PER_WORKER
#     MKL_NUM_THREADS = CPUS_PER_WORKER
# to prevent multi-threaded libraries (e.g. PyTorch, OpenBLAS, MKL) from oversubscribing CPUs.
CPUS_PER_WORKER = 2

# Maximum number of Slurm array tasks (nodes/jobs) allowed to run simultaneously.
# Must be a positive integer (> 0).
# Controls Slurm array concurrency throttling via the `%` specifier:
# Equivalent to: #SBATCH --array=0-(N-1)%<MAX_ACTIVE_TASKS>
# This prevents your sweep from monopolizing cluster nodes or exceeding user job quotas.
MAX_ACTIVE_TASKS = 4


# ==============================================================================
# 5. STORAGE & RESULTS DIRECTORY
# ==============================================================================

# Root directory where experiment outputs, staging files, manifests, and merged
# artifacts will be stored.
# Relative paths resolve against this config file's directory.
#
# Output directory hierarchy created by slpilot:
# <RESULTS_ROOT>/
#   <YYYY-MM>/
#     <JOB_ID>__<EXPERIMENT_NAME>/
#       submission.json          (Submission metadata, hashes, resolved config)
#       task-plan.json           (Full Cartesian product task mapping)
#       manifest.json            (Final status, worker execution, artifact registry)
#       compute.sh               (Generated compute wrapper script)
#       collect.sh               (Generated afterany collector wrapper script)
#       slurm-logs.tar.gz        (Archived Slurm stdout/stderr logs)
#       worker-logs.tar.gz       (Archived per-task stdout/stderr logs)
#       conditions/
#         <CONDITION_ID>/
#           <CSV_NAME>.csv.gz    (Merged, compressed CSV across all seeds)
#           tasks/
#             <TASK_ID>/
#               <OPAQUE_FILE>    (Published opaque artifacts)
RESULTS_ROOT = "results"


# ==============================================================================
# 6. DRY RUN & VERBOSITY
# ==============================================================================

# If True, `slpilot submit` will validate the configuration, compute the Cartesian
# product and task mapping, and display a summary plan without submitting to Slurm.
DRY_RUN = True

# If True (when DRY_RUN is True), displays a full verbose breakdown of every condition
# and every individual task mapping (array task index, slot, condition ID, parameters, seed).
VERBOSE_DRY_RUN = False


# ==============================================================================
# 7. DATA STAGING & CLI FLAGS
# ==============================================================================

# Optional path to a directory containing datasets or input assets.
# When specified:
# 1. The compute wrapper copies this directory to fast node-local scratch ($SLURM_TMPDIR or /tmp)
#    before workers start.
# 2. The staged path is passed to your entrypoint script using `DATA_FLAG`.
# 3. Scratch is automatically cleaned up when the array task exits.
# DATA_SOURCE = "data/imagenet"

# Command-line flag used to pass the staged data directory path to your entrypoint.
# Must start with "--" and contain no whitespace.
# Default: "--data_dir"
# DATA_FLAG = "--data_dir"

# Command-line flag used to pass the random seed to your entrypoint script.
# Must start with "--" and contain no whitespace.
# Default: "--seed"
SEED_FLAG = "--seed"

# Command-line flag used to pass the isolated per-task output directory to your entrypoint.
# Each task writes its outputs to this directory.
# Must start with "--" and contain no whitespace.
# Default: "--output_dir"
OUTPUT_FLAG = "--output_dir"


# ==============================================================================
# 8. ARTIFACT DECLARATIONS
# ==============================================================================
# At least one artifact (CSV or Opaque) is required.
# Paths must be relative and safe (cannot escape task directory with "..").
# File stems across all artifacts must be unique.

# CSV artifacts written by each task inside its output directory.
# During collection:
# 1. Missing CSVs on successful tasks are detected and flagged.
# 2. CSV headers are validated across all tasks.
# 3. Columns with the reserved prefix `s_pilot_` are rejected in source CSVs.
# 4. Engine columns are prepended to each row:
#      `s_pilot_task_id`, `s_pilot_condition_id`, `s_pilot_seed`, `s_pilot_<axis_name>`...
# 5. All seed runs for each condition are merged and saved as gzip-compressed CSV:
#      `conditions/<condition_id>/<name>.csv.gz`
CSV_ARTIFACTS = (
    "metrics.csv",
)

# Opaque / binary artifacts written by each task inside its output directory
# (e.g., model checkpoints, weights, raw binary files, plots).
# During collection:
# 1. Copied into: `conditions/<condition_id>/tasks/<task_id>/<path>`
# 2. Preserved byte-for-byte (no decoding or modification).
# 3. SHA-256 digest and file size are computed and recorded in `manifest.json`.
OPAQUE_ARTIFACTS = (
    "checkpoint.bin",
)


# ==============================================================================
# 9. HYPERPARAMETERS & SWEEP PARAMETERS
# ==============================================================================
# Any uppercase variable not listed in the control variables above is treated as
# a script parameter.
#
# Rules:
# - Parameter name must match regex `^[A-Za-z][A-Za-z0-9_]*$` (cannot start with `S_PILOT_`).
# - The command-line flag passed to your entrypoint is `--` + lowercase variable name:
#     e.g., `WIDTH`         -> `--width <value>`
#     e.g., `LEARNING_RATE` -> `--learning_rate <value>`
#
# Sweep Axes (Sequences):
# - Defining a variable as a list, tuple, or range creates a SWEEP AXIS.
# - slpilot computes the full Cartesian product of all sweep axes in the order declared.
# - Each value must be a JSON scalar (str, int, float, bool, or None).
WIDTH = [32, 64, 128]
LEARNING_RATE = (1e-3, 1e-4)

# Fixed Parameters (Scalars):
# - Defining a variable as a scalar (str, int, float, bool, or None) sets a FIXED argument.
# - Passed to every task with its corresponding flag.
# BATCH_SIZE = 64
# OPTIMIZER = "adamw"
