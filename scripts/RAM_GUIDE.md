# RAM Job Submission Guide

This guide documents how to configure, dry-run, submit, and collect Recurrent Attention Model (RAM) workflows using `slpilot`.

`slpilot` orchestrates Slurm compute arrays for parallel training tasks, isolates worker executions, merges condition metrics, and publishes structured artifacts with run manifests.

---

## 1. Quick Start

Workflows are defined as declarative Python configuration modules containing uppercase constants.

1. **Create your configuration module** (e.g. `ram_sweep.py`):
   ```bash
   cp examples/ram_sweep.py ram_sweep.py
   ```
2. **Perform a dry run to inspect the task plan**:
   Ensure `DRY_RUN = True` in your configuration, then execute:
   ```bash
   slpilot submit ram_sweep.py
   # Or via python module syntax:
   python3 -m slpilot submit ram_sweep.py
   ```
3. **Submit the workflow to Slurm**:
   Set `DRY_RUN = False` in your configuration and run:
   ```bash
   slpilot submit ram_sweep.py
   ```

Upon submission, `slpilot` validates the configuration, computes the Cartesian parameter grid, writes task plans, submits a held compute array job, submits a dependent `afterany` collector job, and releases the array.

Configurations can be kept untracked locally if desired. The resolved configuration and parameter specifications are recorded in each run's `submission.json`.

---

## 2. Configuration Settings

Configurations are Python files (`.py`) containing uppercase constants:

### Core Engine & Slurm Settings
- `EXPERIMENT_NAME`: String identifying the run (e.g., `"ram-grid-sweep"`). Used for Slurm job naming (`s-pilot-<EXPERIMENT_NAME>`) and directory naming.
- `ENTRYPOINT`: Path to the Python training script (e.g., `"train_ram.py"`).
- `PYTHON`: (Optional) Path to the Python interpreter binary. Defaults to `sys.executable`.
- `SEEDS`: Range, list, or tuple of non-negative integer seeds (e.g., `range(64)` or `[0, 1, 2]`). Seeds form the innermost axis of the Cartesian product.
- `PARTITION`: Slurm queue/partition name (e.g., `"normal"`).
- `TIME`: Wall-clock time limit per array task node (e.g., `"02:00:00"`).
- `MEMORY`: Total memory allocated per array task node (e.g., `"16G"`). This memory budget is shared across all concurrent workers running on the node.
- `WORKERS_PER_TASK`: Number of concurrent worker processes packed into each Slurm array task node.
- `CPUS_PER_WORKER`: Number of CPU cores allocated to each worker process. In the compute wrapper, `OMP_NUM_THREADS` and `MKL_NUM_THREADS` are automatically set to this value.
- `MAX_ACTIVE_TASKS`: Maximum number of concurrently running Slurm array tasks (sets Slurm array concurrency throttling: `--array=0-N%<MAX_ACTIVE_TASKS>`).
- `RESULTS_ROOT`: Base output directory (e.g., `"results"`).
- `DRY_RUN`: Boolean. When `True`, validates configuration and outputs the task distribution without submitting Slurm jobs.
- `VERBOSE_DRY_RUN`: Boolean. When `True` (with `DRY_RUN = True`), outputs full condition lists and per-task slot mappings.
- `NOTE`: (Optional) String description recorded in `submission.json` and `manifest.json`.

### Artifacts & Data Staging
- `CSV_ARTIFACTS`: Tuple of relative CSV filenames emitted by workers to merge per condition (e.g., `("results.csv",)`).
- `OPAQUE_ARTIFACTS`: Tuple of relative file paths for model checkpoints or raw binary assets (e.g., `("model.safetensors",)`). These are preserved byte-for-byte and hashed with SHA-256.
- `DATA_SOURCE`: (Optional) Directory path containing input datasets. When specified, `slpilot` copies this to node-local scratch (`$SLURM_TMPDIR` or `/tmp`) before workers launch.
- `DATA_FLAG`: (Optional) Flag used to pass the local staged data directory to the entrypoint. Defaults to `"--data_dir"`.
- `SEED_FLAG`: (Optional) Flag used to pass the seed to the entrypoint. Defaults to `"--seed"`.
- `OUTPUT_FLAG`: (Optional) Flag used to pass the per-task output directory to the entrypoint. Defaults to `"--output_dir"`.

### Parameter Mapping Rules
Any uppercase variable that is not an engine control variable is passed to the entrypoint as a command-line argument:
- The argument flag name is `--` followed by the variable name in lowercase (e.g., `MODEL_TYPE` becomes `--model_type`, `PATCH_SIZE` becomes `--patch_size`, `GLIMPSES` becomes `--glimpses`).
- **Fixed Parameters (Scalars)**: A single value (`int`, `float`, `str`, `bool`) is passed as a fixed argument across all tasks.
- **Sweep Axes (Sequences)**: A `list`, `tuple`, or `range` defines a sweep axis. `slpilot` computes the full Cartesian product across all sweep axes.

---

## 3. Example RAM Configuration (`ram_sweep.py`)

```python
"""RAM sweep configuration for slpilot."""

# Experiment Identity & Entrypoint
EXPERIMENT_NAME = "ram-grid-sweep"
NOTE = "RAM policy and random models across glimpses and seeds"
ENTRYPOINT = "train_ram.py"
PYTHON = "/path/to/conda/envs/torch_env/bin/python"

# Seeds
SEEDS = range(64)

# Slurm Resource Allocations
PARTITION = "normal"
TIME = "02:00:00"
MEMORY = "16G"          # Total memory allocated per array node
WORKERS_PER_TASK = 4    # Concurrent workers packed per node
CPUS_PER_WORKER = 2     # CPUs per worker (sets OMP/MKL thread count to 2)
MAX_ACTIVE_TASKS = 8    # Maximum concurrently active array tasks
RESULTS_ROOT = "results"

# Dry Run Settings
DRY_RUN = True
VERBOSE_DRY_RUN = False

# Sweep Axes (Sequences create Cartesian grid; scalars remain fixed)
MODEL_TYPE = ("policy", "random")   # Evaluates both policy and random architectures
GLIMPSES = (1, 2, 3, 4, 5, 6, 7)    # Sequence lengths
PATCH_SIZE = 8                      # Fixed patch dimension

# Published Artifacts
CSV_ARTIFACTS = ("results.csv",)
OPAQUE_ARTIFACTS = ("model.safetensors",)
```

---

## 4. Array Packing & Slurm Execution

- **Total Tasks**:
  $$\text{Total Tasks} = |\text{SEEDS}| \times \prod_{a \in \text{axes}} |a|$$
  *(For 64 seeds $\times$ 7 glimpse counts $\times$ 2 model types, total tasks = 896).*
- **Array Tasks**:
  $$\text{Array Tasks} = \left\lceil \frac{\text{Total Tasks}}{\text{WORKERS\_PER\_TASK}} \right\rceil$$
- **Array Specifier**: `--array=0-(Array Tasks - 1)%MAX_ACTIVE_TASKS`.
- **Process Management**:
  Each array task uses `srun` step allocation to execute `WORKERS_PER_TASK` processes in parallel:
  ```bash
  srun --ntasks="${S_PILOT_WORKERS_PER_ARRAY_TASK}" \
       --cpus-per-task="${S_PILOT_CPUS_PER_WORKER}" \
       --kill-on-bad-exit=0 \
       "${S_PILOT_PYTHON}" -m slpilot.worker --run-dir "$RUN_DIR"
  ```
- **Task Indexing**:
  $$\text{global\_index} = \text{SLURM\_ARRAY\_TASK\_ID} \times \text{WORKERS\_PER\_TASK} + \text{SLURM\_PROCID}$$
- **Tail Slot Handling**:
  Unused slots in the final array task do not execute redundant training; worker ranks where `global_index >= Total Tasks` exit immediately with status 0.

---

## 5. Dry Run Mode

When `DRY_RUN = True`, `slpilot submit` validates all paths and types, builds the Cartesian plan, and prints a summary:

```text
Experiment: ram-grid-sweep
Note: RAM policy and random models across glimpses and seeds
Conditions: 14
Seeds: range(0, 64) (64 seeds)
Total tasks: 896
Array tasks: 224
Array: --array=0-223%8
Workers per array task: 4
Idle tail slots: 0
```

Setting `VERBOSE_DRY_RUN = True` additionally displays:
- Every condition identifier and its parameter assignments:
  ```text
  Conditions:
    condition-000000-a1b2c3d4e5f6: model_type='policy', glimpses=1, patch_size=8
    condition-000001-f7e8d9c0b1a2: model_type='policy', glimpses=2, patch_size=8
    ...
  ```
- The exact task-to-slot mapping across array indices:
  ```text
  Task mapping:
    array=0 slot=0 task-000000 condition=condition-000000-a1b2c3d4e5f6 seed=0
    array=0 slot=1 task-000001 condition=condition-000000-a1b2c3d4e5f6 seed=1
    ...
  ```

---

## 6. Verification and Output Structure

Completed runs are saved under:
`<RESULTS_ROOT>/<YYYY-MM>/<JOB_ID>__<EXPERIMENT_NAME>/`

### Verifying Run Completion
Inspect `manifest.json` in the run directory:
```bash
python3 -m json.tool "manifest.json"
```

A `manifest["status"] == "complete"` indicates:
- Every planned worker completed with exit code 0.
- No tasks were missing, failed, or unexpected.
- Every condition CSV was merged across seeds and compressed to `.csv.gz`.
- Every declared opaque artifact was verified and registered with its SHA-256 digest.
- `worker-logs.tar.gz` and `slurm-logs.tar.gz` were archived.
- The `.staging/` working directory was removed.

### Directory Hierarchy
```text
<JOB_ID>__<EXPERIMENT_NAME>/
├── submission.json          # Resolved config, task plan summary, submission timestamp
├── task-plan.json           # Deterministic Cartesian task mapping
├── manifest.json            # Final status, coverage report, artifact digests
├── compute.sh               # Generated Slurm compute wrapper
├── collect.sh               # Generated Slurm collector wrapper
├── worker-logs.tar.gz       # Archive of per-task stdout and stderr
├── slurm-logs.tar.gz        # Archive of Slurm compute stdout and stderr
└── conditions/
    └── <CONDITION_ID>/
        ├── results.csv.gz   # Merged condition metrics across all seeds
        └── tasks/
            └── <TASK_ID>/
                └── model.safetensors  # Per-task opaque checkpoint
```

### Merged CSV Columns
Merged CSV files (`conditions/<condition_id>/results.csv.gz`) contain the original CSV columns with engine tracking fields added to each row:
- `s_pilot_task_id`: Generic task identifier (e.g. `task-000004`).
- `s_pilot_condition_id`: Condition identifier (e.g. `condition-000000-a1b2c3d4e5f6`).
- `s_pilot_seed`: Seed for that task run.
- `s_pilot_<parameter>`: Swept and fixed parameters (e.g. `s_pilot_model_type`, `s_pilot_glimpses`, `s_pilot_patch_size`).

To inspect unique seeds in a merged condition CSV:
```bash
python3 -c 'import pandas as pd, sys; print(sorted(pd.read_csv(sys.argv[1])["s_pilot_seed"].unique()))' \
  conditions/condition-000000-*/results.csv.gz
```
Condition seed coverage is also summarized directly in `manifest.json` under the `"conditions"` key (`expected_seeds`, `observed_seeds`, `missing_seeds`, `complete`).

---

## 7. Cancelling a Run

When a workflow is submitted, `slpilot` prints the compute array ID and collector job ID:
```text
Compute array: 123456 (0-223%8)
Collector job: 123457 (afterany:123456)
Run directory: results/2026-09/123456__ram-grid-sweep
```

To cancel the workflow:
```bash
scancel 123456
```

The dependent collector job runs immediately after cancellation. It detects the `CANCELLED` state, marks `manifest.json` as `"status": "aborted"` with collector exit code `20`, archives available logs, and preserves `.staging/` for diagnosis.

---

## 8. Manual Collection

If a collector job must be executed manually or re-run on local data:

Run the generated collection wrapper directly inside the run directory:
```bash
cd results/2026-09/123456__ram-grid-sweep
./collect.sh
```

Alternatively, invoke the collector module directly:
```bash
python3 -m slpilot.collect --run-dir results/2026-09/123456__ram-grid-sweep
```

---

## 9. Current Specifications & Limitations

- **Dry Run Output**: Displays high-level metadata (total tasks, array tasks, idle tail slots) and per-task slot mappings when verbose. It does not output an aggregated ASCII grid table grouped by parameter condition.
- **Domain-Agnostic Validation**: `slpilot` validates that CSV headers match across all tasks and adds engine metadata columns. It does not perform domain-specific row filtering or model-specific sanity checks (such as verifying epoch sequences or checking specific metric ranges inside CSVs).
- **Condition Directory Structure**: Merged artifacts are organized into structured directories by condition (`conditions/<condition_id>/<csv_name>.csv.gz`) rather than using flat filename parameter strings.
- **Manifest Querying**: Manifest inspection is done by reading `manifest.json` directly or using Python. An interactive `slpilot query` CLI subcommand is reserved for a future release.
- **Wrapper Generation**: Each run generates its own local `./collect.sh` script inside the run directory; there is no global collection bash script in the repository root.
