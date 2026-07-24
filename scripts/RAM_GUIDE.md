# RAM Job Submission Guide

Submit recurrent attention model workflows using `submit_ram.py`. It orchestrates a Slurm compute array for training and a dependent collector job.

**Basic Usage:**
Edit the control panel variables at the top of the script, then run:
`python3 submit_ram.py`

**Key Nomenclature & Control Panel Variables:**
- `MODEL`: Model architecture type (`"policy"` or `"random"`).
- `SEED_COUNT`: Number of random seeds to evaluate per condition.
- `GLIMPSES`: Tuple of glimpse sequence lengths (e.g., `(1, 2, 3, 4, 5, 6, 7)`).
- `PATCH_SIZE`: Image patch pixel dimension.
- `WORKERS_PER_ARRAY_TASK`: Number of workers (models to train) packed into each array task.
- `CPUS_PER_WORKER`: Number of CPUs allocated to each worker task.
- `MAX_ACTIVE_ARRAY_TASKS`: Maximum concurrently running Slurm array tasks.
- `RESULTS_ROOT`: Base directory for experiment outputs and logs.
- `DRY_RUN`: Set to `True` to view exact task mapping and configurations without launching jobs.
- `VERBOSE_DRY_RUN`: Set to `True` (when `DRY_RUN = True`) to print the line-by-line task-to-worker mapping.
- **Slurm Configs:** Configure resource limits via `WALL_TIME`, `MEMORY_PER_ARRAY_TASK`, and `PARTITION`.

**Array Packing Calculation:**
- Total models: `SEED_COUNT * len(GLIMPSES)`
- Array task count: `ceil(Total / WORKERS_PER_ARRAY_TASK)`
- Array bounds: `--array=0-{Array task count - 1}%MAX_ACTIVE_ARRAY_TASKS`
- Internal `global_idx`: `SLURM_ARRAY_TASK_ID * WORKERS_PER_ARRAY_TASK + slot`
- **Tail Slot Handling:** Unused tail slots in the final array task do not waste compute; worker ranks beyond the total model count safely exit immediately with status 0.

**Dry Run Mode:**
If you set `DRY_RUN = True` in the control panel and run `python3 submit_ram.py`, the script outputs metadata details and a summary table mapping the array tasks to glimpse/seed configurations without submitting jobs to Slurm:

```text
Distribution Summary:
Array Tasks  | Glimpse | Seeds      | Models
--------------------------------------------
0..7         | 1       | 0..63      | 64    
8..15        | 2       | 0..63      | 64    
16..23       | 3       | 0..63      | 64    
...
```

- **Run Metadata Summary:** Prints total models, array task count, Slurm `--array` specification, workers per task, CPUs per worker, and idle tail slot count.
- **Detailed Task Mapping:** By default, the full line-by-line mapping list is hidden to keep the output concise. Set `VERBOSE_DRY_RUN = True` to output the detailed breakdown showing `task`, `slot`, `global_index`, `glimpse`, and `seed` for every model.


**Tips**
- **Single Model Type Per Run:** A single execution of `submit_ram.py` can evaluate multiple seeds and glimpse counts, but only supports 1 model type at a time (`MODEL = "policy"` or `MODEL = "random"`).
- **Simultaneous Comparisons:** If you wish to train both `"policy"` and `"random"` models concurrently, run two separate submission scripts (one for each model type). In this case, it is advised to set `MAX_ACTIVE_ARRAY_TASKS = 2` (or appropriately throttle resources) per workflow so they share cluster concurrency cleanly.

**Verification and Postprocessing prep**
After your job has finished, cd into the directory in results/month, and run:
`python3 -m json.tool "manifest.json"`
You should see manifest.status == "complete". This guarantees:
- Every expected (glimpse, seed) worker wrote a successful .status file with exit code 0.
- No expected workers are missing, failed, or unexpected.
- Every requested glimpse condition was published.
- Each condition contains exactly the expected seed set (0 through SEED_COUNT - 1).
- Source CSVs passed the collector’s identity checks: model type, patch size, glimpse, seed, and staging-directory label agree.
- No duplicate (model_type, patch_size, glimpses, seed, epoch) rows were found.
- Final condition files and both log archives were successfully published before manifest.json.

`python3 -c 'import pandas as pd; import sys; print(pd.read_csv(sys.argv[1]) ["seed"].unique())' "pxx_gxx.csv.gz"`
This reports how which seeds are contained in the combined CSV. Repeat for each of the outputted csv.gz.

**Cancelling a Job**
When submitting a job, the script prints the Compute array ID and the Collector job ID.
Use scancel compute_array_ID; the collector job will run soon after and exit on its own.
The manifest will report "aborted" and
`sacct -X -j COLLECTOR_JOB_ID --format=JobIDRaw,State,ExitCode` will show an error code of 20.
