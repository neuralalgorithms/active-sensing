# --- Experiment Identity ---
EXPERIMENT_NAME = "ram-grid-sweep"
NOTE = "Noise attempt 4. Fixed: ram_v2, p=4. noise: 0.0, 0.02, 0.03"
ENTRYPOINT = "../train_ram.py"
RESULTS_ROOT = "../results"
PYTHON = "/home/ssz220001/miniforge3/envs/torch_gpu/bin/python"
CWD = ".."

# --- Slurm Resource Allocation ---
PARTITION = "normal"
TIME = "00:30:00"
MEMORY = "10G" # This is per array task. ~500MB per RAM workerB
WORKERS_PER_TASK = 16
CPUS_PER_WORKER = 1
MAX_ACTIVE_TASKS = 4

# Set DRY_RUN = True to test planning without submitting to Slurm
DRY_RUN = False
VERBOSE_DRY_RUN = False

# --- Seeds ---
SEEDS = range(32)

# --- Sweep Parameters ---
# Sequences define sweep axes; scalars pass fixed parameters to train_ram.py
MODEL_TYPE = ("random", "policy")       # or ("policy", "random") to sweep both
PATCH_SIZE = 4
GLIMPSES = (1, 2, 4, 8, 16, 24, 32)
HIDDEN_DIM = 64
SENSOR_NOISE = (0.0, 0.02, 0.03)
EPOCHS = 400
LR = 0.001
LR_SCHEDULE = "step_decay"

# --- Declared Artifacts to Collect & Validate ---
CSV_ARTIFACTS = ("results.csv",)
OPAQUE_ARTIFACTS = ("model.safetensors",)
