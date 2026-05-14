#!/bin/bash
#SBATCH --job-name=gp-active-sensing
#SBATCH --partition=normal             
#SBATCH --nodes=1
#SBATCH --ntasks=16                     
#SBATCH --cpus-per-task=1              
#SBATCH --mem=8G                       
#SBATCH --time=02:30:00                
#SBATCH --array=0-7%4                    
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err

set -e # Exit immediately if a command exits with a non-zero status

# --- Configuration ---
DATA_SRC=~/work/active-sensing/data
SCRIPT_PATH=~/work/active-sensing/train.py

# --- Validation Block ---
echo ">>> Running Validation for Array Job ID: $SLURM_ARRAY_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"

# Ensure critical Slurm variables are present
if [ -z "$SLURM_ARRAY_TASK_ID" ]; then
    echo "ERROR: SLURM_ARRAY_TASK_ID is undefined. Ensure this is run via sbatch --array."
    exit 1
fi

# Confirm resource allocation alignment
if [[ "$SLURM_CPUS_PER_TASK" -ne 1 ]]; then
    echo "WARNING: Unexpected CPU allocation ($SLURM_CPUS_PER_TASK). Performance may degrade."
fi

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Training script not found at $SCRIPT_PATH"
    exit 1
fi

if [ ! -d "$DATA_SRC" ]; then
    echo "ERROR: Source data directory $DATA_SRC does not exist."
    exit 1
fi

# --- Environment Initialization ---
source ~/miniforge3/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate torch_gpu

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Directory Setup ---
JOB_SCRATCH=~/scratch/active-sensing/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
DATA_DIR="$JOB_SCRATCH/data"
OUTPUT_DIR="$JOB_SCRATCH/outputs"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

cp -r "$DATA_SRC"/* "$DATA_DIR/"

# --- Execution ---
# Note: $TASKS_PER_JOB is interpolated by the main shell
# SLURM_PROCID and SEED are escaped to be evaluated within srun
# SLURM_NTASKS is derived automatically from #SBATCH --ntasks
# SLURM_PROCID is the rank of the process within the current job (0 to SLURM_NTASKS-1)
srun --exact -n $SLURM_NTASKS bash -c "
    SEED=\$(( SLURM_ARRAY_TASK_ID * SLURM_NTASKS + SLURM_PROCID ))
    echo \"[Job $SLURM_ARRAY_TASK_ID | Task \$SLURM_PROCID] Launching seed: \$SEED\"
    
    python $SCRIPT_PATH \\
        --seed \"\$SEED\" \\
        --glimpses 1 2 4 8 10 12 16 20 22 24 26 28 30 \\
        --patch_size 8 \\
        --data_dir \"$DATA_DIR\" \\
        --output_dir \"$OUTPUT_DIR\"
"

# --- Post-Processing ---
mkdir -p ~/work/active-sensing/results
cp "$OUTPUT_DIR"/*.csv ~/work/active-sensing/results/

# Clean up
rm -rf "$JOB_SCRATCH"
echo ">>> Job array task $SLURM_ARRAY_TASK_ID completed successfully."