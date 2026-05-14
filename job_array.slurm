#!/bin/bash
#SBATCH --job-name=gp-active-sensing
#SBATCH --partition=normal             
#SBATCH --nodes=1
#SBATCH --ntasks=8                     # Number of concurrent tasks per job
#SBATCH --cpus-per-task=1              # 1 logical thread per process
#SBATCH --mem=4G                       # Memory for the entire job (shared by tasks)
#SBATCH --time=02:00:00                
#SBATCH --array=1-4                    # Array of jobs (e.g., 4 concurrent jobs limit)
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err

# Config
TASKS_PER_JOB=8

# Environment initialization
source ~/miniforge3/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate torch_gpu

# Isolate resources per task (Slurm handles this via srun)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Define isolated directory in scratch 
JOB_SCRATCH=~/scratch/active-sensing/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
DATA_DIR="$JOB_SCRATCH/data"
OUTPUT_DIR="$JOB_SCRATCH/outputs"

mkdir -p "$DATA_DIR"
mkdir -p "$OUTPUT_DIR"

# Copy dataset once for all tasks in this job
cp -r ~/work/active-sensing/data/* "$DATA_DIR/"

# Execute job steps: evaluate the unique seed per process using srun
# SEED calculation: (ArrayID-1)*TasksPerJob + ProcID
srun --exact -n $TASKS_PER_JOB bash -c "
    SEED=\$(( (SLURM_ARRAY_TASK_ID - 1) * $TASKS_PER_JOB + SLURM_PROCID ))
    python ~/work/active-sensing/train.py \\
        --seed \"\$SEED\" \\
        --glimpses 1 2 4 8 10 12 16 20 \\
        --data_dir \"$DATA_DIR\" \\
        --output_dir \"$OUTPUT_DIR\"
"

# Copy out results
mkdir -p ~/work/active-sensing/results
cp "$OUTPUT_DIR"/*.csv ~/work/active-sensing/results/

# Clean up
rm -rf "$JOB_SCRATCH"
