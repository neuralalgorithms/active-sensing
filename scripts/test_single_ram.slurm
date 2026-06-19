#!/bin/bash
#SBATCH --job-name=ram-test-single
#SBATCH --partition=normal             
#SBATCH --nodes=1
#SBATCH --ntasks=1                     
#SBATCH --cpus-per-task=1              
#SBATCH --mem=8G                       
#SBATCH --time=01:00:00                
#SBATCH --output=logs/ram_test_%j.out
#SBATCH --error=logs/ram_test_%j.err

set -e

# --- Configuration ---
DATA_SRC=~/work/active-sensing/data
SCRIPT_PATH=~/work/active-sensing/train_ram.py

echo ">>> Running single test for RAM Model"

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
JOB_SCRATCH=~/scratch/active-sensing/test_${SLURM_JOB_ID}
DATA_DIR="$JOB_SCRATCH/data"
OUTPUT_DIR="$JOB_SCRATCH/outputs"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
cp -r "$DATA_SRC"/* "$DATA_DIR/"

# --- Execution ---
srun --exact -n 1 bash -c "
    python $SCRIPT_PATH \
        --seed 1 \
        --glimpses 5 \
        --patch_size 8 \
        --data_dir \"$DATA_DIR\" \
        --output_dir \"$OUTPUT_DIR\"
"

# --- Post-Processing ---
# Save the results to a different folder as requested
RESULTS_DIR=~/work/active-sensing/test_results
mkdir -p "$RESULTS_DIR"
cp "$OUTPUT_DIR"/*.csv "$RESULTS_DIR/"

rm -rf "$JOB_SCRATCH"
echo ">>> Test job $SLURM_JOB_ID completed successfully. Results saved to $RESULTS_DIR"
