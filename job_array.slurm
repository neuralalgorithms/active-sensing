#!/bin/bash
#SBATCH --job-name=gp-active-sensing
#SBATCH --partition=normal             # Target CPU nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4              # for num_workers
#SBATCH --mem=4G
#SBATCH --time=02:00:00                
#SBATCH --array=1-20                   # 20 parallel seeds
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err

# env initialization
source ~/miniforge3/etc/profile.d/conda.sh
eval "$(mamba shell hook --shell bash)"
mamba activate torch_gpu

# isolate resources on AMD EPYC nodes
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# define isolated directory in scratch/
JOB_SCRATCH=~/scratch/active-sensing/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$JOB_SCRATCH/data"
mkdir -p "$JOB_SCRATCH/outputs"

# map env vars used in Python code
export SLURM_TMPDIR="$JOB_SCRATCH/data"
export OUTPUT_DIR="$JOB_SCRATCH/outputs"

# copy in the dataset from work to scratch for fast storage
cp -r ~/work/active-sensing/data/* "$SLURM_TMPDIR/"

# execution
cd "$JOB_SCRATCH"
python ~/work/active-sensing/train.py \
    --seed "$SLURM_ARRAY_TASK_ID" \
    --glimpses 1 2 4 8 10 12 16 20 \
    --results_file "results_seed_${SLURM_ARRAY_TASK_ID}.csv"

# copy out results from scratch/ to work/
mkdir -p ~/work/active-sensing/results
cp "$OUTPUT_DIR"/*.csv ~/work/active-sensing/results/

# clean up
rm -rf "$JOB_SCRATCH"