#!/bin/bash
set -e

# --- Configuration ---
DATA_SRC="./data"
SCRIPT_PATH="./train.py"
RESULTS_DIR="./results"
LOCAL_SCRATCH="./scratch"
OUTPUT_DIR="$LOCAL_SCRATCH/outputs"

mkdir -p "$RESULTS_DIR" "$OUTPUT_DIR"

# --- Threading & Hardware Optimization ---
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

CONCURRENT_JOBS=2

QUEUE_FILE=$(mktemp)
RUNNER_SCRIPT=$(mktemp)
chmod +x "$RUNNER_SCRIPT"

# Generate a sequence of 40 glimpses (1 through 40). 
# Modify this list if you require a non-linear distribution (e.g., 2 4 8 16...).
GLIMPSES=$(seq 1 40)

# --- Task Runner Sub-script ---
cat << 'EOF' > "$RUNNER_SCRIPT"
#!/bin/bash
SEED=$1
SCRIPT_PATH=$2
DATA_SRC=$3
OUTPUT_DIR=$4
RESULTS_DIR=$5
shift 5
GLIMPSES="$@"

TASK_OUT="$OUTPUT_DIR/seed_${SEED}"
mkdir -p "$TASK_OUT"

echo ">>> Launching seed: $SEED"

# Execute training; unbuffered python output is assumed
if uv run python -u "$SCRIPT_PATH" \
    --seed "$SEED" \
    --glimpses $GLIMPSES \
    --patch_size 8 \
    --data_dir "$DATA_SRC" \
    --output_dir "$TASK_OUT"; then
    
    cp "$TASK_OUT"/*.csv "$RESULTS_DIR/" 2>/dev/null || true
else
    echo "ERROR: Task failed for seed $SEED" >&2
fi
EOF

# --- Queue Generation ---
echo ">>> Populating task queue for 64 seeds..."

# Generate seeds 0 through 63
for SEED in {0..63}; do
    echo "$SEED" >> "$QUEUE_FILE"
done

echo ">>> Starting Processing Pool (Max Concurrent Jobs: $CONCURRENT_JOBS)..."

# Execute the queue
xargs -a "$QUEUE_FILE" -P "$CONCURRENT_JOBS" -I {} "$RUNNER_SCRIPT" {} "$SCRIPT_PATH" "$DATA_SRC" "$OUTPUT_DIR" "$RESULTS_DIR" $GLIMPSES

# --- Cleanup ---
rm -f "$QUEUE_FILE" "$RUNNER_SCRIPT"
rm -rf "$LOCAL_SCRATCH"

echo ">>> All experiments completed. Results aggregated in $RESULTS_DIR"

