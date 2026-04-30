#!/bin/bash
# Conway Game of Life — Scaling Study Submission Script
# Run from dcsfen01 front end. Submits one sbatch job per np value.
#
# Usage:
#   ./run_scaling.sh strong <boardFile>  <resultsDir> <steps> <binary>
#   ./run_scaling.sh weak   <boardsDir>  <resultsDir> <steps> <binary>
#
# binary: conway (MPI+CUDA) or conway-mpi (MPI only)
# Partition: dcs-2024 (6 GPUs per node)

MODE=$1
RESULTS_DIR=$3
STEPS=$4
BINARY=$5

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/submission_log.txt"

echo "=== Conway Scaling Study: $MODE ===" | tee "$LOG"
echo "Binary: $BINARY | Steps: $STEPS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# submit_one <np> <nodes> <tasks_per_node> <inputFile>
submit_one() {
  local NP=$1 NODES=$2 TPR=$3 INPUT=$4
  echo "Submitting np=$NP | ${NODES}N x ${TPR}T" | tee -a "$LOG"
  sbatch \
    --job-name=conway_${MODE}_${BINARY}_np${NP} \
    --partition=dcs-2024 \
    --nodes=$NODES \
    --ntasks-per-node=$TPR \
    --gpus-per-node=6 \
    --gres=gpu:6,nvme \
    --time=00:30:00 \
    --chdir="$PWD" \
    -o "$RESULTS_DIR/np${NP}_%j.log" \
    run_benchmark_job.sh "$NP" "$INPUT" "$RESULTS_DIR" "$STEPS" "$BINARY" \
    2>&1 | tee -a "$LOG"
}

if [[ "$MODE" == "strong" ]]; then
  INPUT=$2
  submit_one   1  1  1 "$INPUT"
  submit_one   4  1  4 "$INPUT"
  submit_one  16  1 16 "$INPUT"
  submit_one  25  1 25 "$INPUT"
  submit_one  64  2 32 "$INPUT"
  submit_one 100  4 25 "$INPUT"
  submit_one 256  8 32 "$INPUT"

elif [[ "$MODE" == "weak" ]]; then
  BOARDS_DIR=$2
  submit_one   1  1  1 "$BOARDS_DIR/board_100.bin"
  submit_one   4  1  4 "$BOARDS_DIR/board_200.bin"
  submit_one  16  1 16 "$BOARDS_DIR/board_400.bin"
  submit_one  64  2 32 "$BOARDS_DIR/board_800.bin"
  submit_one 256  8 32 "$BOARDS_DIR/board_1600.bin"

else
  echo "Unknown mode. Use: strong | weak"
  exit 1
fi

echo "" | tee -a "$LOG"
echo "Monitor:  squeue -u \$USER" | tee -a "$LOG"
echo "Merge:    cat $RESULTS_DIR/np_*/results_np*.csv > $RESULTS_DIR/all_results.csv" | tee -a "$LOG"
