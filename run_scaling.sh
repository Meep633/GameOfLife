#!/bin/bash
# Conway Game of Life — Scaling Study Submission Script
# Run from dcsfen01 front end. Submits one sbatch job per np value.
#
# Usage:
#   ./run_scaling.sh strong <boardFile>  <resultsDir> <steps> <binary>
#   ./run_scaling.sh weak   <boardsDir>  <resultsDir> <steps> <binary>
#
# binary: conway (MPI+CUDA) or conway-mpi (MPI only)
# Perfect-square np values: 1, 4, 9, 16, 25, 36
# Partition: dcs-2024 (6 GPUs per node)
# np layout: np nodes tasks/node gpus/node
#   1  -> 1 node,  1 task,  4 GPUs (full node for CPU cores)
#   4  -> 1 node,  4 tasks, 4 GPUs
#   9  -> 3 nodes, 3 tasks, 3 GPUs
#  16  -> 4 nodes, 4 tasks, 4 GPUs
#  25  -> 5 nodes, 5 tasks, 5 GPUs
#  36  -> 6 nodes, 6 tasks, 6 GPUs

MODE=$1
RESULTS_DIR=$3
STEPS=$4
BINARY=$5

mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/submission_log.txt"

echo "=== Conway Scaling Study: $MODE ===" | tee "$LOG"
echo "Binary: $BINARY | Steps: $STEPS" | tee -a "$LOG"
echo "" | tee -a "$LOG"

# submit_one <np> <nodes> <tasks_per_node> <gpus_per_node> <inputFile>
submit_one() {
  local NP=$1 NODES=$2 TPR=$3 GPUS=$4 INPUT=$5
  echo "Submitting np=$NP | ${NODES}N x ${TPR}T x ${GPUS}GPU" | tee -a "$LOG"
  sbatch \
    --job-name=conway_${MODE}_${BINARY}_np${NP} \
    --partition=dcs-2024 \
    --nodes=$NODES \
    --ntasks-per-node=$TPR \
    --gpus-per-node=$GPUS \
    --gres=gpu:${GPUS} \
    --time=00:30:00 \
    --chdir="$PWD" \
    -o "$RESULTS_DIR/np${NP}_%j.log" \
    run_benchmark_job.sh "$NP" "$INPUT" "$RESULTS_DIR" "$STEPS" "$BINARY" \
    2>&1 | tee -a "$LOG"
}

if [[ "$MODE" == "strong" ]]; then
  INPUT=$2
  submit_one  1  1  1  4 "$INPUT"
  submit_one  4  1  4  4 "$INPUT"
  submit_one  9  3  3  3 "$INPUT"
  submit_one 16  4  4  4 "$INPUT"
  submit_one 25  5  5  5 "$INPUT"
  submit_one 36  6  6  6 "$INPUT"

elif [[ "$MODE" == "weak" ]]; then
  BOARDS_DIR=$2
  submit_one  1  1  1  4 "$BOARDS_DIR/board_1.bin"
  submit_one  4  1  4  4 "$BOARDS_DIR/board_4.bin"
  submit_one  9  3  3  3 "$BOARDS_DIR/board_9.bin"
  submit_one 16  4  4  4 "$BOARDS_DIR/board_16.bin"
  submit_one 25  5  5  5 "$BOARDS_DIR/board_25.bin"
  submit_one 36  6  6  6 "$BOARDS_DIR/board_36.bin"

else
  echo "Unknown mode. Use: strong | weak"
  exit 1
fi

echo "" | tee -a "$LOG"
echo "Monitor:  squeue -u \$USER" | tee -a "$LOG"
echo "Merge:    cat $RESULTS_DIR/np_*/results_np*.csv > $RESULTS_DIR/all_results.csv" | tee -a "$LOG"
