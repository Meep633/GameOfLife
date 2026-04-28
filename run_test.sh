#!/bin/bash
# Quick single-job smoke test. Submits ONE sbatch job (np=1, 5 steps)
# and tails the result so you can confirm the script works before
# submitting all 24 scaling jobs.
#
# Usage: bash run_test.sh
# Run from: p/ directory on dcsfen01

SCRATCH=/gpfs/u/scratch/PCPG/PCPGnhnx
RESULTS="$SCRATCH/results/test_smoke"
mkdir -p "$RESULTS"

echo "Submitting smoke test: np=1, binary=conway, steps=5 ..."
JOBID=$(sbatch \
  --job-name=conway_smoke \
  --partition=dcs-2024 \
  --nodes=1 \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --gres=gpu:4 \
  --time=00:10:00 \
  --chdir="$PWD" \
  -o "$RESULTS/smoke_%j.log" \
  --parsable \
  run_benchmark_job.sh 1 "$SCRATCH/p/board_strong.bin" "$RESULTS" 5 conway)

echo "Submitted job $JOBID"
echo ""
echo "Watching queue (Ctrl+C to stop watching and check result manually)..."
while squeue -j "$JOBID" 2>/dev/null | grep -q "$JOBID"; do
  squeue -j "$JOBID"
  sleep 5
done

echo ""
echo "Job $JOBID finished. Result:"
cat "$RESULTS/np_1/timing.txt"
