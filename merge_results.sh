#!/bin/bash
# ============================================================
# Merge per-job CSVs into one all_results.csv per experiment.
# Run after all sbatch jobs have completed.
# ============================================================

SCRATCH=/gpfs/u/home/PCPG/PCPGnhnx/scratch
RESULTS="$SCRATCH/results"

for exp in strong_cuda strong_mpi weak_cuda weak_mpi; do
  OUT="$RESULTS/$exp/all_results.csv"
  PARTS=("$RESULTS/$exp"/np_*/results_np*.csv)

  if [[ ! -e "${PARTS[0]}" ]]; then
    echo "WARNING: No result CSVs found for $exp — jobs may still be running"
    continue
  fi

  # Write header once, then append data rows from all files
  head -1 "${PARTS[0]}" > "$OUT"
  for f in "${PARTS[@]}"; do
    tail -n +2 "$f" >> "$OUT"
  done

  ROWS=$(tail -n +2 "$OUT" | wc -l)
  echo "Merged $exp → $OUT  ($ROWS data rows)"
done

echo ""
echo "Done. CSVs ready for generate_graphs.py"
