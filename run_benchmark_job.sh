#!/bin/bash
# Inner benchmark job script — called by run_scaling.sh via sbatch.
# DO NOT run directly. sbatch sets --chdir to p/ so binaries are in $PWD.
#   $1 = np (number of MPI ranks)
#   $2 = inputFile (absolute path to .bin board)
#   $3 = resultsDir
#   $4 = steps
#   $5 = binary (conway or conway-mpi)

NP=$1
INPUT_FILE=$2
RESULTS_DIR=$3
STEPS=$4
BINARY=$5

# Source module system so 'module' works reliably inside sbatch
source /etc/profile.d/modules.sh
module load xl_r spectrum-mpi cuda

# --chdir in the sbatch command already puts us in the p/ directory.
# Do NOT cd elsewhere — that would break binary resolution.

# NVMe is only provisioned in sbatch jobs, not interactive salloc sessions.
# Fall back to GPFS tmp when NVMe isn't available.
NVME_DIR="/mnt/nvme/uid_$(id -u)/job_${SLURM_JOB_ID}"
if [[ ! -d "$NVME_DIR" ]]; then
  NVME_DIR="$(dirname "$RESULTS_DIR")/tmp_job_${SLURM_JOB_ID:-interactive}"
  mkdir -p "$NVME_DIR"
  echo "NOTE: NVMe not available, using GPFS fallback: $NVME_DIR"
fi
INPUT_BASENAME=$(basename "$INPUT_FILE")
cp "$INPUT_FILE" "$NVME_DIR/"

OUTDIR="$RESULTS_DIR/np_${NP}"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/timing.txt"

echo "=== np=$NP | binary=$BINARY | steps=$STEPS | job=$SLURM_JOB_ID ===" | tee "$OUTFILE"
echo "Nodes=$SLURM_JOB_NUM_NODES | Tasks/Node=$SLURM_NTASKS_PER_NODE"      | tee -a "$OUTFILE"
echo "Script dir: $PWD"                                                      | tee -a "$OUTFILE"
echo "Input: $NVME_DIR/$INPUT_BASENAME"                                      | tee -a "$OUTFILE"
echo ""                                                                       | tee -a "$OUTFILE"

# Pre-flight checks — fail loudly so errors appear in timing.txt
if [[ ! -f "./$BINARY" ]]; then
  echo "ERROR: binary ./$BINARY not found in $PWD" | tee -a "$OUTFILE"
  exit 1
fi
if [[ ! -f "$NVME_DIR/$INPUT_BASENAME" ]]; then
  echo "ERROR: input not found at $NVME_DIR/$INPUT_BASENAME" | tee -a "$OUTFILE"
  exit 1
fi

# Capture both stdout and stderr from mpirun into timing.txt
mpirun --bind-to core -np "$NP" ./"$BINARY" \
  "$NVME_DIR/$INPUT_BASENAME" "$NVME_DIR" "$STEPS" 0 \
  2>&1 | tee -a "$OUTFILE"

# Copy final step output back
cp "$NVME_DIR/step_${STEPS}" "$OUTDIR/" 2>/dev/null || true

# Write per-job CSV (avoids race condition from concurrent jobs)
RESULTS_CSV="$OUTDIR/results_np${NP}.csv"
echo "binary,np,rank,total_s,comm_s,comm_pct,compute_s,compute_pct,io_s,io_pct" > "$RESULTS_CSV"
while IFS= read -r line; do
  if [[ "$line" =~ ^Rank[[:space:]]+([0-9]+)[[:space:]]\|[[:space:]]Total:[[:space:]]+([0-9.]+)[[:space:]]s[[:space:]]\|[[:space:]]Comm:[[:space:]]+([0-9.]+)[[:space:]]s[[:space:]]\(([0-9.]+)%\)[[:space:]]\|[[:space:]]Compute:[[:space:]]+([0-9.]+)[[:space:]]s[[:space:]]\(([0-9.]+)%\)[[:space:]]\|[[:space:]]IO:[[:space:]]+([0-9.]+)[[:space:]]s[[:space:]]\(([0-9.]+)%\) ]]; then
    echo "$BINARY,$NP,${BASH_REMATCH[1]},${BASH_REMATCH[2]},${BASH_REMATCH[3]},${BASH_REMATCH[4]},${BASH_REMATCH[5]},${BASH_REMATCH[6]},${BASH_REMATCH[7]},${BASH_REMATCH[8]}" >> "$RESULTS_CSV"
  fi
done < "$OUTFILE"

echo "Done. Results in $RESULTS_CSV"
