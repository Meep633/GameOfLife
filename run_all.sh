#!/bin/bash
# ============================================================
# Conway Game of Life — Full Benchmark Submission
# ============================================================

STEPS=${1:-100}

STRONG_BOARD="./board_10000.bin"
WEAK_BOARDS="."
RESULTS="../results"

# ── 1. Strong scaling: MPI + CUDA ────────────────────────────
echo ">>> [1/4] Strong scaling — conway (MPI+CUDA)"
bash run_scaling.sh strong "$STRONG_BOARD" "$RESULTS/strong_cuda" "$STEPS" conway
echo ""

# ── 2. Strong scaling: MPI only ──────────────────────────────
echo ">>> [2/4] Strong scaling — conway-mpi (MPI only)"
bash run_scaling.sh strong "$STRONG_BOARD" "$RESULTS/strong_mpi" "$STEPS" conway-mpi
echo ""

# ── 3. Weak scaling: MPI + CUDA ──────────────────────────────
echo ">>> [3/4] Weak scaling — conway (MPI+CUDA)"
bash run_scaling.sh weak "$WEAK_BOARDS" "$RESULTS/weak_cuda" "$STEPS" conway
echo ""

# ── 4. Weak scaling: MPI only ────────────────────────────────
echo ">>> [4/4] Weak scaling — conway-mpi (MPI only)"
bash run_scaling.sh weak "$WEAK_BOARDS" "$RESULTS/weak_mpi" "$STEPS" conway-mpi
echo ""
