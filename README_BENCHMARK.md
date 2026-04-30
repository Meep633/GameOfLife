# Conway Game of Life — Benchmarking Guide

This directory contains the scripts and tools needed to run the Strong and Weak scaling benchmarks for both the MPI-only (`conway-mpi`) and MPI+CUDA (`conway`) implementations of Conway's Game of Life on the AiMOS (DCS) cluster.

## 1. Building the Binaries

Before running anything, compile the necessary binaries using the provided `Makefile`:

```bash
make all
```
This will compile `conway` (MPI+CUDA), `conway-mpi` (MPI only), and `conway-basic`.

---

## 2. Configuration & Paths

**IMPORTANT:** Before running the scripts, you must update the path variables so they point to your actual directories:

1. **`merge_results.sh`**: Change `SCRATCH=/gpfs/u/home/PCPG/$USER/scratch` to point to your actual scratch or results directory. Otherwise, the script will fail to find your generated CSVs.
2. **`run_all.sh`**: Ensure `RESULTS="../results"` points to your desired output directory, and that `STRONG_BOARD` and `WEAK_BOARDS` correctly point to where you generated the boards.

---

## 3. Prerequisites (Generating Boards)

Before running the benchmarks, ensure you have generated the necessary board files.

For **Strong Scaling**, we use a massive board:
```bash
python3 generate_board.py 10000 10000 0.7 board_10000.bin
```

For **Weak Scaling**, we use proportionally sized boards:
```bash
python3 generate_board.py 100 100 0.7 board_100.bin
python3 generate_board.py 200 200 0.7 board_200.bin
python3 generate_board.py 400 400 0.7 board_400.bin
python3 generate_board.py 800 800 0.7 board_800.bin
python3 generate_board.py 1600 1600 0.7 board_1600.bin
```

---

## 4. Running the Benchmarks

```bash
bash run_all.sh
```

### What happens under the hood?
1. **`run_all.sh`**: The master script. It calls `run_scaling.sh` four times (Strong CUDA, Strong MPI, Weak CUDA, Weak MPI).
2. **`run_scaling.sh`**: The submission loop. It loops over the MPI ranks (1, 4, 16, 25, 64, 100, 256) and computes the exact nodes/tasks layout. It then uses `sbatch` to submit each individual job to the `dcs-2024` partition.
3. **`run_benchmark_job.sh`**: The payload script. Once Slurm allocates the nodes, this script is executed. It copies the required board to the temporary fast NVMe storage, runs `mpirun`, times the IO/Compute/Comm, and writes the output to CSV files.

---

## 5. Merging the Results

Once **all** jobs have finished running (they no longer appear in `squeue`), you need to merge the scattered CSV outputs into consolidated result files for plotting.

Run the merge script directly on the login node:
```bash
bash merge_results.sh
```

This will create a clean `all_results.csv` inside each experiment's folder:
- `../results/strong_cuda/all_results.csv`
- `../results/strong_mpi/all_results.csv`
- `../results/weak_cuda/all_results.csv`
- `../results/weak_mpi/all_results.csv`

---

## 6. Generating Graphs

With your merged `all_results.csv` files ready, use the Python plotting script to generate performance graphs:

```bash
python3 generate_graphs.py
```
*(Make sure the paths inside `generate_graphs.py` point to the `../results` directories generated above).*

## 7. Cleanup

```
rm -r ../results
```

