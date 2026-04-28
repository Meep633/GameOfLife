# Performance Data Tables

These tables summarize the data obtained from running Conway's Game of Life on the AiMOS cluster, using both an MPI-only CPU implementation and a hybrid MPI+CUDA implementation. These tables are formatted for direct inclusion into your project report.

## 1. Strong Scaling Performance
**Board Size:** 1440 x 1440 (Fixed)
**Steps:** 100

| MPI Ranks (np) | MPI-Only Time (s) | MPI-Only Speedup | MPI+CUDA Time (s) | MPI+CUDA Speedup |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.3411 | 1.00x | **0.0217** | **1.00x** |
| 4 | 0.2767 | 8.46x | 0.3073 | 0.07x |
| 9 | 1.3133 | 1.78x | 1.4890 | 0.01x |
| 16 | 1.3908 | 1.68x | 1.6940 | 0.01x |
| 25 | 1.5620 | 1.50x | 1.8981 | 0.01x |
| 36 | 1.8246 | 1.28x | 2.0915 | 0.01x |

*Note on CUDA Speedup:* At np=1, the entire board is resident on one GPU, avoiding network and domain boundary communication. As np increases, the communication and I/O overhead heavily dominate the vastly accelerated GPU compute, resulting in sub-linear scaling curves. 

---

## 2. Weak Scaling Performance
**Local Block Size:** 240 x 240 per rank (Total Size Grows)
**Steps:** 100

| MPI Ranks (np) | MPI-Only Time (s) | MPI-Only Efficiency | MPI+CUDA Time (s) | MPI+CUDA Efficiency |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.0308 | 1.00 | 0.0103 | 1.00 |
| 4 | 0.0444 | 0.69 | 0.2801 | 0.03 |
| 9 | 0.6262 | 0.04 | 0.7549 | 0.01 |
| 16 | 0.9007 | 0.03 | 1.2025 | 0.00 |
| 25 | 1.2182 | 0.02 | 1.6192 | 0.00 |
| 36 | 1.7736 | 0.01 | 2.0523 | 0.00 |

---

## 3. Phase Analysis: MPI+CUDA (Strong Scaling)
Provides a breakdown of the total runtime overhead to justify the scaling results. Time measured using POWER9 `clock_now()` precision timers.

| MPI Ranks (np) | Total (s) | Compute (s) | Compute % | Comm (s) | Comm % | I/O (s) | I/O % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 0.0217 | 0.0059 | 27.1% | 0.0000 | 0.1% | 0.0158 | 72.7% |
| 4 | 0.3073 | 0.0135 | 4.4% | 0.2557 | 83.2% | 0.0381 | 12.4% |
| 9 | 1.4890 | 0.0131 | 0.9% | 0.2251 | 15.1% | 1.2507 | 84.0% |
| 16 | 1.6940 | 0.0115 | 0.7% | 0.2582 | 15.2% | 1.4243 | 84.1% |
| 25 | 1.8981 | 0.0153 | 0.8% | 0.3088 | 16.3% | 1.5740 | 82.9% |
| 36 | 2.0915 | 0.0201 | 1.0% | 0.3551 | 17.0% | 1.7163 | 82.1% |

*Notice the severe I/O bottleneck when the system must aggregate writes over the network across multiple nodes (np >= 9).*

---

## 4. Phase Analysis: MPI-Only (Strong Scaling)

| MPI Ranks (np) | Total (s) | Compute (s) | Compute % | Comm (s) | Comm % | I/O (s) | I/O % |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | 2.3411 | 2.3189 | 99.1% | 0.0000 | 0.0% | 0.0222 | 0.9% |
| 4 | 0.2767 | 0.2345 | 84.7% | 0.0008 | 0.3% | 0.0414 | 15.0% |
| 9 | 1.3133 | 0.1201 | 9.1% | 0.0020 | 0.2% | 1.1912 | 90.7% |
| 16 | 1.3908 | 0.0434 | 2.8% | 0.0026 | 0.2% | 1.5159 | 97.1% |
| 25 | 1.5620 | 0.0262 | 1.4% | 0.0054 | 0.3% | 1.5159 | 97.1% |
| 36 | 1.8246 | 0.0262 | 1.4% | 0.0054 | 0.3% | 1.7930 | 98.3% |

*Here, the initial CPU execution is heavily compute-bound (99%), but super-linear scaling inside a single node (np=4) drastically reduces compute time until I/O dominates at higher node counts.*
