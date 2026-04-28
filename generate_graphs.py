#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np

out_dir = "report_figures"
os.makedirs(out_dir, exist_ok=True)

def load_data(study_name):
    path = f"results/{study_name}/all_results.csv"
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    return df[df['rank'] == 0].sort_values('np')

df_strong_cuda = load_data("strong_cuda")
df_strong_mpi = load_data("strong_mpi")
df_weak_cuda = load_data("weak_cuda")
df_weak_mpi = load_data("weak_mpi")

plt.style.use('seaborn-v0_8-whitegrid')

plt.figure(figsize=(8, 6))
if not df_strong_cuda.empty:
    t1_cuda = df_strong_cuda[df_strong_cuda['np'] == 1]['total_s'].values[0]
    plt.plot(df_strong_cuda['np'], t1_cuda / df_strong_cuda['total_s'], marker='o', label='MPI+CUDA')
if not df_strong_mpi.empty:
    t1_mpi = df_strong_mpi[df_strong_mpi['np'] == 1]['total_s'].values[0]
    plt.plot(df_strong_mpi['np'], t1_mpi / df_strong_mpi['total_s'], marker='s', label='MPI-Only')

nps = df_strong_cuda['np'].values if not df_strong_cuda.empty else df_strong_mpi['np'].values
plt.plot(nps, nps, '--', color='gray', label='Ideal Speedup')

plt.title('Strong Scaling Speedup (1440x1440 Board)')
plt.xlabel('Number of MPI Ranks (np)')
plt.ylabel('Speedup (T1 / Tnp)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}/strong_scaling_speedup.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
if not df_weak_cuda.empty:
    t1_cuda = df_weak_cuda[df_weak_cuda['np'] == 1]['total_s'].values[0]
    plt.plot(df_weak_cuda['np'], t1_cuda / df_weak_cuda['total_s'], marker='o', label='MPI+CUDA')
if not df_weak_mpi.empty:
    t1_mpi = df_weak_mpi[df_weak_mpi['np'] == 1]['total_s'].values[0]
    plt.plot(df_weak_mpi['np'], t1_mpi / df_weak_mpi['total_s'], marker='s', label='MPI-Only')

nps = df_weak_cuda['np'].values if not df_weak_cuda.empty else df_weak_mpi['np'].values
plt.plot(nps, [1.0]*len(nps), '--', color='gray', label='Ideal Efficiency')

plt.title('Weak Scaling Efficiency (240x240 Local Block)')
plt.xlabel('Number of MPI Ranks (np)')
plt.ylabel('Efficiency (T1 / Tnp)')
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}/weak_scaling_efficiency.png", dpi=300)
plt.close()

plt.figure(figsize=(8, 6))
if not df_strong_cuda.empty:
    plt.plot(df_strong_cuda['np'], df_strong_cuda['total_s'], marker='o', label='MPI+CUDA')
if not df_strong_mpi.empty:
    plt.plot(df_strong_mpi['np'], df_strong_mpi['total_s'], marker='s', label='MPI-Only')

plt.title('Strong Scaling Total Execution Time')
plt.xlabel('Number of MPI Ranks (np)')
plt.ylabel('Total Time (Seconds)')

plt.yscale('log')
plt.legend()
plt.tight_layout()
plt.savefig(f"{out_dir}/total_time_comparison_strong.png", dpi=300)
plt.close()

if not df_strong_cuda.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    nps = df_strong_cuda['np'].astype(str).values
    compute = df_strong_cuda['compute_s'].values
    comm = df_strong_cuda['comm_s'].values
    io = df_strong_cuda['io_s'].values
    
    x = np.arange(len(nps))
    width = 0.6
    
    p1 = ax.bar(x, compute, width, label='Compute (GPU)')
    p2 = ax.bar(x, comm, width, bottom=compute, label='Communication (MPI)')
    p3 = ax.bar(x, io, width, bottom=compute+comm, label='File I/O (NVMe)')
    
    ax.set_ylabel('Time (Seconds)')
    ax.set_title('Phase Breakdown: Strong Scaling (MPI+CUDA)')
    ax.set_xticks(x)
    ax.set_xticklabels(nps)
    ax.set_xlabel('Number of MPI Ranks (np)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/phase_breakdown_strong_cuda.png", dpi=300)
    plt.close()

if not df_strong_mpi.empty:
    fig, ax = plt.subplots(figsize=(10, 6))
    
    nps = df_strong_mpi['np'].astype(str).values
    compute = df_strong_mpi['compute_s'].values
    comm = df_strong_mpi['comm_s'].values
    io = df_strong_mpi['io_s'].values
    
    x = np.arange(len(nps))
    width = 0.6
    
    p1 = ax.bar(x, compute, width, label='Compute (CPU)')
    p2 = ax.bar(x, comm, width, bottom=compute, label='Communication (MPI)')
    p3 = ax.bar(x, io, width, bottom=compute+comm, label='File I/O (NVMe)')
    
    ax.set_ylabel('Time (Seconds)')
    ax.set_title('Phase Breakdown: Strong Scaling (MPI-Only)')
    ax.set_xticks(x)
    ax.set_xticklabels(nps)
    ax.set_xlabel('Number of MPI Ranks (np)')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f"{out_dir}/phase_breakdown_strong_mpi.png", dpi=300)
    plt.close()

print(f"Graphs successfully generated in ./{out_dir}/")
