#!/usr/bin/env python3
import subprocess
import datetime
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import numpy as np

print("Regenerating CSVs from timing.txt files...")
subprocess.run(["python3", "parse_results.py"], check=True)

out_dir = "report_figures"
os.makedirs(out_dir, exist_ok=True)

def load_data(study_name):
    path = f"results/{study_name}/all_results.csv"
    if not os.path.exists(path):
        return pd.DataFrame()

    mtime = os.path.getmtime(path)
    ts = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
    print(f"  {study_name}: loaded {path}  (last modified: {ts})")
    df = pd.read_csv(path)
    return df[df['rank'] == 0].sort_values('np')

df_strong_cuda = load_data("strong_cuda")
df_strong_mpi = load_data("strong_mpi")
df_weak_cuda = load_data("weak_cuda")
df_weak_mpi = load_data("weak_mpi")

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 12,
    'axes.labelsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.figsize': (5, 4),
    'figure.dpi': 300,
    'lines.linewidth': 2.5,
    'lines.markersize': 8
})

def save_fig(filename):
    plt.tight_layout(pad=1.0)
    plt.savefig(f"{out_dir}/{filename}", bbox_inches='tight')
    plt.close()

plt.figure()
if not df_strong_cuda.empty:
    t1_cuda = df_strong_cuda[df_strong_cuda['np'] == 1]['total_s'].values[0]
    plt.plot(df_strong_cuda['np'], t1_cuda / df_strong_cuda['total_s'], marker='o', label='MPI+CUDA')
if not df_strong_mpi.empty:
    t1_mpi = df_strong_mpi[df_strong_mpi['np'] == 1]['total_s'].values[0]
    plt.plot(df_strong_mpi['np'], t1_mpi / df_strong_mpi['total_s'], marker='s', label='MPI-Only')

nps = df_strong_cuda['np'].values if not df_strong_cuda.empty else df_strong_mpi['np'].values
plt.plot(nps, nps, '--', color='gray', label='Ideal')

plt.title('Strong Scaling Speedup')
plt.xlabel('Number of MPI Ranks')
plt.ylabel('Speedup')
plt.legend()
save_fig('strong_scaling_speedup.png')

# 2. Weak Scaling Efficiency
plt.figure()
if not df_weak_cuda.empty:
    t1_cuda = df_weak_cuda[df_weak_cuda['np'] == 1]['total_s'].values[0]
    plt.plot(df_weak_cuda['np'], t1_cuda / df_weak_cuda['total_s'], marker='o', label='MPI+CUDA')
if not df_weak_mpi.empty:
    t1_mpi = df_weak_mpi[df_weak_mpi['np'] == 1]['total_s'].values[0]
    plt.plot(df_weak_mpi['np'], t1_mpi / df_weak_mpi['total_s'], marker='s', label='MPI-Only')

nps = df_weak_cuda['np'].values if not df_weak_cuda.empty else df_weak_mpi['np'].values
plt.plot(nps, [1.0]*len(nps), '--', color='gray', label='Ideal')

plt.title('Weak Scaling Efficiency')
plt.xlabel('Number of MPI Ranks')
plt.ylabel('Efficiency')

plt.legend(loc='lower left')
save_fig('weak_scaling_efficiency.png')

plt.figure()
if not df_strong_cuda.empty:
    plt.plot(df_strong_cuda['np'], df_strong_cuda['total_s'], marker='o', label='MPI+CUDA')
if not df_strong_mpi.empty:
    plt.plot(df_strong_mpi['np'], df_strong_mpi['total_s'], marker='s', label='MPI-Only')

plt.title('Strong Scaling Total Time')
plt.xlabel('Number of MPI Ranks')
plt.ylabel('Time (s)')
plt.yscale('log')
plt.legend()
save_fig('total_time_comparison_strong.png')

if not df_strong_cuda.empty:
    fig, ax = plt.subplots()
    
    nps = df_strong_cuda['np'].astype(str).values
    compute = df_strong_cuda['compute_s'].values
    comm = df_strong_cuda['comm_s'].values
    io = df_strong_cuda['io_s'].values
    
    x = np.arange(len(nps))
    width = 0.6
    
    p1 = ax.bar(x, compute, width, label='Compute')
    p2 = ax.bar(x, comm, width, bottom=compute, label='Comm')
    p3 = ax.bar(x, io, width, bottom=compute+comm, label='I/O')
    
    ax.set_ylabel('Time (s)')
    ax.set_title('Phase Breakdown (MPI+CUDA)')
    ax.set_xticks(x)
    ax.set_xticklabels(nps)
    ax.set_xlabel('MPI Ranks')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    save_fig('phase_breakdown_strong_cuda.png')

if not df_strong_mpi.empty:
    fig, ax = plt.subplots()
    
    nps = df_strong_mpi['np'].astype(str).values
    compute = df_strong_mpi['compute_s'].values
    comm = df_strong_mpi['comm_s'].values
    io = df_strong_mpi['io_s'].values
    
    x = np.arange(len(nps))
    width = 0.6
    
    p1 = ax.bar(x, compute, width, label='Compute')
    p2 = ax.bar(x, comm, width, bottom=compute, label='Comm')
    p3 = ax.bar(x, io, width, bottom=compute+comm, label='I/O')
    
    ax.set_ylabel('Time (s)')
    ax.set_title('Phase Breakdown (MPI-Only)')
    ax.set_xticks(x)
    ax.set_xticklabels(nps)
    ax.set_xlabel('MPI Ranks')
    
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=3)
    save_fig('phase_breakdown_strong_mpi.png')

print(f"Graphs successfully regenerated for 2-column format in ./{out_dir}/")
