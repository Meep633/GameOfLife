#!/usr/bin/env python3
"""
Parse all timing.txt files from scaling results and produce all_results.csv
for each study (strong_cuda, strong_mpi, weak_cuda, weak_mpi).

Usage: python3 parse_results.py [results_dir]
Default results_dir: ./results
"""

import re
import os
import sys

RESULTS_DIR = sys.argv[1] if len(sys.argv) > 1 else "./results"

# Matches: Rank  0 | Total:   0.0217 s | Comm:   0.0000 s (  0.1%) | Compute: ...
RANK_RE = re.compile(
    r"Rank\s+(\d+)\s+\|\s+Total:\s+([\d.]+)\s+s\s+\|"
    r"\s+Comm:\s+([\d.]+)\s+s\s+\(\s*([\d.]+)%\)\s+\|"
    r"\s+Compute:\s+([\d.]+)\s+s\s+\(\s*([\d.]+)%\)\s+\|"
    r"\s+IO:\s+([\d.]+)\s+s\s+\(\s*([\d.]+)%\)"
)

HEADER = "binary,np,rank,total_s,comm_s,comm_pct,compute_s,compute_pct,io_s,io_pct"

studies = ["strong_cuda", "strong_mpi", "weak_cuda", "weak_mpi"]

for study in studies:
    study_dir = os.path.join(RESULTS_DIR, study)
    if not os.path.isdir(study_dir):
        print(f"SKIP {study} — directory not found")
        continue

    rows = []
    # Determine binary name from study name
    binary = "conway" if "cuda" in study else "conway-mpi"

    for entry in sorted(os.listdir(study_dir)):
        if not entry.startswith("np_"):
            continue
        timing_file = os.path.join(study_dir, entry, "timing.txt")
        if not os.path.isfile(timing_file):
            print(f"  WARNING: {timing_file} not found")
            continue
        np_val = entry.split("_")[1]
        with open(timing_file) as f:
            for line in f:
                m = RANK_RE.search(line)
                if m:
                    rank, total, comm, comm_pct, compute, compute_pct, io, io_pct = m.groups()
                    rows.append(f"{binary},{np_val},{rank},{total},{comm},{comm_pct},{compute},{compute_pct},{io},{io_pct}")

    out_path = os.path.join(study_dir, "all_results.csv")
    with open(out_path, "w") as f:
        f.write(HEADER + "\n")
        for row in rows:
            f.write(row + "\n")
    print(f"✅ {study}: {len(rows)} rows → {out_path}")

print("\nDone. Summary of rank-0 totals per study:")
for study in studies:
    csv = os.path.join(RESULTS_DIR, study, "all_results.csv")
    if not os.path.isfile(csv):
        continue
    print(f"\n{study}:")
    with open(csv) as f:
        header = f.readline()
        cols = header.strip().split(",")
        np_idx = cols.index("np")
        rank_idx = cols.index("rank")
        total_idx = cols.index("total_s")
        for line in sorted(f, key=lambda l: int(l.split(",")[np_idx])):
            parts = line.strip().split(",")
            if parts[rank_idx] == "0":
                print(f"  np={parts[np_idx]:>2}  total={parts[total_idx]}s")
