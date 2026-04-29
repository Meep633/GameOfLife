#!/bin/bash
mpirun --bind-to core -np $SLURM_NPROCS ./conway $1 $2 $3 $4
