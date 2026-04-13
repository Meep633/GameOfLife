#!/bin/bash
INPUT_FILE_NAME=$(basename "$2")
NVME_DIR="/mnt/nvme/uid_${SLURM_JOB_UID}/job_${SLURM_JOB_ID}"
cp $2 $NVME_DIR/
mpirun --bind-to core -np $1 ./conway $NVME_DIR/$INPUT_FILE_NAME $NVME_DIR $4
cp -r $NVME_DIR $3