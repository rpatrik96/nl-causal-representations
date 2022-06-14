#!/bin/bash 
PARTITION=gpu-2080ti-preemptable
FLAGS=--exclude=slurm-bm-62
PYTHONPATH=. srun --time=600 --job-name="$JOB_NAME" --partition=$PARTITION $FLAGS --mem=8G --gpus=1 -- /mnt/qb/work/bethge/preizinger/nl-causal-representations//scripts/run_singularity_server.sh wandb agent --count 1 "$@"

