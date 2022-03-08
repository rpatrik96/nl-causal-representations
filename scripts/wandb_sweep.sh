#!/bin/bash 
PARTITION=gpu-2080ti-preemptable
#FLAGS= 
PYTHONPATH=. srun --time=360 --job-name="$JOB_NAME" --partition=$PARTITION --cpus-per-task=4 --mem=8G --gpus=1 -- /mnt/qb/work/bethge/preizinger/nl-causal-representations//scripts/run_singularity_server.sh wandb agent --count 1 "$@"

