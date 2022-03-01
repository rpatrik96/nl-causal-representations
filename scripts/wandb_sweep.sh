#!/bin/bash 
PARTITION=gpu-2080ti-preemptable
#FLAGS= 
PYTHONPATH=. srun --job-name="$JOB_NAME" --partition=$PARTITION --cpus-per-task=4 --mem=8G --pty --gpus=1 -- /mnt/qb/work/bethge/preizinger/nl-causal-representations//scripts/run_singularity_server.sh wandb agent --count 1 causal-representation-learning/nl-causal-representations/q3yh8bx6  "$@"

