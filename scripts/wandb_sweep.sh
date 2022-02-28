#!/bin/bash 
PARTITION=gpu-2080ti-preemptable
#FLAGS= 
PYTHONPATH=. srun --job-name="$JOB_NAME" --partition=$PARTITION --cpus-per-task=4 --mem=8G --pty --gpus=1 -- wandb agent --count 1 causal-representation-learning/nl-causal-representations/<SWEEP_ID>  "$@"

