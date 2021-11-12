#!/bin/bash 
PARTITION=gpu-2080ti-interactive 
FLAGS=--exclude=slurm-bm-06
#FLAGS= 
srun --job-name="$JOB_NAME" --partition=$PARTITION --pty --gres=gpu:1 "$FLAGS" -- ./scripts/interactive_job_inner.sh 

