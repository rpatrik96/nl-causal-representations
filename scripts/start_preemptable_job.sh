#!/bin/bash 
PARTITION=gpu-2080ti
#FLAGS= 
PYTHONPATH=. srun --job-name="$JOB_NAME" --partition=$PARTITION --cpus-per-task=4 --mem=8G --pty --gres=gpu:1 -- ./scripts/run_singularity_server.sh python3 ./care_nl_ica/cl_causal.py  --project mlp-test --use-ar-mlp --use-batch-norm  --use-dep-mat --use-wandb --n-steps 300001 "$@"

