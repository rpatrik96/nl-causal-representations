#!/bin/bash 
PARTITION=gpu-2080ti
#FLAGS= 
PYTHONPATH=. srun --job-name="$JOB_NAME" --partition=$PARTITION --cpus-per-task=4 --mem=16G --pty --gres=gpu:1 -- /mnt/qb/work/bethge/preizinger/nl-causal-representations//scripts/run_singularity_server.sh python3 /mnt/qb/work/bethge/preizinger/nl-causal-representations/care_nl_ica/cl_causal.py  --project mlp-test --use-batch-norm  --use-dep-mat --use-wandb --n-steps 125001 "$@"

