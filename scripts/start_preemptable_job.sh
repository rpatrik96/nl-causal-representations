#!/bin/bash 
PARTITION=gpu-2080ti-preemptable
#FLAGS= 
PYTHONPATH=. srun --job-name="$JOB_NAME" --partition=$PARTITION --cpus-per-task=2 --mem=4G --gpus=1 -- /mnt/qb/work/bethge/preizinger/nl-causal-representations//scripts/run_singularity_server.sh python3 /mnt/qb/work/bethge/preizinger/nl-causal-representations/care_nl_ica/main.py  --project mlp-test --use-batch-norm  --use-dep-mat --use-wandb  "$@"

