#!/bin/bash 
PARTITION=gpu-2080ti-preemptable
#FLAGS= 
PYTHONPATH=. srun --time=360 --job-name="$JOB_NAME" --partition=$PARTITION --mem=8G --pty --gpus=1 -- scripts/run_singularity_server.sh python3 care_nl_ica/cli.py fit --config configs/config.yaml  --trainer.max_epochs=25000 --data.latent_dim=10 --model.offline=true --data.variant=4 --data.force_chain=true --data.nonlin_sem=true  --data.use_sem=false --data.permute=false  --data.data_gen_mode=rvs "$@"

