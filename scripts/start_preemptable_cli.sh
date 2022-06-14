#!/bin/bash 
PARTITION=gpu-2080ti-preemptable
#FLAGS= 
PYTHONPATH=. srun --time=60 --job-name="$JOB_NAME" --partition=$PARTITION --mem=8G --pty --gpus=1 -- scripts/run_singularity_server.sh python3 care_nl_ica/cli.py fit --config configs/config.yaml  --model.weight_init_fn=orthogonal --trainer.max_epochs=1 --data.latent_dim=5 --model.start_step=400 --model.gain=1 --model.qr=0.0 --model.offline=true --data.variant=4 --data.force_chain=true --data.nonlin_sem=true  --data.use_sem=true --data.permute=false --model.use_ar_mlp=false --data.offset=1 "$@"

