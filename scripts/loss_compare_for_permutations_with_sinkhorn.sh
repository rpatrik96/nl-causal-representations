#!/bin/bash

N_STEPS=25001
START_STEP=2750

QR_LOSS=5e-1
TRIANGULARITY_LOSS=0
ENTROPY_COEFF=3e-3
LOSS_FLAGS="--qr-loss ${QR_LOSS} --triangularity-loss ${TRIANGULARITY_LOSS} --entropy-coeff ${ENTROPY_COEFF}"

NOTE="permute + small S + ar bottleneck + uniform"
DATA_FLAGS="--use-sem --permute"
LOG_FLAGS="--verbose --normalize-latents"
MODEL_FLAGS="--use-ar-mlp --sinkhorn"

for seed in 84645646; do
  for n in 3; do
    fact=1
    for ((i = 1; i <= ${n}; i++)); do
      fact=$(($fact * $i))
    done

    for ((variant = 0; variant < ${fact}; variant++)); do
      ./scripts/start_preemptable_job.sh  --seed ${seed} --n ${n} --variant "${variant}" --note "${NOTE}"  ${LOSS_FLAGS} --n-steps ${N_STEPS} --start-step ${START_STEP} ${DATA_FLAGS} ${LOG_FLAGS} ${MODEL_FLAGS} "$@" &
      sleep 5
    done
  done
done
