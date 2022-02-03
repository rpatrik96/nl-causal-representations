#!/bin/bash

N_STEPS=12001
START_STEP=5000

QR_LOSS=3e-1
TRIANGULARITY_LOSS=0
ENTROPY_COEFF=0
LOSS_FLAGS="--qr-loss ${QR_LOSS} --triangularity-loss ${TRIANGULARITY_LOSS} --entropy-coeff ${ENTROPY_COEFF}"

DATA_FLAGS="--use-sem --permute"
LOG_FLAGS="--verbose --normalize-latents"
MODEL_FLAGS="--use-ar-mlp"

for seed in 84645646; do
  for n in 3; do
    fact=1
    for ((i = 1; i <= ${n}; i++)); do
      fact=$(($fact * $i))
    done

    for ((variant = 0; variant < ${fact}; variant++)); do
      ./scripts/start_preemptable_job.sh  --seed ${seed} --n ${n} --variant "${variant}"  ${LOSS_FLAGS} --n-steps ${N_STEPS} --start-step ${START_STEP} ${DATA_FLAGS} ${LOG_FLAGS}  "$@"  &
      sleep 5
    done
  done
done
