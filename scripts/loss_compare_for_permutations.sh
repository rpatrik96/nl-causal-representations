#!/bin/bash

for seed in 84645646; do
  for n in 3; do
    fact=1
    for ((i = 1; i <= ${n}; i++)); do
      fact=$(($fact * $i))
    done

    for ((variant = 0; variant < ${fact}; variant++)); do
      ./scripts/start_preemptable_job.sh --use-ar-mlp --seed ${seed} --n ${n} --variant ${variant} --note "Loss comparison with permuted data but no sinkhorn" --tags normalization nonlinear sem residual permute --use-sem --nonlin-sem --normalize-latents --verbose --permute &
      sleep 20
    done
  done
done
