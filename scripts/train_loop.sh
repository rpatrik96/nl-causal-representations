#!/bin/bash

for seed in 7676
do
    for n in 4
    do  
        ./scripts/start_preemptable_job.sh --use-ar-mlp --seed ${seed} --n ${n} --note "metrics test normalization" --tags normalization nonlinear sem residual --use-sem --nonlin-sem --normalize-latents &
        sleep 20
    done
done
