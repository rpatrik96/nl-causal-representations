#!/bin/bash

for seed in 7676
do
    for n in 4
    do
        for l1 in  0 #1e1 1 1e-1 1e-2 1e-3 #1e-1 1 1e2 #0 1e-2 3e-3 1e-3 3e-4 #0 1e-6 1e-5 1e-4
        do  
            ./scripts/start_preemptable_job.sh --use-ar-mlp --seed ${seed} --n ${n} --l1 ${l1} --note "sinkhorn" --tags normalization nonlinear sem residual entropy sinkhorn --use-sem --nonlin-sem --normalize-latents --verbose --permute &
            sleep 20
        done
    done
done
