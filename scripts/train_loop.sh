#!/bin/bash

for variant in 1 #2 7 25 45646
do
    for n in 8
    do  
        ./scripts/start_preemptable_job.sh --use-ar-mlp --variant ${variant} --n ${n} --note "metrics test normalization" --tags normalization --normalize-latents &
        sleep 20
    done
done
