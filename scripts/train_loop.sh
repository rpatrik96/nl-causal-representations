#!/bin/bash

for variant in  2 
do
    for n in 9
    do  
        ./scripts/start_preemptable_job.sh --use-ar-mlp --variant ${variant} --n ${n} --note "metrics test normalization" --tags normalization --normalize-latents &
        sleep 20
    done
done
