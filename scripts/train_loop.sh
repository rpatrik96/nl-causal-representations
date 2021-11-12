#!/bin/bash
for seed in 0 42 7676
do
    for variant in 1 2 7 
    do
        for n in 2 6
        do  
            ./start_preemptable_job.sh --seed ${seed} --variant ${variant} --n ${n} &
            sleep 10
        done
    done
done