#!/bin/bash
for seed in 0 42 7676 8778675
do
    ./start_preemptable_job.sh --seed ${seed} --note "jacobian scale seed test nonlinear sem" --tags nonlinear sem --use-sem --nonlin-sem &
    sleep 10
done