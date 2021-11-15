#!/bin/bash
for cparam in 0.05 0.01 0.1
do
    for tau in 0.5 1.0 2.0
    do
    ./scripts/start_preemptable_job.sh --c-param ${cparam} --tau ${tau} --note "jacobian normalization with marginal assumption" --tags nonlinear sem normalization --use-sem --nonlin-sem &
    sleep 10
    done
done