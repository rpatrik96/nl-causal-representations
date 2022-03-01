#!/bin/bash

for ((i = 1; i <= 36; i++));
do
    ./scripts/wandb_sweep.sh "$@" &
    sleep 2
done
