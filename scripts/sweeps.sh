#!/bin/bash

for ((i = 1; i <= 1; i++));
do
    ./scripts/wandb_sweep.sh "$@" &
    sleep 1
done
