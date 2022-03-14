#!/bin/bash
python3 care_nl_ica/cli.py fit --config configs/config.yaml --config configs/model/ar_non_tri.yaml   --config configs/data/permute.yaml --config configs/data/nl_sem.yaml --trainer.profiler=simple --trainer.max_epochs=2

