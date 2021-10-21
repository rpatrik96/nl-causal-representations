# Identifying nonlinear causal graphs with ICA

## Singularity container build

```bash
singularity build --fakeroot nv.sif nv.def
```


## Usage 

1. Clone
```bash
 git clone https://github.com/rpatrik96/nl-causal-representations.git
```

2. Install
```bash
pip3 install -e .
```

3. Run:
```bash
 PYTHONPATH=/home/bethge/preizinger/nl-causal-representations/ python3 care_nl_ica/cl_causal.py --variant 1 --project mlp-test --use-ar-mlp --use-wandb --use-dep-mat --n-steps 150001 --n 2 --notes "Description of the run"
```

## Logging
https://wandb.ai/causal-representation-learning