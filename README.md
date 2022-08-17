[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7002143.svg)](https://doi.org/10.5281/zenodo.7002143)
![CI testing](https://github.com/rpatrik96/nl-causal-representations/workflows/python-package/badge.svg?branch=master&event=push)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

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
# install pre-commit hooks
pre-commit install
```

3. Run:
```bash
 PYTHONPATH=/mnt/qb/work/bethge/preizinger/nl-causal-representations/ python3 care_nl_ica/main.py --variant 1 --project mlp-test --use-ar-mlp --use-wandb --use-dep-mat --use-sem --nonlin-sem --n-steps 1501 --n 3 --notes "Description of the run" --permute
```

Or
```bash
python3 care_nl_ica/cli.py fit --config configs/config.yaml
```

## Logging
https://wandb.ai/causal-representation-learning


### Code credits
Our repo extensively relies on `cl-ica` [repo](https://github.com/brendel-group/cl-ica), so please consider citing the corresponding [paper](http://proceedings.mlr.press/v139/zimmermann21a/zimmermann21a.pdf) as well


# Reference
If you find our work useful, please consider citing our workshop paper

```bibtex
@article{reizinger2022multivariable,
  author = {
    Reizinger, Patrik and
    Sharma, Yash and
    Bethge, Matthias and
    Schölkopf, Bernhard and
    Huszár, Ferenc and
    Brendel, Wieland
  },
  title = {
    Multivariable Causal Discovery with General Nonlinear Relationships
  },
  booktitle = {First Workshop on Causal Representation Learning ({CRL} 2022), {UAI} 2022},
  year = {2022},
  url = {https://openreview.net/forum?id=TsXe-CyYJqx},
}
```

