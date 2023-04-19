<div align="center"> 

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7002143.svg)](https://doi.org/10.5281/zenodo.7002143)

![CI testing](https://github.com/rpatrik96/nl-causal-representations/workflows/Python%20package/badge.svg?branch=master&event=push)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

</div>

# Jacobian-based Causal Discovery with Nonlinear ICA


## Description
This is the code for the paper _Jacobian-based Causal Discovery with Nonlinear ICA_, demonstrating how identifiable representations (particularly, with Nonlinear ICA) can be used to extract the causal graph from an underlying structural equation model (SEM).

## Before running the code

### Singularity container build

```bash
singularity build --fakeroot nv.sif nv.def
```

### Logging

1. First, you need to log into `wandb`
```bash
wandb login #you will find your API key at https://wandb.ai/authorize
```

2. Second, you need to specify the project for logging, which you can in the `before_fit` method in [cli.py](https://github.com/rpatrik96/nl-causal-representations/blob/master/care_nl_ica/cli.py#L37)
```python
    def before_fit(self):
        if isinstance(self.trainer.logger, WandbLogger) is True:
            # required as the parser cannot parse the "-" symbol
            self.trainer.logger.__dict__["_wandb_init"][
                "entity"
            ] = "causal-representation-learning" # <--- modify this line
```

3. Then, you can create and run the sweep
```bash
wandb sweep sweeps/sweep_file.yaml  # returns sweep ID
wandb agent <ID-comes-here> --count=<number of runs> # when used on a cluster, set it to one and start multiple processes
```


## Usage 

1. Clone
```bash
 git clone https://github.com/rpatrik96/nl-causal-representations.git
```

2. Install
```bash
# install package
pip3 install -e .

# install requirements 
pip install -r requirements.txt

# install pre-commit hooks
pre-commit install
```

3. Run:
```bash
python3 care_nl_ica/cli.py fit --config configs/config.yaml
```




### Code credits
Our repo extensively relies on `cl-ica` [repo](https://github.com/brendel-group/cl-ica), so please consider citing the corresponding [paper](http://proceedings.mlr.press/v139/zimmermann21a/zimmermann21a.pdf) as well


# Reference
If you find our work useful, please consider citing our [TMLR paper](https://openreview.net/forum?id=2Yo9xqR6Ab)

```bibtex
@article{reizinger2023jacobianbased,
  author = {
    Reizinger, Patrik and
    Sharma, Yash and
    Bethge, Matthias and
    Schölkopf, Bernhard and
    Huszár, Ferenc and
    Brendel, Wieland
  },
  title = {
    Jacobian-based Causal Discovery with Nonlinear {ICA}
  },
  journal={Transactions on Machine Learning Research},
  issn={2835-8856},
  year={2023},
  url={https://openreview.net/forum?id=2Yo9xqR6Ab},
}
```


