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
 git clone --recurse-submodules https://github.com/rpatrik96/nl-causal-representations.git
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


# Using Structured Neural Networks (StrNNs) as s structural inductive bias
Since th publication in TMLR, we have been experimenting with encoding the inductive bias of SEMs (the triangular structure) 
in Structured Neural Networks (StrNNs).

There are two scenarios...:
- Causal Discovery (CD):
    -  the latent variables and observations have the same dimenions, we assume observing the causal variables
    -  if we assume that we know the causal order (i.e., there is no permutation indeterminacy), then StrNNs improve identifiability scores/convergence (tested in CL-ICA)
    -  if the causal order is not known, then we can try to learn a permutation matrix (via Sinkhorn networks, see CL-ICA for detail), but learning the permutation doesn't work
- Causal Representation Learning (CRL):
    -  the latent variables lie on a low-dimensional manifold, the observations are a high-dimensional mixture of the causal variables
    -  in this case, the encoder is either a single MLP or an MLP (to map to causal variables) and an StrNN
    -  StrNNs do not help

...and three algorithms:
- Contrastive ICA (CL-ICA)
- iVAE
- ICE-BeeM
- (CauCA is not used currently)

## Contrastive ICA
- The StrNN unmixing is defined in the `_setup_unmixing` function in `care_nl_ica/models/model.py`
  - To use an StrNN, set `model.strnn: true` (in either `configs/config.yaml` or any of the sweep config files)
  - `model.obs_layers` selects between CD (set to 0) and CRL (>0)
  - `model.strnn_layers` defines the number of layers
  - the width depends on `model.width_factor` and is calculated by `model.latent_dim * model.width_factor`
  - if `model.permute is True` then a learnable permutation, in form of a Sinkhorn Network (`models/sinkhorn.py`) is added to `nn.Sequential`
- To run `wandb` sweeps, check any sweep configuration file matching the pattern `configs/sem/*_strnn.yaml`.
- `model.path_optimizer` turns a Path optimizer On/Off

## iVAE
- The StrNN unmixing is defined in the `_setup_encoder` function in `ivae/nets.py`, the observational unmixing (in case of CRL) is in the same file in the `_setup_obs_unmixing` function
   - To use an StrNN, set `use_strnn: true` (in either `ivae/configs/ivae.yaml` or any of the sweep config files)
  - `cond_strnn` switches between vanilla and conditional StrNN (conditional in a sense that the blocks can depend on another variable  think attention
  - `strnn_adjacency_override` enables to set the StrNN adjacency to the ground-truth adjacency (the one used to generate the data)
  - `separate_aux` adds a separate MLP before passing the input to the StrNN
  - `residual_aux` converts the StrNN into a residual network where an auxiliary MLP is used in the residual branch
  - `ignore_u` (also applies to the vanilla iVAE without StrNNs) decides whether the auxiliary information `u` is used in calculating the mean encodings
- the SEM can also be set to an StrNN by configuring `nl>=2` (number of layers in the data generating process)
- iVAE sweep configs match the pattern `ivae/configs/ivae_sweep_*.yaml`
  - use `--sweep` when running `ivae/main.py` to run a wandb sweep
  - specify the config with `--config`
  - if you have a sweep ID, use `--sweep` and pass the ID into `--sweep-id`


## ICE-BeeM
- The StrNN unmixing is defined in the `_ICEBEEM_wrapper` function in `icebeem/icebeem_wrapper.py`
   - To use an StrNN, set `use_strnn: true` (in either `icebeem/configs/imca.yaml` or any of the sweep config files)
- the main file is `icebeem/simulations.py`
- the config is `icebeem/configs/imca.yaml`
- sweeps match the pattern `icebeem/configs/imca_sweep_*.yaml`
