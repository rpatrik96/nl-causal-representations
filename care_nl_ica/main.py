import os

import pip


def install_package():
    """
    Install the current package to ensure that imports work.
    """
    try:
        import care_nl_ica
    except:
        print("Package not installed, installing...")
        pip.main(
            [
                "install",
                f"{os.path.dirname(os.path.dirname(os.path.abspath(__file__)))}",
                "--upgrade",
            ]
        )


import hydra
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # install the package
    install_package()

    from pytorch_lightning import Trainer, seed_everything
    from care_nl_ica.datamodules import ContrastiveDataModule
    from care_nl_ica.runner import ContrastiveICAModule

    seed_everything(cfg.seed_everything)

    trainer = Trainer.from_argparse_args(Namespace(**cfg))
    model = ContrastiveICAModule(**OmegaConf.to_container(cfg.model))
    dm = ContrastiveDataModule.from_argparse_args(Namespace(**cfg.data))

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
