from care_nl_ica.utils import install_package
import hydra
from omegaconf import DictConfig, OmegaConf
from argparse import Namespace


@hydra.main(config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    # install the package
    install_package()

    from pytorch_lightning import Trainer, seed_everything
    from care_nl_ica.data.datamodules import ContrastiveDataModule
    from care_nl_ica.runner import ContrastiveICAModule

    seed_everything(cfg.seed_everything)

    trainer = Trainer.from_argparse_args(Namespace(**cfg))
    model = ContrastiveICAModule(**OmegaConf.to_container(cfg.model))
    dm = ContrastiveDataModule.from_argparse_args(Namespace(**cfg.data))

    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    main()
