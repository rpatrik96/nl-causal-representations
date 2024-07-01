from care_nl_ica.cli import MyLightningCLI
from care_nl_ica.runner import ContrastiveICAModule
from care_nl_ica.data.datamodules import ContrastiveDataModule
from os.path import abspath, dirname, join


def test_cli_fast_dev_run():
    config_path = join(dirname(dirname(abspath(__file__))), "configs", "config.yaml")

    args = [
        "fit",
        "--config",
        config_path,
        "--trainer.fast_dev_run",
        "true",
        "--trainer.max_epochs",
        "5",
        "--trainer.logger",
        "null",
        "--model.strnn",
        "true",
        "--data.obs_dim",
        "5",
        "--data.batch_size",
        "64",
        "--data.latent_dim",
        "3",
    ]
    cli = MyLightningCLI(
        ContrastiveICAModule,
        ContrastiveDataModule,
        save_config_callback=None,
        run=True,
        args=args,
        parser_kwargs={"parse_as_dict": False},
    )
