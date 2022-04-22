from pytorch_lightning.loggers.wandb import WandbLogger
from pytorch_lightning.utilities.cli import LightningCLI

from care_nl_ica.utils import add_tags
from care_nl_ica.data.datamodules import ContrastiveDataModule
from care_nl_ica.runner import ContrastiveICAModule


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--notes",
            type=str,
            default=None,
            help="Notes for the run on Weights and Biases",
        )
        # todo: process notes based on args in before_instantiate_classes
        parser.add_argument(
            "--tags",
            type=str,
            nargs="*",  # 0 or more values expected => creates a list
            default=None,
            help="Tags for the run on Weights and Biases",
        )

        parser.link_arguments("data.latent_dim", "model.latent_dim")
        parser.link_arguments("data.box_min", "model.box_min")
        parser.link_arguments("data.box_max", "model.box_max")
        parser.link_arguments("data.sphere_r", "model.sphere_r")
        parser.link_arguments("data.normalize_latents", "model.normalize_latents")

    def before_instantiate_classes(self) -> None:
        self.config[self.subcommand].trainer.logger.init_args.tags = add_tags(
            self.config[self.subcommand]
        )

    def before_fit(self):
        if isinstance(self.trainer.logger, WandbLogger) is True:
            # required as the parser cannot parse the "-" symbol
            self.trainer.logger.__dict__["_wandb_init"][
                "entity"
            ] = "causal-representation-learning"

            # todo: maybe set run in the CLI to false and call watch before?
            self.trainer.logger.watch(self.model, log="all", log_freq=250)


cli = MyLightningCLI(
    ContrastiveICAModule,
    ContrastiveDataModule,
    save_config_callback=None,
    run=True,
    parser_kwargs={"parse_as_dict": False},
)
