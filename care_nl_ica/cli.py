from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers.wandb import WandbLogger

from care_nl_ica.utils import install_package, add_tags


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

        # parser.link_arguments("data.latent_dim", "model.latent_dim")
        # parser.link_arguments("data.box_min", "model.box_min")
        # parser.link_arguments("data.box_max", "model.box_max")
        # parser.link_arguments("data.sphere_r", "model.sphere_r")
        # parser.link_arguments("data.normalize_latents", "model.normalize_latents")
        #
        # parser.link_arguments("data.num_data", "model.num_data")
        # parser.link_arguments("data.num_layer", "model.num_layer")
        # parser.link_arguments("data.num_comp", "model.num_comp")
        # parser.link_arguments("data.num_basis", "model.num_basis")
        # parser.link_arguments("data.ar_order", "model.ar_order")
        # parser.link_arguments("data.batch_size", "model.batch_size")
        # parser.link_arguments("data.net_model", "model.net_model")
        # parser.link_arguments("data.num_segment", "model.num_segment")

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

            if self.config[self.subcommand].model.init_args.offline is True:
                self.trainer.logger.__dict__["_wandb_init"]["mode"] = "offline"
            else:
                self.trainer.logger.__dict__["_wandb_init"]["mode"] = "online"

            # todo: maybe set run in the CLI to false and call watch before?
            self.trainer.logger.watch(self.model, log="all", log_freq=250)


if __name__ == "__main__":
    install_package()

    cli = MyLightningCLI(
        save_config_callback=None,
        run=True,
        parser_kwargs={"parse_as_dict": False},
    )
