import torch
from torch.nn import functional as F

from model import ContrastiveLearningModel
from utils import unpack_item_list


class Runner(object):

    def __init__(self, hparams) -> None:
        super().__init__()

        self.hparams = hparams

        self.model = ContrastiveLearningModel(self.hparams)
        self.optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=self.hparams.lr)

    def reset_encoder(self):
        self.model.reset_encoder()
        self.optimizer = torch.optim.Adam(self.model.encoder.parameters(), lr=self.hparams.lr)

    def train_step(self, data, h, test):
        device = self.hparams.device

        z1, z2_con_z1, z3 = data
        z1 = z1.to(device)
        z2_con_z1 = z2_con_z1.to(device)
        z3 = z3.to(device)

        # create random "negative" pairs
        # this is faster than sampling z3 again from the marginal distribution
        # and should also yield samples as if they were sampled from the marginal
        # import pdb; pdb.set_trace()
        # z3_shuffle_indices = torch.randperm(len(z1))
        # z3_shuffle_indices = torch.roll(torch.arange(len(z1)), 1)
        # z3 = z1[z3_shuffle_indices]
        # z3 = z3.to(device)

        self.optimizer.zero_grad()

        z1_rec = h(z1)
        z2_con_z1_rec = h(z2_con_z1)
        z3_rec = h(z3)
        # z3_rec = z1_rec[z3_shuffle_indices]

        if test:
            total_loss_value = F.mse_loss(z1_rec, z1)
            losses_value = [total_loss_value]
        else:
            total_loss_value, _, losses_value = self.model.loss(
                z1, z2_con_z1, z3, z1_rec, z2_con_z1_rec, z3_rec
            )

            # writer.add_scalar("loss_hz", total_loss_value, global_step)
            # writer.add_scalar("loss_z", loss(
            #    z1, z2_con_z1, z3, z1, z2_con_z1, z3
            # )[0], global_step)
            # writer.flush()

        if not self.hparams.identity_mixing_and_solution and self.hparams.lr != 0:
            total_loss_value.backward()
            self.optimizer.step()

        return total_loss_value.item(), unpack_item_list(losses_value)

    def train(self, data, h, test):
        if self.hparams.lr != 0:
            total_loss, losses = self.train_step(data, h=h, test=test)
        else:
            with torch.no_grad():
                total_loss, losses = self.train_step(data, h=h, test=test)

        return total_loss, losses
