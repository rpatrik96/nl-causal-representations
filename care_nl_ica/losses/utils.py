from dataclasses import dataclass

from typing import Dict


@dataclass
class Losses:
    cl_pos: float = 0.0
    cl_neg: float = 0.0
    sinkhorn_entropy: float = 0.0
    bottleneck_l1: float = 0.0
    sparsity_budget: float = 0.0
    triangularity: float = 0.0
    qr: float = 0.0

    @property
    def total_loss(self):
        total_loss = 0.0
        for loss in self.__dict__.values():
            total_loss += loss

        return total_loss

    def log_dict(self, panel_name) -> Dict[str, float]:
        return {
            f"{panel_name}/loss/cl_pos": self.cl_pos,
            f"{panel_name}/loss/cl_neg": self.cl_neg,
            f"{panel_name}/loss/sinkhorn_entropy": self.sinkhorn_entropy,
            f"{panel_name}/loss/bottleneck_l1": self.bottleneck_l1,
            f"{panel_name}/loss/sparsity_budget": self.sparsity_budget,
            f"{panel_name}/loss/triangularity": self.triangularity,
            f"{panel_name}/loss/qr": self.qr,
            f"{panel_name}/loss/total": self.total_loss,
        }
