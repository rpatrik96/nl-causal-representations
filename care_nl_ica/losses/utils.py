from dataclasses import dataclass

from typing import Dict


@dataclass
class Losses:
    cl_pos: float = 0.0
    cl_neg: float = 0.0
    cl_entropy: float = 0.0
    sinkhorn_entropy: float = 0.0
    bottleneck_l1: float = 0.0
    sparsity_budget: float = 0.0
    triangularity: float = 0.0
    qr: float = 0.0

    @property
    def total_loss(self):
        total_loss = 0.0
        for key, loss in self.__dict__.items():
            if key != "cl_entropy":
                total_loss += loss

        return total_loss

    def log_dict(self) -> Dict[str, float]:
        return {
            "cl_pos": self.cl_pos,
            "cl_neg": self.cl_neg,
            "cl_entropy": self.cl_entropy,
            "sinkhorn_entropy": self.sinkhorn_entropy,
            "bottleneck_l1": self.bottleneck_l1,
            "sparsity_budget": self.sparsity_budget,
            "triangularity": self.triangularity,
            "qr": self.qr,
            "total": self.total_loss,
        }
