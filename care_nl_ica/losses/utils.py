from dataclasses import dataclass

from typing import Dict


@dataclass
class ContrastiveLosses:
    cl_pos: float = 0.0
    cl_neg: float = 0.0
    cl_entropy: float = 0.0

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
            "total": self.total_loss,
        }
