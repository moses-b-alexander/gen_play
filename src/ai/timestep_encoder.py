
import torch
import torch.nn as nn

from ai.utils import sinusoidal_positional_encoding


class TimestepEncoder(nn.Module):
    def __init__(
        self,
        trajectory_length: int,
        dim_stp: int, tau_stp: float
    ) -> None:
        super().__init__()

        if trajectory_length < 2:  self.trajectory_length = 2
        else:  self.trajectory_length = trajectory_length

        self.dim_stp = dim_stp

        self.tau_stp = 1.0 if tau_stp <= 0.0 or tau_stp > 1.0 else tau_stp

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        z_f = sinusoidal_positional_encoding(
            max_len=self.trajectory_length,
            actual_lens=x,
            dim_out=self.dim_stp,
            tau=torch.sigmoid(torch.tensor(self.tau_stp)).item(),
            device=x.device
        )

        return z_f
