
import torch
import torch.nn as nn
import torchsde

from ai.dynamics import Dynamics


class DiffEq(nn.Module):
    def __init__(
        self,
        drift_size: int, diffusion_size: int,
        dim_embedding: int, dim_middle: int,
        steps: int,
        rtol: float, atol: float,
        scale: float,
        dt: float,
        backwards: bool
    ) -> None:
        super().__init__()

        self.backwards = backwards

        self.dynamics = Dynamics(
            drift_size=drift_size, diffusion_size=diffusion_size,
            dim_embedding=dim_embedding, dim_middle=dim_middle,
            scale=scale, backwards=self.backwards
        )

        self.steps = steps
        self.rtol, self.atol = rtol, atol
        self.dt = dt

        self.dim_embedding = dim_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        x_shape = x.shape
        assert x_shape[-1] == self.dim_embedding

        x_flat = x.reshape(-1, x_shape[-1])

        times = torch.linspace(0.0, 1.0, self.steps, device=x.device)

        # diff_eq = torchsde.sdeint(
        diff_eq = torchsde.sdeint_adjoint(
            sde=self.dynamics,
            y0=x_flat,
            ts=times, dt=self.dt,
            rtol=self.rtol, atol=self.atol,
            method="euler"
        )

        return diff_eq[-1]
