
import torch
import torch.nn as nn


class LLinear(nn.Module):
    def __init__(
        self,
        dim_start: int, dim_end: int,
        pow_iters: int
    ) -> None:
        super().__init__()

        self.layer = nn.Linear(
            in_features=dim_start, out_features=dim_end, bias=False)

        with torch.no_grad():
            nn.init.xavier_uniform_(self.layer.weight)

        self.layer = nn.utils.parametrizations.spectral_norm(
            module=self.layer, n_power_iterations=pow_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        z_f = self.layer(x)

        return z_f
