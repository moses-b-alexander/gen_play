
import torch
import torch.nn as nn


class Normalizer(nn.Module):
    def __init__(
        self,
        type_norm: str,
        dim_start: int, dim_middle: int, dim_end: int,
        biases: tuple[bool], norm_bias: bool,
        pow_iters: int=1
    ) -> None:
        super().__init__()

        self.type_norm = type_norm.lower()

        self.biases = biases

        self.first = nn.Linear(dim_start, dim_middle, bias=biases[0])
        if self.type_norm == "rms":
            self.normalization = nn.RMSNorm(dim_middle, eps=1e-6)
        elif self.type_norm == "layer":
            self.normalization = nn.LayerNorm(
                dim_middle, eps=1e-6, bias=norm_bias)
        elif self.type_norm == "batch":
            self.normalization = nn.BatchNorm1d(
                dim_middle,
                eps=1e-6, momentum=0.1, affine=True,
                track_running_stats=False
            )
        self.projection = nn.Linear(dim_middle, dim_end, bias=biases[1])

        with torch.no_grad():
            nn.init.xavier_uniform_(self.first.weight)
            nn.init.xavier_uniform_(self.projection.weight)
            if biases[0]:  nn.init.zeros_(self.first.bias)
            if biases[1]:  nn.init.zeros_(self.projection.bias)

        self.projection = nn.utils.parametrizations.spectral_norm(
            module=self.projection, n_power_iterations=pow_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        z_i = self.first(x)
        if self.type_norm != "":  z_i = self.normalization(z_i)
        z_f = self.projection(z_i)

        return z_f
