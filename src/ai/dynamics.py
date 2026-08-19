
from math import sqrt
import torch
import torch.nn as nn
import torchsde


class Dynamics(torchsde.SDEIto):
    def __init__(
        self,
        drift_size: int, diffusion_size: int,
        dim_embedding: int, dim_middle: int,
        backwards: bool
    ) -> None:
        super().__init__(noise_type="diagonal")

        self.backwards = backwards

        self.drift_size = drift_size if drift_size in [2, 3, 4] else 2
        self.diffusion_size = \
            diffusion_size if diffusion_size in list(range(7)) else 0

        f0 = nn.Linear(dim_embedding, dim_middle)
        fm = [nn.Linear(dim_middle, dim_middle)
            for _ in range(self.drift_size - 2)]
        f1 = nn.Linear(dim_middle, dim_embedding)
        with torch.no_grad():
            nn.init.kaiming_normal_(f0.weight)
            nn.init.zeros_(f0.bias)
            for l in fm:
                nn.init.kaiming_normal_(l.weight)
                nn.init.zeros_(l.bias)
            nn.init.zeros_(f1.weight)
            nn.init.zeros_(f1.bias)
        self.f_linears = nn.ModuleList(([f0] + fm + [f1]))

        if self.diffusion_size > 1:
            g0 = nn.Linear(dim_embedding, dim_middle)
            gm = [nn.Linear(dim_middle, dim_middle)
                for _ in range(self.diffusion_size - 2)]
            g1 = nn.Linear(dim_middle, dim_embedding)
            with torch.no_grad():
                nn.init.kaiming_normal_(g0.weight)
                nn.init.zeros_(g0.bias)
                for l in gm:
                    nn.init.kaiming_normal_(l.weight)
                    nn.init.zeros_(l.bias)
                nn.init.zeros_(g1.weight)
                nn.init.zeros_(g1.bias)
            self.g_linears = nn.ModuleList(([g0] + gm + [g1]))
        else:
            self.g_linears = \
                nn.ModuleList([nn.Linear(dim_embedding, dim_embedding)])

    def f(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        drift = nn.functional.gelu(self.f_linears[0](h))
        for layer in self.f_linears[1:-1]:
            drift = nn.functional.gelu(layer(drift))
        drift = torch.tanh(self.f_linears[-1](drift))

        return drift

    def g(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        if self.diffusion_size == 0:
            diffusion = torch.zeros_like(h)
        else:
            diffusion = nn.functional.gelu(self.g_linears[0](h))
            for layer in self.g_linears[1:-1]:
                diffusion = nn.functional.gelu(layer(diffusion))
            diffusion = torch.tanh(self.g_linears[-1](diffusion))

        return diffusion
