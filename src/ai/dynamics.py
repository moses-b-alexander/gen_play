
from math import sqrt
import torch
import torch.nn as nn
import torchsde


class Dynamics(torchsde.SDEIto):
    def __init__(
        self,
        drift_size: int, diffusion_size: int,
        dim_embedding: int, dim_middle: int,
        scale: float,
        backwards: bool
    ) -> None:
        super().__init__(noise_type="diagonal")

        self.backwards = backwards

        self.drift_size = drift_size if drift_size in [2, 3, 4] else 2
        self.diffusion_size = \
            diffusion_size if diffusion_size in list(range(6)) else 0
        self.scale = 1e-3 if scale < 1e-5 or scale > 1e-1 else scale

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
        h = h

        drift = nn.functional.gelu(self.f_linears[0](h))
        for layer in self.f_linears[1:-1]:
            drift = nn.functional.gelu(layer(drift))
        drift = torch.tanh((0.50 * self.f_linears[-1](drift))) * 0.50

        return drift

    def g(self, t: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        h = h

        if self.diffusion_size == 0:
            diffusion = torch.zeros_like(h)
        elif self.diffusion_size == 1:
            diffusion = torch.full_like(h, fill_value=self.scale)
        else:
            diffusion = nn.functional.gelu(self.g_linears[0](h))
            for layer in self.g_linears[1:-1]:
                diffusion = nn.functional.gelu(layer(diffusion))
            diffusion = torch.tanh(self.g_linears[-1](diffusion))

        diffusion_fwd = torch.tensor((0 + t.item()) * sqrt(self.scale))
        diffusion_bwd = torch.tensor((1 - t.item()) * sqrt(self.scale))
        diffusion_sch = diffusion_fwd
        # diffusion_sch = diffusion_fwd if not self.backwards else diffusion_bwd
        # diffusion_sch = diffusion_fwd if self.backwards else diffusion_bwd
        # diffusion_sch = diffusion_bwd
        noise_scale = self.scale + torch.sigmoid(diffusion_sch).item()

        if noise_scale >= 1.00:  noise_scale = 1.00
        if noise_scale <= -1.00:  noise_scale = -1.00
        diffusion_scaled = noise_scale * diffusion

        return diffusion_scaled
