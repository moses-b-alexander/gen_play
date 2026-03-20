
import torch
import torch.nn as nn

from common.constants import team_players


class IndexEncoder(nn.Module):
    def __init__(
        self,
        dim_idx: int,
        pow_iters: int
    ) -> None:
        super().__init__()

        if dim_idx <= team_players:  dim_index = 16
        else:  dim_index = dim_idx * 1

        W0 = torch.randn((dim_index, dim_index))
        Q, _ = torch.linalg.qr(W0)
        W = Q[:team_players]

        self.idx_encoder = nn.Embedding(team_players, dim_index)

        with torch.no_grad():
            self.idx_encoder.weight.copy_(W.view_as(self.idx_encoder.weight))

        self.idx_encoder = nn.utils.parametrizations.spectral_norm(
            module=self.idx_encoder, n_power_iterations=pow_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        z_f = self.idx_encoder(torch.arange(x.size(2) // 2, device=x.device))

        return z_f
