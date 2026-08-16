
import torch
import torch.nn as nn

from ai.constants import action_dim
from ai.custom_linear import LLinear
from data.constants import shape_players


class Decoder(nn.Module):
    def __init__(
        self,
        player_count: int,
        trajectory_length: int,
        dim_input: int,
        min_stdv: float, max_stdv: float,
        max_deltas: list[float],
        pow_iters: int,
        backwards: bool
    ) -> None:
        super().__init__()

        self.backwards = backwards

        assert isinstance(player_count, int)
        assert player_count == (shape_players // 2)
        self.player_count = player_count

        assert isinstance(trajectory_length, int) and trajectory_length > 3
        self.trajectory_length = trajectory_length

        self.register_buffer("max_delta", torch.tensor(max_deltas))

        std_init = 0.00
        assert max_stdv > min_stdv and max_stdv <= 1e-1 and min_stdv >= 1e-5
        self.min_stdv = min_stdv
        self.max_stdv = max_stdv

        self.output_projs = nn.ModuleDict({
            "def_mean": nn.Linear(
                in_features=dim_input, out_features=action_dim,
                bias=True
            ),
            "def_stdv": nn.Linear(
                in_features=dim_input, out_features=action_dim,
                bias=True
            ),
            "off_mean": nn.Linear(
                in_features=dim_input, out_features=action_dim,
                bias=True
            ),
            "off_stdv": nn.Linear(
                in_features=dim_input, out_features=action_dim,
                bias=True
            )
        })

        with torch.no_grad():
            nn.init.kaiming_normal_(
                self.output_projs["def_mean"].weight, nonlinearity="relu")
            nn.init.kaiming_normal_(
                self.output_projs["off_mean"].weight, nonlinearity="relu")
            self.output_projs["def_mean"].weight.mul_(1e-3)
            self.output_projs["off_mean"].weight.mul_(1e-3)
            nn.init.zeros_(self.output_projs["def_mean"].bias)
            nn.init.zeros_(self.output_projs["off_mean"].bias)
            nn.init.kaiming_normal_(
                self.output_projs["def_stdv"].weight, nonlinearity="relu")
            nn.init.kaiming_normal_(
                self.output_projs["off_stdv"].weight, nonlinearity="relu")
            self.output_projs["def_stdv"].weight.mul_(1e-3)
            self.output_projs["off_stdv"].weight.mul_(1e-3)
            nn.init.constant_(self.output_projs["def_stdv"].bias, std_init)
            nn.init.constant_(self.output_projs["off_stdv"].bias, std_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        bs = x.shape[0] // \
            (2 * (self.trajectory_length - 2) * (self.player_count * 1))
        x = x.reshape(
            2, (self.trajectory_length - 2), bs, (self.player_count * 1), -1)
        x_shape = x.shape
        _, T, B, N, d = x_shape

        assert T > 2 and self.trajectory_length > 3
        assert N == self.player_count

        z_def, z_off = x.unbind(dim=0)

        means_def = \
            self.max_delta * torch.tanh(self.output_projs["def_mean"](z_def))
        means_off = \
            self.max_delta * torch.tanh(self.output_projs["off_mean"](z_off))

        stdvs_def = self.min_stdv + (
            (self.max_stdv - self.min_stdv) *
            torch.sigmoid(self.output_projs["def_stdv"](z_def))
        )
        stdvs_off = self.min_stdv + (
            (self.max_stdv - self.min_stdv) *
            torch.sigmoid(self.output_projs["off_stdv"](z_off))
        )

        means_r_def = means_def.reshape(T, B, N, -1)
        means_r_off = means_off.reshape(T, B, N, -1)
        stdvs_r_def = stdvs_def.reshape(T, B, N, -1)
        stdvs_r_off = stdvs_off.reshape(T, B, N, -1)

        means = torch.stack([means_r_def, means_r_off], dim=0)
        stdvs = torch.stack([stdvs_r_def, stdvs_r_off], dim=0)
        means_r = \
            means.permute(1, 2, 0, 3, 4).contiguous().reshape(T, B, N*2, -1)
        stdvs_r = \
            stdvs.permute(1, 2, 0, 3, 4).contiguous().reshape(T, B, N*2, -1)

        z0_f = torch.stack([means_r, stdvs_r], dim=0)
        z_f = z0_f.reshape(2, T*B, 2, N, -1)

        return z_f
