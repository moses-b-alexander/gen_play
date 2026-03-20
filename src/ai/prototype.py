
import torch
import torch.nn as nn

from ai.perceptronner import Perceptronner
from data.play_gfn import PlayStates


class PrototypeModel(nn.Module):
    def __init__(
        self,
        dim_global: int, dim_agent: int, dim_output: int,
        dim_prj_play: int, dim_lin_play: int, dropout_play: float,
        dim_prj_player: int, dim_lin_player: int, dropout_player: float,
        dim_prj_fused: int, dim_lin_fused: int, dropout_fused: float,
        beta_sp: float, epsilon_sp: float,
        device: torch.device
    ) -> None:
        super().__init__()

        self.device = device
        self.to(self.device)

        self.dim_global = dim_global
        self.dim_agent = dim_agent
        self.dim_output = dim_output

        self.beta_sp = beta_sp
        self.epsilon_sp = epsilon_sp

        self.play_mlp = Perceptronner(
            dim_input=dim_global,
            dim_projection=dim_prj_play,
            dim_output=dim_lin_play,
            norm=True, drop=dropout_play,
            project_init="kaiming", embed_init="kaiming",
            device=self.device
        )

        self.player_mlps = nn.ModuleList([
            Perceptronner(
                dim_input=dim_agent,
                dim_projection=dim_prj_player,
                dim_output=dim_lin_player,
                norm=True, drop=dropout_player,
                project_init="kaiming", embed_init="kaiming",
                device=self.device
            )
        ])

        self.b_prj_play_fused = nn.Linear(dim_lin_play, dim_prj_fused)
        nn.init.xavier_uniform_(self.b_prj_play_fused.weight)
        nn.init.constant_(self.b_prj_play_fused.bias, 0.0)

        self.b_prj_player_fused = nn.Linear(dim_lin_player, dim_prj_fused)
        nn.init.xavier_uniform_(self.b_prj_play_fused.weight)
        nn.init.constant_(self.b_prj_player_fused.bias, 0.0)

        self.b_mlp = Perceptronner(
            dim_input=dim_prj_fused,
            dim_projection=dim_prj_fused,
            dim_output=dim_lin_fused,
            norm=False, drop=dropout_fused,
            project_init="kaiming", embed_init="kaiming",
            device=self.device
        )

        self.prj_output = nn.Linear(dim_lin_fused, dim_output)

    def forward(self, states: PlayStates) -> torch.Tensor:
        x = states.tensor

        x_shape = x.shape
        if len(x_shape) == 3:
            TB, N, d = x_shape
            T = TB // batch_size
            B = batch_size * 1
        if len(x_shape) == 4:
            B, T, N, d = x_shape
        x = x.reshape(T, B, N, d)

        d_play = x[0, :, 0, :self.dim_global]
        d_player = x[:, :, :, self.dim_global:]

        z_play_single = self.play_mlp(d_play)
        z_play = z_play_single.unsqueeze(0).expand(T, B, -1)
        z_play = z_play.unsqueeze(2).expand(T, B, N, -1)

        d_player_split = torch.unbind(d_player, dim=2)
        z_player_split = [
            mlp(x) for mlp, x in zip(self.player_mlps, d_player_split)
        ]
        z_player = torch.stack(z_player_split, dim=2)

        p_play = self.b_prj_play_fused(z_play)
        p_player = self.b_prj_player_fused(z_player)
        d_fused = p_play + (p_player / team_players)
        z_fused = self.b_mlp(d_fused)

        z_final = self.prj_output(z_fused)
        z_final = z_final.reshape(T * B, N, -1)

        return z_final

    def to_probability_distribution(
        self,
        states: PlayStates,
        estimator_outputs: torch.Tensor | None,
        inference: bool=False
    ) -> torch.distributions.Independent:
        if estimator_outputs is None:  params = self.forward(states)
        else:  params = estimator_outputs.to(self.device)

        mean, log_std = params.chunk(2, dim=-1)
        std = nn.functional.softplus(log_std + self.beta_sp) + self.epsilon_sp

        base_dist = torch.distributions.Normal(mean, std)
        final_dist = torch.distributions.Independent(base_dist, 1)

        return final_dist
