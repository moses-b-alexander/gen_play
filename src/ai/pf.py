
import torch
from typing import Callable, TypeVar

from ai.constants import model_name
from ai.decoder import *
from ai.diff_eq import *
from ai.encoder import *
from data.play_gfn import PlayStates
from gflownet.estimators import Estimator


class PF(Estimator):
    def __init__(
        self,
        player_count: int, trajectory_length: int,
        backwards: bool,
        dim_start: int, dim_hidden: int, dim_end: int,
        pow_iters: int,
        encoder_hps: dict, diff_eq_hps: dict, decoder_hps: dict,
        noise_floor: float, noise_ceiling: float, noise_gap: int,
        noise_exp: float,
        prior_means: tuple[tuple[float | list[float]]],
        prior_stdvs: tuple[tuple[float | list[float]]],
        device: torch.device
    ) -> None:
        super().__init__(is_backward=backwards)

        self.device = device
        self.to(self.device)

        self.name = model_name

        self.register_buffer("global_step", torch.tensor(0, dtype=torch.long))

        self.player_count = player_count
        self.trajectory_length = trajectory_length

        self.backwards = backwards

        self.dim_start = dim_start
        self.dim_hidden = dim_hidden
        self.dim_end = dim_end

        self.pow_iters = pow_iters

        self.encoder = Encoder(
            dim_output=self.dim_hidden,
            **encoder_hps,
            pow_iters=self.pow_iters,
            backwards=self.backwards
        )
        self.diff_eq = DiffEq(
            dim_embedding=self.dim_hidden,
            **diff_eq_hps,
            backwards=self.backwards
        )
        self.decoder = Decoder(
            dim_input=self.dim_hidden,
            **decoder_hps,
            pow_iters=self.pow_iters,
            backwards=self.backwards
        )

        if noise_gap >= 1 and noise_gap <= 1e9 and isinstance(noise_gap, int):
            self.noise_gap = noise_gap
        else:
            self.noise_gap = 2
        self.noise_floor = noise_floor if noise_floor > 0.00 else 1e-9
        self.noise_ceiling = noise_ceiling if noise_ceiling < 1.00 else 1e-1
        if self.noise_floor >= self.noise_ceiling:
            self.noise_floor = 1e-9
            self.noise_ceiling = 1e-1
        self.noise_exp = \
            noise_exp if noise_exp >= 0.5 and noise_exp <= 2.0 else 1.0

        prior_means_dx = torch.cat([
            torch.tensor(([prior_means[0][0]] * player_count)) \
                if isinstance(prior_means[0][0], float)
                else torch.tensor(prior_means[0][0]),
            torch.tensor(([prior_means[0][1]] * player_count)) \
                if isinstance(prior_means[0][1], float)
                else torch.tensor(prior_means[0][1]),
        ], dim=0)
        prior_means_dy = torch.cat([
            torch.tensor(([prior_means[1][0]] * player_count)) \
                if isinstance(prior_means[1][0], float)
                else torch.tensor(prior_means[1][0]),
            torch.tensor(([prior_means[1][1]] * player_count)) \
                if isinstance(prior_means[1][1], float)
                else torch.tensor(prior_means[1][1]),
        ], dim=0)
        self.prior_means = torch.stack(
            [prior_means_dx, prior_means_dy], dim=-1
        ).to(self.device)

        prior_stdvs_dx = torch.cat([
            torch.tensor(([prior_stdvs[0][0]] * player_count)) \
                if isinstance(prior_stdvs[0][0], float)
                else torch.tensor(prior_stdvs[0][0]),
            torch.tensor(([prior_stdvs[0][1]] * player_count)) \
                if isinstance(prior_stdvs[0][1], float)
                else torch.tensor(prior_stdvs[0][1]),
        ], dim=0)
        prior_stdvs_dy = torch.cat([
            torch.tensor(([prior_stdvs[1][0]] * player_count)) \
                if isinstance(prior_stdvs[1][0], float)
                else torch.tensor(prior_stdvs[1][0]),
            torch.tensor(([prior_stdvs[1][1]] * player_count)) \
                if isinstance(prior_stdvs[1][1], float)
                else torch.tensor(prior_stdvs[1][1]),
        ], dim=0)
        self.prior_stdvs = torch.stack(
            [prior_stdvs_dx, prior_stdvs_dy], dim=-1
        ).to(self.device)

        self.to(self.device)

    def set_step(self, step: int) -> None:
        self.global_step.fill_(step)

    def get_step(self) -> int:

        return int(self.global_step.item())

    def forward(self, states: PlayStates, training: bool) -> torch.Tensor:
        x = states.tensor.to(self.device)
        s = self.get_step()

        x_shape = x.shape
        T = self.trajectory_length - 2
        B = x_shape[0] // (T + 1)
        N = self.player_count * 2
        x_flat = x.reshape(-1, self.dim_start)

        h0 = self.encoder(x_flat)

        hT = self.diff_eq(h0)

        if training:
            n_scale = ((hT.detach()).std(dim=-1, keepdim=True))
            n_step = self.noise_floor + (
                (self.noise_ceiling - self.noise_floor) /
                ((s + self.noise_gap) ** self.noise_exp)
            )
            noise = torch.randn_like(hT) * n_scale * n_step
        else:
            noise = torch.zeros_like(hT)

        hT_sum = hT + noise + 1e-12

        hf = self.decoder(hT_sum)

        return hf

    def to_probability_distribution(
        self,
        states: PlayStates,
        estimator_outputs: torch.Tensor | None,
        inference: bool=False
    ) -> torch.distributions.Independent:
        if states.tensor.size(0) == self.trajectory_length:
            states = PlayStates(states.tensor[:-1, ...])

        if estimator_outputs is None:
            if inference:
                with torch.inference_mode():
                    params_0 = self.forward(states, False)
            else:
                params_0 = self.forward(states, True)
        else:
            params_0 = estimator_outputs.to(self.device)

        means, stdvs = params_0.unbind(dim=0)

        bs = means.size(0) // (self.trajectory_length - 2)
        T, B, N, d = \
            self.trajectory_length-2, bs, self.player_count*1, means.size(-1)

        means_r = means.reshape(T, B, N*2, -1)
        prior_means_r1 = self.prior_means.unsqueeze(0).expand(B, -1, -1)
        prior_means_r = prior_means_r1.unsqueeze(0)
        means_c = torch.cat([prior_means_r, means_r], dim=0)
        means_f = means_c.reshape((T+1)*B, N*2, -1)

        stdvs_r = stdvs.reshape(T, B, N*2, -1)
        prior_stdvs_r1 = self.prior_stdvs.unsqueeze(0).expand(B, -1, -1)
        prior_stdvs_r = prior_stdvs_r1.unsqueeze(0)
        stdvs_c = torch.cat([prior_stdvs_r, stdvs_r], dim=0)
        stdvs_f = stdvs_c.reshape((T+1)*B, N*2, -1)

        base_dist = torch.distributions.Normal(means_f, stdvs_f)
        final_dist = torch.distributions.Independent(base_dist, 1)

        return final_dist
