
from __future__ import annotations

import torch
from torch import Tensor
from typing import Sequence

from gflownet.actions import Actions
from gflownet.containers import Container
from gflownet.states import States


class Trajectories(Container):
    def __init__(
        self, states: States, actions: Actions, log_rewards: torch.Tensor
    ) -> None:
        self.states = states
        assert len(self.states.batch_shape) == 2

        self.actions = actions
        assert (
            self.actions.batch_shape[0] == self.states.batch_shape[0] - 1 and
            self.actions.batch_shape[1] == self.states.batch_shape[1]
        )

        self._log_rewards = log_rewards
        assert (
            self._log_rewards.shape == (self.n_trajectories,) and
            self._log_rewards.is_floating_point()
        )

    @property
    def device(self) -> torch.device:

        return self.states.device

    @property
    def n_trajectories(self) -> int:

        return self.states.batch_shape[1]

    def __len__(self) -> int:

        return self.n_trajectories

    @property
    def max_length(self) -> int:

        return self.actions.batch_shape[0]

    @property
    def log_rewards(self) -> torch.Tensor:

        return self._log_rewards

    def __getitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | Tensor
    ) -> Trajectories:
        states = self.states[:, index, ...]
        actions = self.actions[:, index, ...]
        log_rewards = self._log_rewards[index]

        return Trajectories(
            states=states, actions=actions, log_rewards=log_rewards)
