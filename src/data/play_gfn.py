
from __future__ import annotations

from math import log, sqrt
import torch
from torch import Tensor
from typing import Optional, Sequence

from ai.constants import action_dim, state_dim
from common.constants import dtyp
from common.devices import class_device
from data.constants import fstv, lstv, shape_players, state_mapping
from data.bridge import tensorize_df
from gflownet.actions import Actions
from gflownet.env import Env
from gflownet.containers import Container
from gflownet.preprocessors import Preprocessor
from gflownet.states import States


class PlayStates(States):
    state_shape = (shape_players, state_dim)
    s0 = torch.full(state_shape, fstv).to(class_device).float()
    sf = torch.full(state_shape, lstv).to(class_device).float()

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor.to(class_device)
        super().__init__(self.tensor)

    @classmethod
    def from_batch_shape(
        self,
        batch_shape: int | tuple[int, ...],
        random: bool=False, sink: bool=False,
        device: torch.device | None=class_device,
    ) -> PlayStates:
        if isinstance(batch_shape, int):
            batch_shape = (batch_shape,)
        elif isinstance(batch_shape, list):
            batch_shape = tuple(batch_shape)

        if random:
            t = self.make_random_states_tensor(batch_shape, device=device)
        elif sink:
            t = self.make_sink_states_tensor(batch_shape, device=device)
        else:
            t = self.make_initial_states_tensor(batch_shape, device=device)

        return PlayStates(t)

    @classmethod
    def make_initial_states_tensor(
        self,
        batch_shape: tuple[int, ...],
        device: torch.device | None=class_device
    ) -> torch.Tensor:
        device = self.tensor.device if device is None else device

        t = self.s0.repeat(
            *batch_shape, *((1,) * len(self.state_shape))
        ).to(device)

        return t

    @classmethod
    def make_random_states_tensor(
        self, batch_shape: tuple[int, ...]
    ) -> torch.Tensor:
        device = self.tensor.device if device is None else device

        random_tensors = []
        for i in range(self.state_shape[1]):
            tup = state_mapping[i]
            if tup[0] == "bernoulli":
                rt = torch.randint(
                    low=0, high=2,
                    size=(*batch_shape, self.state_shape[0], 1),
                    dtype=torch.float32
                )
            if tup[0] == "float":
                if tup[4] == "uniform":
                    tup_low = tup[2] - 1e-6
                    tup_high = tup[3] + 1e-6
                    rt = torch.empty(
                        (*batch_shape, self.state_shape[0], 1),
                        dtype=torch.float32
                    ).uniform_(tup_low, tup_high)
                if tup[4] == "normal":
                    tup_mean = (tup[2] + tup[3]) / 2
                    tup_std = sqrt(((tup[3] - tup[2]) ** 2) / 12)
                    rt = torch.empty(
                        (*batch_shape, self.state_shape[0], 1),
                        dtype=torch.float32
                    ).normal_(tup_mean, tup_std)
                # rt = rt.clamp(min=tup[2], max=tup[3])
            random_tensors.append(rt)
        t = torch.cat(random_tensors, dim=-1).to(device)

        return t

    @classmethod
    def make_sink_states_tensor(
        self,
        batch_shape: tuple[int, ...],
        device: torch.device | None=class_device
    ) -> torch.Tensor:
        device = self.tensor.device if device is None else device

        t = self.sf.repeat(
            *batch_shape, *((1,) * len(self.state_shape))
        ).to(device)

        return t

    @classmethod
    def stack(self, states: list[PlayStates]) -> PlayStates:
        st = torch.stack(
            [s.tensor for s in states], dim=0
        ).to(self.tensor.device)

        return PlayStates(st)

class PlayActions(Actions):
    action_shape = (shape_players, action_dim)
    dummy_action = torch.full(action_shape, (fstv-1)).to(class_device).float()
    exit_action = torch.full(action_shape, (lstv+1)).to(class_device).float()

    def __init__(self, tensor: torch.Tensor) -> None:
        self.tensor = tensor.to(class_device)
        super().__init__(self.tensor)

    @classmethod
    def make_dummy_actions(
        self,
        batch_shape: tuple[int, ...],
        device: torch.device | None=class_device
    ) -> PlayActions:
        device = self.tensor.device if device is None else device

        t = self.dummy_action.repeat(
            *batch_shape, *((1,) * len(self.action_shape))
        ).to(device)

        return PlayActions(t)

    @classmethod
    def make_exit_actions(
        self,
        batch_shape: tuple[int, ...],
        device: torch.device | None=class_device
    ) -> PlayActions:
        device = self.tensor.device if device is None else device

        t = self.exit_action.repeat(
            *batch_shape, *((1,) * len(self.action_shape))
        ).to(device)

        return PlayActions(t)

    @classmethod
    def stack(self, actions: list[Actions]) -> PlayActions:
        st = torch.stack(
            [a.tensor for a in actions], dim=0
        ).to(self.tensor.device)

        return PlayActions(st)

class PlayContainer(Container):
    def __init__(
        self, states: PlayStates, actions: PlayActions,
        rewards: torch.Tensor, padding_mask: torch.Tensor,
        is_initial: torch.Tensor, is_terminal: torch.Tensor,
        test_mask: torch.Tensor, play_index: torch.Tensor
    ) -> None:
        self.states = states
        self.actions = actions
        self.rewards = rewards
        self.padding_mask = padding_mask
        self.is_initial = is_initial
        self.is_terminal = is_terminal
        self.test_mask = test_mask
        self.play_index = play_index

    def __len__(self) -> int:

        return int(self.states.tensor.size(0))

    def __getitem__(
        self,
        idx: int | slice | tuple | Sequence[int] | Sequence[bool] | Tensor
    ) -> PlayContainer:

        return PlayContainer(
            self.states[idx], self.actions[idx], self.rewards[idx],
            self.padding_mask[idx],
            self.is_initial[idx], self.is_terminal[idx],
            self.test_mask[idx], self.play_index[idx]
        )

    def log_rewards(self) -> torch.Tensor:

        return torch.log(self.rewards)

    def device(self) -> torch.device:

        return self.states.tensor.device

class PlayPreprocessor(Preprocessor):
    def __init__(self, output_dim: int) -> None:
        super().__init__(output_dim)

    def preprocess(states: PlayStates) -> torch.Tensor:
        t = states.tensor.float()

        return t

class PlayEnv(Env):
    def __init__(
        self, dfs: list[pd.DataFrame], preprocessor: PlayPreprocessor
    ) -> None:
        super().__init__(
            s0=PlayStates.s0,
            state_shape=PlayStates.state_shape,
            action_shape=PlayActions.action_shape,
            dummy_action=PlayActions.dummy_action,
            exit_action=PlayActions.exit_action,
            sf=PlayStates.sf,
        )
        self.preprocessor = preprocessor

        self.containers = []
        total_reward = dtyp(0.0)
        c = -1
        for df in dfs:
            trjs = df.groupby(level=0, sort=False)
            for _, trj in trjs:
                c += 1
                s, a, r, m, i = tensorize_df(trj, c, class_device)
                container = PlayContainer(
                    states=PlayStates(s),
                    actions=PlayActions(a),
                    rewards=r,
                    padding_mask=m[:, 1],
                    is_initial=m[:, 2],
                    is_terminal=m[:, 3],
                    test_mask=m[:, 4],
                    play_index=i
                )
                total_reward += dtyp(
                    container.rewards[
                        container.is_terminal.bool()
                    ].sum().item()
                )
                self.containers.append(container)
        self.log_z = dtyp(log((total_reward + 1e-12)))

    def step(
        self, states: PlayStates, actions: PlayActions
    ) -> torch.Tensor:

        return self.sf.clone().detach()

    def backward_step(
        self, states: PlayStates, actions: PlayActions
    ) -> torch.Tensor:

        return self.s0.clone().detach()

    def is_action_valid(
        self, states: PlayStates, actions: PlayActions, backward: bool=False
    ) -> bool:

        return True

    def make_random_states_tensor(
        self, batch_shape: tuple[int, ...]
    ) -> torch.Tensor:

        return PlayStates.make_random_states_tensor(batch_shape)

    def make_states_class(self) -> type[PlayStates]:
        env = self

        class DefaultStates(PlayStates):
            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            make_random_states_tensor = env.make_random_states_tensor

        return DefaultStates

    def make_actions_class(self) -> type[PlayActions]:
        env = self

        class DefaultActions(PlayActions):
            action_shape = env.action_shape
            dummy_action = env.dummy_action
            exit_action = env.exit_action

        return DefaultActions

    def reset(
        self,
        batch_shape: int | tuple[int, ...] | list[int],
        random: bool=False, sink: bool=False
    ) -> PlayStates:

        return (
            PlayStates(torch.zeros(batch_shape, device=self.s0.tensor.device))
        )

    def reward(self, states: PlayStates) -> torch.Tensor:

        return torch.zeros(states.batch_shape, device=states.tensor.device)

    @property
    def log_partition(self) -> dtyp:

        return self.log_z
