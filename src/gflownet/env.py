
from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from typing import Optional, Tuple

from gflownet.actions import Actions
from gflownet.states import States


class Env(ABC):
    def __init__(
        self,
        s0: torch.Tensor,
        state_shape: Tuple,
        action_shape: Tuple,
        dummy_action: torch.Tensor,
        exit_action: torch.Tensor,
        sf: Optional[torch.Tensor],
    ) -> None:
        if isinstance(s0.device, str):  s0.device = torch.device(s0.device)
        assert isinstance(s0.device, torch.device)

        self.s0 = s0
        self.sf = sf.to(s0.device)

        assert self.s0.shape == self.sf.shape == state_shape

        self.state_shape = state_shape
        self.action_shape = action_shape

        self.dummy_action = dummy_action.to(s0.device)
        self.exit_action = exit_action.to(s0.device)

        self.States = self.make_states_class()
        self.Actions = self.make_actions_class()

    @property
    def device(self) -> torch.device:

        return self.s0.device

    def states_from_tensor(self, tensor: torch.Tensor) -> States:

        return self.States(tensor)

    def states_from_batch_shape(
        self, batch_shape: Tuple, random: bool=False, sink: bool=False
    ) -> States:

        return self.States.from_batch_shape(
            batch_shape, random=random, sink=sink, device=self.device)

    def actions_from_tensor(self, tensor: torch.Tensor) -> Actions:

        return self.Actions(tensor)

    def actions_from_batch_shape(self, batch_shape: Tuple) -> Actions:

        return self.Actions.make_dummy_actions(
            batch_shape, device=self.device)

    @abstractmethod
    def step(self, states: States, actions: Actions) -> States:

        pass

    @abstractmethod
    def backward_step(self, states: States, actions: Actions) -> States:

        pass

    @abstractmethod
    def is_action_valid(
        self, states: States, actions: Actions, backward: bool=False
    ) -> bool:

        pass

    @abstractmethod
    def make_random_states_tensor(
        self, batch_shape: Tuple, device: torch.device | None=None
    ) -> States:

        pass

    def make_states_class(self) -> type[States]:
        env = self

        class DefaultEnvState(States):
            state_shape = env.state_shape
            s0 = env.s0
            sf = env.sf
            make_random_states_tensor = env.make_random_states_tensor

        return DefaultEnvState

    def make_actions_class(self) -> type[Actions]:
        env = self

        class DefaultEnvAction(Actions):
            action_shape = env.action_shape
            dummy_action = env.dummy_action
            exit_action = env.exit_action

        return DefaultEnvAction

    def reset(
        self,
        batch_shape: int | Tuple[int, ...] | list[int],
        random: bool=False, sink: bool=False
    ) -> States:
        if isinstance(batch_shape, int):  batch_shape = (batch_shape,)
        elif isinstance(batch_shape, list):  batch_shape = tuple(batch_shape)

        return self.states_from_batch_shape(
            batch_shape=batch_shape, random=random, sink=sink
        )

    def _step(self, states: States, actions: Actions) -> States:
        assert states.batch_shape == actions.batch_shape
        assert len(states.batch_shape) == 1

        valid_states_idx: torch.Tensor = ~states.is_sink_state
        assert valid_states_idx.shape == states.batch_shape
        assert valid_states_idx.dtype == torch.bool

        new_valid_states_idx = valid_states_idx & ~actions.is_exit

        not_done_states = states[new_valid_states_idx].clone()
        not_done_actions = actions[new_valid_states_idx]

        not_done_states = self.step(not_done_states, not_done_actions)
        assert isinstance(not_done_states, States)

        new_states = self.States.make_sink_states(
            states.batch_shape, device=states.device)
        new_states[new_valid_states_idx] = not_done_states

        return new_states

    def _backward_step(self, states: States, actions: Actions) -> States:
        assert states.batch_shape == actions.batch_shape

        new_states = states.clone()

        valid_states_idx: torch.Tensor = ~new_states.is_initial_state
        assert valid_states_idx.shape == new_states.batch_shape
        assert valid_states_idx.dtype == torch.bool
        valid_actions = actions[valid_states_idx]
        valid_states = new_states[valid_states_idx]

        new_states[valid_states_idx] = \
            self.backward_step(valid_states, valid_actions)

        return new_states

    @abstractmethod
    def reward(self, states: States) -> torch.Tensor:

        pass
