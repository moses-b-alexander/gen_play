
from __future__ import annotations

from abc import ABC
from math import prod
import torch
from torch import Tensor
from typing import ClassVar, List, Sequence


class Actions(ABC):
    action_shape: ClassVar[tuple[int, ...]]
    dummy_action: ClassVar[torch.Tensor]
    exit_action: ClassVar[torch.Tensor]

    def __init__(self, tensor: torch.Tensor) -> None:
        assert tensor.shape[-len(self.action_shape):] == self.action_shape

        self.tensor = tensor

    @property
    def device(self) -> torch.device:

        return self.tensor.device

    @property
    def batch_shape(self) -> tuple[int, ...]:

        return tuple(self.tensor.shape)[: -len(self.action_shape)]

    @classmethod
    def make_dummy_actions(
        self, batch_shape: tuple[int, ...], device: torch.device | None=None
    ) -> Actions:
        action_ndim = len(self.action_shape)
        device = self.dummy_action.device if device is None else device

        return self.__class__(self.dummy_action.repeat(
            *batch_shape, *((1,) * action_ndim)
        ).to(device))

    @classmethod
    def make_exit_actions(
        self, batch_shape: tuple[int, ...], device: torch.device | None=None
    ) -> Actions:
        action_ndim = len(self.action_shape)
        device = self.exit_action.device if device is None else device

        return self.__class__(self.exit_action.repeat(
            *batch_shape, *((1,) * action_ndim)
        ).to(device))

    def __len__(self) -> int:

        return prod(self.batch_shape)

    def __getitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | Tensor
    ) -> Actions:
        actions = self.tensor[index]

        return self.__class__(actions)

    def __setitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | Tensor,
        actions: Actions
    ) -> None:
        self.tensor[index] = actions.tensor

    @classmethod
    def stack(self, actions_list: List[Actions]) -> Actions:
        stacked_actions = \
            torch.stack([actions.tensor for actions in actions_list], dim=0)

        return self.__class__(stacked_actions)

    def _compare(self, other: torch.Tensor) -> torch.Tensor:
        n_batch_dims = len(self.batch_shape)

        if n_batch_dims == 1:
            assert (other.shape == self.action_shape) or (
                other.shape == self.batch_shape + self.action_shape
            )
        else:
            assert (other.shape == self.batch_shape + self.action_shape)

        if self.tensor.device != other.device:
            other_d = other.to(self.tensor.device)
        else:
            other_d = other.to(other.device)

        out = torch.isclose(self.tensor, other_d)
        if len(self.action_shape) > 1:
            out = out.flatten(start_dim=n_batch_dims)
        out = out.all(dim=-1)
        assert out.shape == self.batch_shape

        return out

    @property
    def is_dummy(self) -> torch.Tensor:
        if len(self.batch_shape) == 1:
            dummy_actions_tensor = self.__class__.dummy_action
        else:
            dummy_actions_tensor = self.__class__.dummy_action.repeat(
                *self.batch_shape, *((1,) * len(self.__class__.action_shape))
            )

        return self._compare(dummy_actions_tensor)

    @property
    def is_exit(self) -> torch.Tensor:
        if len(self.batch_shape) == 1:
            exit_actions_tensor = self.__class__.exit_action
        else:
            exit_actions_tensor = self.__class__.exit_action.repeat(
                *self.batch_shape, *((1,) * len(self.__class__.action_shape))
            )

        return self._compare(exit_actions_tensor)

    def to_device(self, device: torch.device) -> Actions:
        self.dummy_action = self.dummy_action.to(device)
        self.exit_action = self.exit_action.to(device)
        self.tensor = self.tensor.to(device)

        return self
