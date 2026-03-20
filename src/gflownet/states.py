
from __future__ import annotations

from abc import ABC
from math import prod
import numpy as np
import torch
from typing import ClassVar, List, Sequence


class States(ABC):
    state_shape: ClassVar[tuple[int, ...]]
    s0: ClassVar[torch.Tensor]
    sf: ClassVar[torch.Tensor]

    def __init__(self, tensor: torch.Tensor) -> None:
        assert self.s0.shape == self.state_shape
        assert self.sf.shape == self.state_shape
        assert tensor.shape[-len(self.state_shape):] == self.state_shape

        self.tensor = tensor

    @property
    def device(self) -> torch.device:

        return self.tensor.device

    @property
    def batch_shape(self) -> tuple[int, ...]:

        return tuple(self.tensor.shape)[:-len(self.state_shape)]

    @classmethod
    def from_batch_shape(
        self,
        batch_shape: int | tuple[int, ...],
        random: bool=False, sink: bool=False,
        device: torch.device | None=None
    ) -> States:
        if isinstance(batch_shape, int):  batch_shape = (batch_shape,)

        if random and sink:  sink = False

        if random:
            return self.make_random_states(batch_shape, device=device)
        elif sink:
            return self.make_sink_states(batch_shape, device=device)
        else:
            return self.make_initial_states(batch_shape, device=device)

    @classmethod
    def make_initial_states(
        self, batch_shape: tuple[int, ...], device: torch.device | None=None
    ) -> States:
        state_ndim = len(self.state_shape)
        device = self.s0.device if device is None else device

        return self.__class__(
            self.s0.repeat(*batch_shape, *((1,) * state_ndim)).to(device))

    @classmethod
    def make_sink_states(
        self, batch_shape: tuple[int, ...], device: torch.device | None=None
    ) -> States:
        state_ndim = len(self.state_shape)
        device = self.sf.device if device is None else device

        return self.__class__(
            self.sf.repeat(*batch_shape, *((1,) * state_ndim)).to(device))

    def __len__(self) -> int:

        return prod(self.batch_shape)

    def __getitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | Tensor
    ) -> States:

        return self.__class__(self.tensor[index])

    def __setitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | Tensor,
        states: States
    ) -> None:
        self.tensor[index] = states.tensor

    @classmethod
    def stack(self, states_list: List[States]) -> States:
        stacked_states = \
            torch.stack([states.tensor for states in states_list], dim=0)

        return self.__class__(stacked_states)

    def _compare(self, other: torch.Tensor) -> torch.Tensor:
        n_batch_dims = len(self.batch_shape)

        if n_batch_dims == 1:
            assert (other.shape == self.state_shape) or (
                other.shape == self.batch_shape + self.state_shape
            )
        else:
            assert (other.shape == self.batch_shape + self.state_shape)

        if self.tensor.device != other.device:
            other_d = other.to(self.tensor.device)
        else:
            other_d = other.to(other.device)

        out = torch.isclose(self.tensor, other_d)
        if len(self.__class__.state_shape) > 1:
            out = out.flatten(start_dim=n_batch_dims)
        out = out.all(dim=-1)
        assert out.shape == self.batch_shape

        return out

    @property
    def is_initial_state(self) -> torch.Tensor:
        if len(self.batch_shape) == 1:
            source_states_tensor = self.__class__.s0
        else:
            source_states_tensor = self.__class__.s0.repeat(
                *self.batch_shape, *((1,) * len(self.__class__.state_shape))
            )

        return self._compare(source_states_tensor)

    @property
    def is_sink_state(self) -> torch.Tensor:
        if len(self.batch_shape) == 1:
            sink_states_tensor = self.__class__.sf
        else:
            sink_states_tensor = self.__class__.sf.repeat(
                *self.batch_shape, *((1,) * len(self.__class__.state_shape))
            ).to(self.tensor.device)

        return self._compare(sink_states_tensor)

    def clone(self) -> States:

        return self.__class__(self.tensor.clone())

    def to_device(self, device: torch.device) -> States:
        self.s0 = self.s0.to(device)
        self.sf = self.sf.to(device)
        self.tensor = self.tensor.to(device)

        return self
