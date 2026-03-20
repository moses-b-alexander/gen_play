
from __future__ import annotations

from abc import ABC, abstractmethod
import torch
from torch import Tensor
from typing import Sequence


class Container(ABC):
    @abstractmethod
    def __len__(self) -> int:

        pass

    @abstractmethod
    def __getitem__(
        self,
        index: int | slice | tuple | Sequence[int] | Sequence[bool] | Tensor
    ) -> Container:

        pass

    def sample(self, n_samples: int) -> Container:

        return self[torch.randperm(len(self))[:n_samples]]

    @property
    @abstractmethod
    def log_rewards(self) -> torch.Tensor:

        pass

    @property
    @abstractmethod
    def device(self) -> torch.device:

        pass
