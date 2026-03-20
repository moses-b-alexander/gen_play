
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Optional

from gflownet.states import States


class Estimator(ABC, nn.Module):
    def __init__(self, module: nn.Module, is_backward: bool=False) -> None:
        nn.Module.__init__(self)

        self.module = module
        self.is_backward = is_backward

    @abstractmethod
    def forward(self, states: States) -> torch.Tensor:

        pass

    @abstractmethod
    def to_probability_distribution(
        self,
        states: States,
        estimator_output: torch.Tensor | None,
        **policy_kwargs: Any,
    ) -> torch.distributions.Distribution:

        pass
