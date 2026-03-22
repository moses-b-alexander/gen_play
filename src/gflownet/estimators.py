
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, Optional

from gflownet.states import States


class Estimator(ABC, nn.Module):
    def __init__(self, is_backward: bool) -> None:
        nn.Module.__init__(self)

        self.is_backward = is_backward

    @abstractmethod
    def forward(self, states: States) -> torch.Tensor:

        pass

    @abstractmethod
    def to_probability_distribution(
        self,
        states: States,
        estimator_outputs: torch.Tensor | None,
        **policy_kwargs: Any,
    ) -> torch.distributions.Distribution:

        pass
