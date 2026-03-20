
from __future__ import annotations

from abc import ABC, abstractmethod
import torch

from gflownet.states import States


class Preprocessor(ABC):
    def __init__(self, output_dim: int) -> None:
        self.output_dim = output_dim

    @abstractmethod
    def preprocess(self, states: States) -> torch.Tensor:

        pass
