
import torch
import torch.nn as nn

from ai.constants import agent_dim, player_tensor_indices
from ai.normalizer import Normalizer


class PlayerNormalizer(nn.Module):
    def __init__(
        self,
        type_feature: str,
        dim_start: int, dim_end: int,
        pow_iters: int
    ) -> None:
        super().__init__()

        self.type_feature = type_feature.lower()
        self.type_norm = "layer" if self.type_feature != "role" else ""

        self.normalizer = Normalizer(
            type_norm=self.type_norm,
            dim_start=len(player_tensor_indices[self.type_feature]),
            dim_middle=dim_start,
            dim_end=dim_end,
            biases=(True, False), norm_bias=False,
            pow_iters=pow_iters
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        x1 = torch.index_select(
            x, dim=-1,
            index=torch.tensor(
                player_tensor_indices[self.type_feature]
            ).to(x.device)
        ).float()
        x2 = x1.reshape(-1, len(player_tensor_indices[self.type_feature]))

        z_f = self.normalizer(x2)

        return z_f
