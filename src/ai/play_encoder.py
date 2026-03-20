
import torch
import torch.nn as nn

from ai.constants import global_dim, play_tensor_indices
from ai.mlp_encoder import MLPEncoder
from ai.normalizer import Normalizer


class PlayEncoder(nn.Module):
    def __init__(
        self,
        num_mlps: int,
        dim_projection: int,
        dim_embedding_start: int, dim_embedding_end: int,
        dropout: float,
        pow_iters: int
    ) -> None:
        super().__init__()

        self.exclude_indices = \
            (play_tensor_indices["snap_y"] + play_tensor_indices["length"])

        self.normalizer_play = Normalizer(
            type_norm="layer",
            dim_start=global_dim-len(self.exclude_indices),
            dim_middle=dim_projection,
            dim_end=dim_embedding_start,
            biases=(True, False), norm_bias=False,
            pow_iters=pow_iters
        )
        self.encoder = MLPEncoder(
            residual=False, num_mlps=num_mlps,
            dim_start=dim_embedding_start, dim_end=dim_embedding_end,
            norms=[""]*num_mlps, dropouts=[dropout]*num_mlps,
            biases=[(False, False)]*num_mlps,
            pow_iters=pow_iters
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        x1 = torch.index_select(
            x, dim=-1,
            index=torch.tensor([
                i for i in range(global_dim) if i not in self.exclude_indices
            ]).to(x.device)
        ).float()
        x2 = x1.reshape(-1, global_dim-len(self.exclude_indices))

        n = self.normalizer_play(x2)
        z_f = self.encoder(n)

        return z_f
