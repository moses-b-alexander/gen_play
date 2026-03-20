
from math import log2
import torch
import torch.nn as nn

from ai.perceptronner import Perceptronner


class MLPEncoder(nn.Module):
    def __init__(
        self,
        residual: bool, num_mlps: int,
        dim_start: int, dim_end: int,
        norms: list[bool], dropouts: list[float],
        biases: list[tuple[bool]],
        pow_iters: int
    ) -> None:
        super().__init__()

        self.residual = residual

        if num_mlps < 1:  num_mlps = 1

        if dim_start > dim_end:
            dims = [
                dim_start // (2 ** i)
                for i in range(
                    int(log2(dim_start // dim_end)) + 1
                )
            ]
        elif dim_end > dim_start:
            dims = [
                dim_start * (2 ** i)
                for i in range(
                    int(log2(dim_end // dim_start)) + 1
                )
            ]
        else:  dims = [dim_start] * num_mlps

        if len(dims) != num_mlps + 1:
            dims = [dim_start] * num_mlps
        if len(norms) != num_mlps:
            norms = [""] * num_mlps
        if len(dropouts) != num_mlps:
            dropouts = [0.0] * num_mlps
        if len(biases) != num_mlps:
            biases = [(False, False)] * num_mlps

        if residual:
            self.mlps = nn.ModuleList([
                Perceptronner(
                    dim_input=dim_start,
                    dim_projection=dim_start,
                    dim_output=dim_start,
                    norm=norms[i], dropout=dropouts[i],
                    project_init="kaiming", embed_init="kaiming",
                    biases=biases[i],
                    pow_iters=pow_iters
                ) for i in range(num_mlps - 1)
            ])
            self.mlps.append(
                Perceptronner(
                    dim_input=dim_start,
                    dim_projection=dim_end,
                    dim_output=dim_end,
                    norm=norms[-1], dropout=dropouts[-1],
                    project_init="kaiming", embed_init="xavier",
                    biases=biases[-1],
                    pow_iters=pow_iters
                )
            )
        else:
            if num_mlps == 1:
                self.mlps = nn.ModuleList([
                    Perceptronner(
                        dim_input=dim_start,
                        dim_projection=dim_end,
                        dim_output=dim_end,
                        norm=norms[0], dropout=dropouts[0],
                        project_init="kaiming", embed_init="xavier",
                        biases=biases[0],
                        pow_iters=pow_iters
                    )
                ])
            else:
                self.mlps = nn.ModuleList([
                    Perceptronner(
                        dim_input=dims[i],
                        dim_projection=dims[i+1],
                        dim_output=dims[i+1],
                        norm=norms[i], dropout=dropouts[i],
                        project_init="kaiming", embed_init="kaiming",
                        biases=biases[i],
                        pow_iters=pow_iters
                    ) for i in range(num_mlps - 1)
                ])
                self.mlps.append(
                    Perceptronner(
                        dim_input=dims[-2],
                        dim_projection=dims[-1],
                        dim_output=dims[-1],
                        norm=norms[-1], dropout=dropouts[-1],
                        project_init="kaiming", embed_init="xavier",
                        biases=biases[-1],
                        pow_iters=pow_iters
                    )
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        z_i = x
        for mlp in self.mlps[:-1]:
            if self.residual:  z_i += mlp(z_i)
            else:  z_i = mlp(z_i)
        z_f = self.mlps[-1](z_i)

        return z_f
