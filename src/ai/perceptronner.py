
import torch
import torch.nn as nn


class Perceptronner(nn.Module):
    def __init__(
        self,
        dim_input: int, dim_projection: int, dim_output: int,
        project_init: str, embed_init: str,
        init_val: float=0.0,
        norm: bool=False, dropout: float=0.0,
        biases: tuple[bool]=(False, False),
        pow_iters: int=1
    ) -> None:
        super().__init__()

        self.norm = norm.lower()
        self.dropout = dropout

        if self.norm == "rms":
            self.norm = nn.RMSNorm(dim_input, eps=1e-6)
        else:
            self.norm = nn.LayerNorm(dim_input, eps=1e-6, bias=False)
        self.proj = nn.Linear(dim_input, dim_projection, bias=biases[0])
        self.actv = nn.GELU("tanh")
        if self.dropout > 0.0:  self.drpt = nn.Dropout(dropout)
        self.embd = nn.Linear(dim_projection, dim_output, bias=biases[1])

        with torch.no_grad():
            if project_init == "zeros":  nn.init.zeros_(self.proj.weight)
            if project_init == "ones":  nn.init.ones_(self.proj.weight)
            if project_init == "kaiming":
                nn.init.kaiming_normal_(self.proj.weight)
            if project_init == "xavier":
                nn.init.xavier_uniform_(self.proj.weight)
            if project_init == "orthogonal":
                nn.init.orthogonal_(self.proj.weight)
            if project_init == "constant":
                nn.init.constant_(self.proj.weight, init_val)
            if embed_init == "zeros":  nn.init.zeros_(self.embd.weight)
            if embed_init == "ones":  nn.init.ones_(self.embd.weight)
            if embed_init == "kaiming":
                nn.init.kaiming_normal_(self.embd.weight)
            if embed_init == "xavier":
                nn.init.xavier_uniform_(self.embd.weight)
            if embed_init == "orthogonal":
                nn.init.orthogonal_(self.embd.weight)
            if embed_init == "constant":
                nn.init.constant_(self.embd.weight, init_val)
            if biases[0]:  nn.init.zeros_(self.proj.bias)
            if biases[1]:  nn.init.zeros_(self.embd.bias)

        self.embd = nn.utils.parametrizations.spectral_norm(
            module=self.embd, n_power_iterations=pow_iters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        if self.norm != "":  n = self.norm(x)
        else:  n = x
        p = self.proj(n)
        a = self.actv(p)
        if self.dropout > 0.0:  d = self.drpt(a)
        else:  d = a
        z_f = self.embd(d)

        return z_f
