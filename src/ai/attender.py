
from math import sqrt
import torch
import torch.nn as nn


class Attender(nn.Module):
    def __init__(
        self,
        dim_embedding: int, num_heads: int,
        tau: float, tau_stdev: float,
        dropout: float,
        pow_iters: int
    ) -> None:
        super().__init__()

        self.num_heads = num_heads

        self.dim_embedding = dim_embedding

        self.dim_head = self.dim_embedding // self.num_heads
        self.dim_scale = sqrt(self.dim_head)

        self.tau = tau if tau > 0.0 and tau < 1.0 else 1.0
        if tau_stdev > 1e-6 and tau_stdev < 1.0:
            unif_bnd = tau_stdev * sqrt(3)
        else:  unif_bnd = 0.05

        self.linear_kernel = torch.cos
        # self.linear_kernel = torch.elu

        self.t_q = nn.Parameter(torch.ones(self.num_heads))
        self.t_k = nn.Parameter(torch.ones(self.num_heads))

        self.w_q = nn.Linear(dim_embedding, dim_embedding, bias=False)
        self.w_k = nn.Linear(dim_embedding, dim_embedding, bias=False)
        self.w_v = nn.Linear(dim_embedding, dim_embedding, bias=False)
        self.w_o = nn.Linear(dim_embedding, dim_embedding, bias=False)

        with torch.no_grad():
            Wp_q = self.w_q.weight.view(
                self.num_heads, self.dim_head, self.dim_embedding)
            for h in range(self.num_heads):  nn.init.orthogonal_(Wp_q[h])
            self.w_q.weight.copy_(Wp_q.view_as(self.w_q.weight))
            Wp_k = self.w_k.weight.view(
                self.num_heads, self.dim_head, self.dim_embedding)
            for h in range(self.num_heads):  nn.init.orthogonal_(Wp_k[h])
            self.w_k.weight.copy_(Wp_k.view_as(self.w_k.weight))
            nn.init.xavier_uniform_(self.w_v.weight)
            nn.init.xavier_uniform_(self.w_o.weight)
            self.t_q.data.uniform_(-unif_bnd, +unif_bnd)
            self.t_k.data.uniform_(-unif_bnd, +unif_bnd)

        self.w_q = nn.utils.parametrizations.spectral_norm(
            module=self.w_q, n_power_iterations=pow_iters)
        self.w_k = nn.utils.parametrizations.spectral_norm(
            module=self.w_k, n_power_iterations=pow_iters)
        self.w_o = nn.utils.parametrizations.spectral_norm(
            module=self.w_o, n_power_iterations=pow_iters)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x1: torch.Tensor, x2: torch.Tensor, x3: torch.tensor
    ) -> torch.Tensor:
        x1 = x1
        x2 = x2
        x3 = x3

        x_shape = x1.shape
        N, TB, d = x_shape
        N *= 2

        q_p = self.w_q(x1)
        k_p = self.w_k(x2)
        v_p = self.w_v(x3)

        q_h = q_p.view(N//2, TB, self.num_heads, self.dim_head)
        k_h = k_p.view(N//2, TB, self.num_heads, self.dim_head)
        v_h = v_p.view(N//2, TB, self.num_heads, self.dim_head)

        q_b = q_h.permute(1, 2, 0, 3).contiguous()
        k_b = k_h.permute(1, 2, 0, 3).contiguous()
        v_b = v_h.permute(1, 2, 0, 3).contiguous()

        tau_qk = torch.sigmoid(torch.tensor(self.tau)).item()

        tt_q = (
            tau_qk * torch.tanh(self.t_q).clamp_(min=-1.0, max=+1.0)
        ) + 1.0
        tt_k = (
            tau_qk * torch.tanh(self.t_k).clamp_(min=-1.0, max=+1.0)
        ) + 1.0

        tt_q += 1e-3
        tt_k += 1e-3

        q_f = self.linear_kernel(q_b / tt_q.view(-1, 1, 1).unsqueeze(0)) + 1.0
        k_f = self.linear_kernel(k_b / tt_k.view(-1, 1, 1).unsqueeze(0)) + 1.0

        num = q_f @ k_f.transpose(-2, -1)
        num /= self.dim_scale
        num = self.dropout(num)
        denom = (q_f @ k_f.sum(dim=-2, keepdim=True).transpose(-2, -1)) + 1e-6

        n_qk = num / denom
        n_qkv = n_qk @ v_b

        n_o1 = n_qkv.permute(0, 2, 1, 3).contiguous()
        n_o2 = n_o1.view(N//2, TB, self.dim_embedding)
        n_o3 = n_o2.view((N//2)*TB, -1)
        o = self.w_o(n_o3)

        z_f = o.view(N//2, TB, -1)

        return z_f
