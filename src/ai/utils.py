
from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import torch
import torch.nn as nn

from ai.constants import action_dim
from common.constants import dtyp, seed, team_players
from common.dirs import output_dir
from data.constants import shape_players, x_field_max, y_mid, y_bnd
from data.play_gfn import PlayStates
from gflownet.estimators import Estimator


def seed_worker(worker_id: int) -> None:
    worker_seed = (seed + worker_id) % ((2 ** 31) - 1)

    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    return None

def inv_sqrt(q: float) -> float:

    return (1.00 / (sqrt(q)))

def sinusoidal_positional_encoding(
    max_len: int, actual_lens: torch.Tensor, dim_out: int, tau: float,
    device: torch.device
) -> torch.Tensor:
    B = actual_lens.size(0)
    true_lens = actual_lens.unsqueeze(0).float()

    p = torch.arange(max_len, device=device).float().unsqueeze(1)
    nt = ((2 * torch.pi) * p) / (true_lens * 1)

    i = torch.arange(dim_out, device=device).float().view(1, 1, -1)
    ei = 2 * (i // 2) / dim_out

    angles = ((nt / tau).unsqueeze(-1)) / (10000 ** ei)

    pe = torch.zeros(max_len, B, dim_out, device=device)
    pe[..., 0::2] = torch.sin(angles[..., 0::2])
    pe[..., 1::2] = torch.cos(angles[..., 1::2])

    return pe

def visualize_latent(
    h: torch.Tensor, label: str="latent", max_cos_rows: int=200,
    save_path: str | None=None,
) -> dict:
    eps = 1e-12
    h = h.detach().reshape(-1, h.shape[-1])

    var = h.var(dim=-1, unbiased=False)
    mean_sq = h.mean(dim=-1) ** 2
    rms = (var + mean_sq).sqrt()
    direction_frac = var / (var + mean_sq + eps)

    n = h.size(0)
    if n > max_cos_rows:
        idx = torch.randperm(n, device=h.device)[:max_cos_rows]
        h_s = h[idx]
    else:
        h_s = h
    h_norm = nn.functional.normalize(h_s, dim=-1, eps=eps)
    cos = h_norm @ h_norm.T
    mask = ~torch.eye(cos.size(0), dtype=torch.bool, device=cos.device)
    off_diag_cos = cos[mask]

    stats = dict(
        rms=rms.cpu().numpy(),
        var=var.cpu().numpy(),
        mean_sq=mean_sq.cpu().numpy(),
        direction_frac=direction_frac.cpu().numpy(),
        cos_sim=off_diag_cos.cpu().numpy(),
    )

    print(
        f"[{label}]  n={n}"
        f"  rms median={np.median(stats['rms']):.4g}"
        f"  direction_frac median={np.median(stats['direction_frac']):.4g}"
        f"  (min={np.min(stats['direction_frac']):.4g})"
        f"  cos_sim mean={np.mean(stats['cos_sim']):.4g}"
        f"  (max={np.max(stats['cos_sim']):.4g})"
    )

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), dpi=150)

    axes[0].hist(stats["rms"], bins=40, color="#2a78d6")
    axes[0].set_title("Per-row RMS")
    axes[0].set_xlabel("sqrt(var + mean^2)")

    axes[1].hist(stats["direction_frac"], bins=40, color="#1baf7a")
    axes[1].set_title("Direction-dominance fraction")
    axes[1].set_xlabel("var / (var + mean^2)")
    axes[1].set_xlim(0, 1)

    axes[2].hist(stats["cos_sim"], bins=40, color="#eb6834")
    axes[2].set_title("Pairwise cosine similarity")
    axes[2].set_xlabel("cosine similarity (subsampled, off-diagonal)")
    axes[2].set_xlim(-1, 1)

    fig.suptitle(label)
    fig.tight_layout()

    if save_path is None:
        save_path = os.path.join(output_dir, f"z_diagnostics_{label}.png")
    fig.savefig(save_path)
    plt.close(fig)
    print(f"saved: {os.path.abspath(save_path)}")

    return stats

def load_models(
    uid: str, storage_device: torch.device
) -> list[tuple[dtyp, dict]]:
    l = []
    if uid == "":  return []
    udir = os.path.join(output_dir, uid)
    if os.path.exists(udir) and os.path.isdir(udir):
        wd = udir
    else:
        return []
    sl = [f for f in os.listdir(wd)]
    for s in sl:
        ss = os.path.join(wd, s)
        t = torch.load(ss, map_location=storage_device, weights_only=False)
        log_z = dtyp(t["log_z"])
        hps = t["hyperparameters"].copy()
        cfg = t["config"].copy()
        p_f = t["pf"].copy()
        l.append((log_z, hps, cfg, p_f))

    return l

def collect_distributions(
    tups: list[tuple[dtyp, Estimator]], states: PlayStates
) -> list[torch.distributions.independent.Independent]:
    dists = [
        t[1].to_probability_distribution(states, None, True) for t in tups
    ]

    return dists

def pad_second_dimension(t: torch.Tensor, n: int) -> torch.Tensor:
    val = n - t.size(1)
    if val > 0:
        if len(t.shape) == 4:  tup = (0, 0, 0, 0, 0, val, 0, 0)
        elif len(t.shape) == 3:  tup = (0, 0, 0, val, 0, 0)
        elif len(t.shape) == 2:  tup = (0, val, 0, 0)
        elif len(t.shape) == 1:  tup = (0, val)

    if val > 0:  return nn.functional.pad(t, tup, "constant", dtyp(0.0))
    else:  return t

def permute_batch_first(t: torch.Tensor) -> torch.Tensor:
    if len(t.shape) == 2:  t1 = t.permute(1, 0)
    elif len(t.shape) == 3:  t1 = t.permute(1, 0, 2)
    elif len(t.shape) == 4:  t1 = t.permute(1, 0, 2, 3)
    else:  t1 = t

    return t1.contiguous()

def accumulate_xy( # TODO null padded T
    td: torch.Tensor, ts: torch.Tensor
) -> tuple[tuple]:

    t = torch.zeros((ts.shape[0], shape_players, action_dim))
    te = torch.zeros((ts.shape[0], shape_players, action_dim))

    tdd_x = torch.cat(
        [torch.zeros_like(td[:1, ..., 0]), td[..., 0].cumsum(dim=0)[:-1]],
        dim=0
    )
    tdd_y = torch.cat(
        [torch.zeros_like(td[:1, ..., 1]), td[..., 1].cumsum(dim=0)[:-1]],
        dim=0
    )

    t[..., 0] = (ts[1, :, 23] + tdd_x + ts[1, :, -5])
    t[..., 1] = (ts[1, :, 24] + tdd_y + ts[1, :, -4])

    t[..., 0] *= x_field_max
    t[..., 1] *= np.abs(y_bnd)

    tt = te.clone() if t.size(1) != shape_players else t.clone()

    return (
        (tt[:, :team_players, 0], tt[:, team_players:, 0]),
        (tt[:, :team_players, 1], tt[:, team_players:, 1]),
        (
            torch.round(ts[..., 23] * x_field_max, decimals=0)[1][1].item(),
            torch.round(
                (ts[..., 24] * np.abs(y_bnd)) + y_mid, decimals=3
            )[1][1].item()
        )
    )
