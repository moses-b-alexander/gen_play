
from math import sqrt
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


def seed_worker(worker_id: int) -> None:
    worker_seed = (seed + worker_id) % (2 ** 32)

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

def collect_distributions(
    tups: list[tuple[dtyp, nn.Module]], states: PlayStates
) -> list[torch.distributions.independent.Independent]:
    dists = [
        t[1].to_probability_distribution(states, None, True) for t in tups
    ]

    return dists

def pad_second_dimension(t: torch.Tensor) -> torch.Tensor:
    val = batch_size - t.size(1)
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

def accumulate_xy(
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

    t[..., 0] = (ts[1, :, 21] + tdd_x + ts[1, :, -4])
    t[..., 1] = (ts[1, :, 22] + tdd_y + ts[1, :, -3])

    t[..., 0] *= x_field_max
    t[..., 1] *= np.abs(y_bnd)

    tt = te.clone() if t.size(1) != shape_players else t.clone()

    return (
        (tt[:, :team_players, 0], tt[:, team_players:, 0]),
        (tt[:, :team_players, 1], tt[:, team_players:, 1]),
        (
            torch.round(ts[..., 21] * x_field_max, decimals=0)[1][1].item(),
            torch.round(
                (ts[..., 22] * np.abs(y_bnd)) + y_mid, decimals=3
            )[1][1].item()
        )
    )

def get_trajectory_by_index(
    t: torch.Tensor, batch_index: int, timestep: int
) -> torch.Tensor:

    return t[batch_index, timestep, :, ...]

def load_models(
    uid: str, storage_device: torch.device
) -> list[tuple[dtyp, dict]]:
    l = []
    if uid == "":  return []
    if os.path.exists(os.path.join(output_dir, uid)):
        wd = os.path.join(output_dir, uid)
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
