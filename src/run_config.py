
from __future__ import annotations

import json
from pathlib import Path

from common.constants import fps


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_CONFIG_PATH = PROJECT_ROOT / "dashboard_configs" / "active.json"

DEFAULTS: dict[str, str] = {
    "dim_play": "128",
    "dim_player": "128",
    "num_heads": "4",
    "dropout": "0.10",
    "expansion": "2",
    "lags_def_tm": "0,15",
    "lags_off_tm": "0,15",
    "lags_def_op": "15,30",
    "lags_off_op": "15,30",
    "drift_size": "2",
    "diffusion_size": "2",
    "diff_eq_dim_middle": "32",
    "dt": "1e-2",
    "final_dim": "64",
    "pow_iters": "1",
    "learning_rate": "1e-4",
    "weight_decay_rate": "1e-5",
    "batch_size": "8",
    "num_epochs": "1",
    "reward_scale": "1.0",
    "reward_beta": "10.0",
    "reward_sign": "False",
    "max_score_diff": "0.21",
    "snap_x_low": "0.20",
    "snap_x_high": "0.80",
    "seasons": "TB:2022",
    "match_count": "15",
    "train_ratio": "0.9",
    "max_window": str(fps * 30),
    "cut_two_min": "True",
    "catg_idxs": "0",
    "km_alpha_decay": "0.15",
    "km_num_clusters": "16",
    "decoder_min_stdv": "1e-5",
    "decoder_max_stdv": "1e-2",
    "noise_floor": "1e-6",
    "noise_ceiling": "1e-1",
    "noise_decay": "0.2",
    "noise_exp": "1.0",
    "n_trials": "12",
    "n_parallel": "1",
    "claude_model": "claude-opus-4-8",
}

FIXED_PRIOR_MEANS = ((0.0, 0.0), (0.0, 0.0))
FIXED_PRIOR_STDVS = ((0.0010, 0.0010), (0.0010, 0.0010))

def load_active_config() -> dict[str, str]:
    cfg = dict(DEFAULTS)
    if ACTIVE_CONFIG_PATH.exists():
        try:
            payload =\
                json.loads(ACTIVE_CONFIG_PATH.read_text(encoding="utf-8"))
            cfg.update(payload.get("values", {}))
        except (json.JSONDecodeError, OSError):
            pass
    return cfg

def _b(s: str) -> bool:
    return str(s).strip().lower() in ("true", "1", "yes", "y", "on")

def parse_seasons(raw: str) -> list[tuple[str, int]]:
    out = []
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        team, year = part.split(":")
        out.append((team.strip().upper(), int(year.strip())))
    return out

def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]

def parse_catg_idxs(raw: str) -> list[int]:
    return parse_int_list(raw)

def build_run_config(cfg: dict[str, str]) -> dict:
    postprocess_kwargs = dict(
        ci=parse_catg_idxs(cfg["catg_idxs"]),
        mw=int(cfg["max_window"]),
        ad=float(cfg["km_alpha_decay"]),
        nc=int(cfg["km_num_clusters"]),
        sn=_b(cfg["reward_sign"]),
        rs=float(cfg["reward_scale"]),
        rb=float(cfg["reward_beta"]),
        msd=float(cfg["max_score_diff"]),
        sxr=(float(cfg["snap_x_low"]), float(cfg["snap_x_high"])),
        ctm=_b(cfg["cut_two_min"]),
    )
    encoder_kwargs = dict(
        lags=(
            parse_int_list(cfg["lags_def_tm"]),
            parse_int_list(cfg["lags_off_tm"]),
            parse_int_list(cfg["lags_def_op"]),
            parse_int_list(cfg["lags_off_op"]),
        ),
        dim_play=int(cfg["dim_play"]), dim_player=int(cfg["dim_player"]),
        num_heads=int(cfg["num_heads"]), dropout=float(cfg["dropout"]),
        expansion=int(cfg["expansion"]),
    )
    diff_eq_kwargs = dict(
        drift_size=int(cfg["drift_size"]),
        diffusion_size=int(cfg["diffusion_size"]),
        dim_middle=int(cfg["diff_eq_dim_middle"]),
        steps=2,
        rtol=1e-5, atol=1e-5,
        dt=float(cfg["dt"]),
    )
    decoder_kwargs = dict(
        min_s=float(cfg["decoder_min_stdv"]),
        max_s=float(cfg["decoder_max_stdv"]),
    )
    output_kwargs = dict(
        noise_floor=float(cfg["noise_floor"]),
        noise_ceiling=float(cfg["noise_ceiling"]),
        noise_decay=float(cfg["noise_decay"]),
        noise_exp=float(cfg["noise_exp"]),
        prior_means=FIXED_PRIOR_MEANS,
        prior_stdvs=FIXED_PRIOR_STDVS,
    )
    shared_kwargs = dict(
        dim_h=int(cfg["final_dim"]),
        pow_iters=int(cfg["pow_iters"])
    )
    opt_args = dict(
        bs=int(cfg["batch_size"]),
        lr=float(cfg["learning_rate"]),
        wdr=float(cfg["weight_decay_rate"]),
        ne=int(cfg["num_epochs"]),
    )

    return dict(
        seasons=parse_seasons(cfg["seasons"]),
        match_count=int(cfg["match_count"]),
        train_ratio=float(cfg["train_ratio"]),
        postprocess_kwargs=postprocess_kwargs,
        encoder_kwargs=encoder_kwargs,
        diff_eq_kwargs=diff_eq_kwargs,
        decoder_kwargs=decoder_kwargs,
        output_kwargs=output_kwargs,
        shared_kwargs=shared_kwargs,
        opt_args=opt_args,
    )
