
"""
Agentic hyperparameter search using Claude.

SEASON RECOMMENDATION FOR HP SEARCH
-------------------------------------
Default is TB 2022 — matches your existing baseline runs
(bag_ct=1, ratio=0.999) and (bag_ct=8, ratio=0.700).
Use TB 2021 for the cleanest search signal (see ranking);
swap with --seasons TB:2021.

Season quality ranking for HP search:
  1. TB 2021  — best: ~95% Super Bowl LV roster retained,
               highest 11-personnel rate (~60-65% snaps),
               13-4 record keeps game scripts clean.
  2. NE 2018  — runner-up: McDaniels final NE year, but
               Josh Gordon mid-season adds formation noise.
  3. NE 2017  — solid scheme consistency, Gronk present.
  4. TB 2022  — default here. Brady retirement drama,
               Gronk retired, Godwin recovering; more
               roster noise than 2021.
  5. TB 2020  — first Brady TB year, new system.
  6. NE 2019  — worst: no proven WR corps all season.

  Estimated effect on trial time (RTX 3060):
    - All 6 seasons  -> ~2 hr/trial
    - TB 2022 only   -> ~20-30 min/trial
    - 12 trials feasible locally (~5 hrs) or cheap
      on cloud (~$1-2 on g5.xlarge spot).

PARALLELISM MODES
-----------------
  n_parallel=1 (default)
    Sequential loop. Claude proposes one config per
    round via `propose_hyperparameters`.

  n_parallel=k  (k >= 2)
    Batched parallel. Claude proposes k configs in one
    call. Trials run concurrently, one subprocess per
    GPU via CUDA_VISIBLE_DEVICES=0..k-1.
    12 trials, k=4 -> 3 rounds x trial-time.
    Ideal for g5.12xlarge (~$1.50/hr spot) or
    g2-standard-48 (~$1.12/hr preemptible).
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from time import perf_counter

import anthropic
import numpy as np
import pandas as pd
import torch

from ai.constants import (
    action_dim,
    downsampling,
    shape_players,
    state_dim,
)
from ai.hyperparameters import (
    get_decoder_hyperparameters,
    get_encoder_hyperparameters,
)
from ai.pf import PF
from ai.train import train_bagged_model
from common.constants import dtyp
from common.devices import learning_device
from common.dirs import processed_dir
from data.constants import max_window, reward_sign, x_field_max
from data.processing import postprocess_df, split_df


# ---------------------------------------------------------------------------
# Recommended season (matches existing baseline runs)
# ---------------------------------------------------------------------------

RECOMMENDED_SEARCH_SEASONS: list[tuple[str, int]] = [("TB", 2022)]


# ---------------------------------------------------------------------------
# Hyperparameter search space
# ---------------------------------------------------------------------------

_HP_PROPS: dict = {
    "dim_play": {
        "type": "integer",
        "description": "Play-level embedding dimension.",
        "enum": [64, 128, 256],
    },
    "dim_player": {
        "type": "integer",
        "description": (
            "Per-player embedding dimension."
            " num_heads must evenly divide this."
        ),
        "enum": [64, 128, 256],
    },
    "num_heads": {
        "type": "integer",
        "description": (
            "Attention heads. Must divide"
            " dim_player evenly."
        ),
        "enum": [2, 4, 8],
    },
    "dropout": {
        "type": "number",
        "description": (
            "Dropout probability across all"
            " MLP and attention blocks."
        ),
        "enum": [0.0, 0.05, 0.10, 0.20],
    },
    "expansion": {
        "type": "integer",
        "description": (
            "MLP width expansion factor"
            " (output = expansion^depth * dim)."
        ),
        "enum": [1, 2, 3],
    },
    "final_dim": {
        "type": "integer",
        "description": (
            "Shared hidden dim (dim_h)"
            " feeding into the SDE and decoder."
        ),
        "enum": [32, 64, 128],
    },
    "drift_size": {
        "type": "integer",
        "description": "Depth of the SDE drift network.",
        "enum": [1, 2, 3, 4],
    },
    "diffusion_size": {
        "type": "integer",
        "description": "Depth of the SDE diffusion network.",
        "enum": [1, 2, 3, 4],
    },
    "diff_eq_dim_middle": {
        "type": "integer",
        "description": (
            "Hidden width of SDE networks."
            " Should be <= final_dim."
        ),
        "enum": [16, 32, 64],
    },
    "sde_steps": {
        "type": "integer",
        "description": "Euler integration steps for the SDE.",
        "enum": [2, 4, 6, 8],
    },
    "sde_scale": {
        "type": "number",
        "description": "Initial noise scale for the SDE.",
        "enum": [0.01, 0.1, 1.0],
    },
    "learning_rate": {
        "type": "number",
        "description": "AdamW learning rate.",
        "enum": [3e-5, 1e-4, 3e-4, 1e-3],
    },
    "batch_size": {
        "type": "integer",
        "description": (
            "Training batch size"
            " (trajectories per step)."
        ),
        "enum": [4, 8, 16],
    },
    "rationale": {
        "type": "string",
        "description": (
            "1-2 sentence explanation of why you"
            " chose this config given results so far."
        ),
    },
}

_HP_REQUIRED = list(_HP_PROPS.keys())

_SINGLE_TOOL = {
    "name": "propose_hyperparameters",
    "description": (
        "Propose the next hyperparameter configuration."
        " Training returns log_Z (log partition function)."
        " HIGHER is better."
    ),
    "input_schema": {
        "type": "object",
        "properties": _HP_PROPS,
        "required": _HP_REQUIRED,
        "additionalProperties": False,
    },
}


def _make_batch_tool(k: int) -> dict:
    """Return a tool requesting exactly k configs in one call."""
    per_cfg = {
        key: val for key, val in _HP_PROPS.items()
        if key != "rationale"
    }
    per_cfg["rationale"] = {
        "type": "string",
        "description": (
            "Why this config, and how it differs"
            " from the others in this batch."
        ),
    }
    return {
        "name": "propose_hyperparameter_batch",
        "description": (
            f"Propose exactly {k} distinct"
            " hyperparameter configurations to"
            f" evaluate in parallel. Return all {k}"
            " in one call. HIGHER log_Z is better."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "configurations": {
                    "type": "array",
                    "description": (
                        f"Exactly {k} HP configurations"
                        " for this parallel round."
                    ),
                    "minItems": k,
                    "maxItems": k,
                    "items": {
                        "type": "object",
                        "properties": per_cfg,
                        "required": list(per_cfg.keys()),
                        "additionalProperties": False,
                    },
                },
                "batch_rationale": {
                    "type": "string",
                    "description": (
                        "Overall exploration strategy"
                        " for this batch."
                    ),
                },
            },
            "required": [
                "configurations",
                "batch_rationale",
            ],
            "additionalProperties": False,
        },
    }


# ---------------------------------------------------------------------------
# Build model kwargs from a flat HP dict
# ---------------------------------------------------------------------------

def _build_pf_hps(
    hps: dict,
    num_timesteps: int,
) -> tuple[dict, dict]:
    input_hps = dict(
        player_count=(shape_players // 2),
        trajectory_length=num_timesteps,
    )
    encoder_hps = get_encoder_hyperparameters(
        lags=([0, 15], [15, 30]),
        dim_play=hps["dim_play"],
        dim_player=hps["dim_player"],
        num_heads=hps["num_heads"],
        dropout=hps["dropout"],
        expansion=hps["expansion"],
    ) | input_hps
    diff_eq_hps = dict(
        drift_size=hps["drift_size"],
        diffusion_size=hps["diffusion_size"],
        dim_middle=hps["diff_eq_dim_middle"],
        steps=hps["sde_steps"],
        rtol=1e-5, atol=1e-5,
        scale=hps["sde_scale"],
        dt=1e-2,
    )
    decoder_hps = (
        get_decoder_hyperparameters(min_s=1e-5, max_s=1e-2)
        | input_hps
    )
    output_hps = dict(
        noise_floor=1e-6,
        noise_ceiling=1e-1,
        noise_gap=10,
        noise_exp=1.0,
        prior_means=((0.0, 0.0), (0.0, 0.0)),
        prior_stdvs=((1e-3, 1e-3), (1e-3, 1e-3)),
        device=learning_device,
    )
    shared = dict(dim_h=hps["final_dim"], pow_iters=1)
    base = dict(
        dim_start=state_dim,
        dim_hidden=shared["dim_h"],
        dim_end=action_dim,
        pow_iters=shared["pow_iters"],
        encoder_hps=encoder_hps,
        diff_eq_hps=diff_eq_hps,
        decoder_hps=decoder_hps,
        **output_hps,
        **input_hps,
    )
    return (
        dict(backwards=False, **base),
        dict(backwards=True, **base),
    )


def _validate_hps(hps: dict) -> dict:
    hps = deepcopy(hps)
    if hps["dim_player"] % hps["num_heads"] != 0:
        for h in [8, 4, 2]:
            if hps["dim_player"] % h == 0:
                hps["num_heads"] = h
                break
    if hps["diff_eq_dim_middle"] > hps["final_dim"]:
        hps["diff_eq_dim_middle"] = hps["final_dim"]
    return hps


# ---------------------------------------------------------------------------
# Sequential trial runner (main process)
# ---------------------------------------------------------------------------

def run_trial(
    hps: dict,
    df_train: pd.DataFrame,
    num_timesteps: int,
) -> dict:
    """Train one bag and return metrics."""
    hps = _validate_hps(hps)
    pf_hps, pb_hps = _build_pf_hps(hps, num_timesteps)

    import ai.constants as _ac
    _orig_bs = _ac.batch_size
    _ac.batch_size = hps["batch_size"]

    t0 = perf_counter()
    rets, _ = train_bagged_model(
        bag_ct=1, ratio=0.9,
        df_m=df_train,
        pf_cls=PF, pf_args=pf_hps,
        pb_cls=PF, pb_args=pb_hps,
        random=True,
        write_model=False,
        runner_device=learning_device,
    )
    _ac.batch_size = _orig_bs

    log_z = float(rets[0][0])
    return {
        "log_z": log_z,
        "yards_approx": float(np.exp(log_z) * x_field_max),
        "seconds": round(perf_counter() - t0, 1),
    }


# ---------------------------------------------------------------------------
# Parallel trial worker — top-level for pickle compatibility
# ---------------------------------------------------------------------------

def _trial_worker(args: tuple) -> dict:
    """
    Subprocess entry point. Sets CUDA_VISIBLE_DEVICES before
    any CUDA import so torch sees the correct physical GPU.

    args = (
        gpu_id, hps, parquet_paths,
        proc_params, num_timesteps, syspath,
    )
    proc_params = (reward_sign, ci, max_window, downsampling)
    """
    (
        gpu_id, hps, parquet_paths,
        proc_params, num_timesteps, syspath,
    ) = args

    for p in reversed(syspath):
        if p not in sys.path:
            sys.path.insert(0, p)

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    import pandas as pd
    import numpy as np
    import torch
    from copy import deepcopy
    from time import perf_counter

    import ai.constants as _ac
    from ai.constants import (
        action_dim,
        shape_players,
        state_dim,
    )
    from ai.hyperparameters import (
        get_decoder_hyperparameters,
        get_encoder_hyperparameters,
    )
    from ai.pf import PF
    from ai.train import train_bagged_model
    from common.constants import dtyp
    from common.devices import learning_device
    from data.constants import x_field_max
    from data.processing import postprocess_df, split_df

    torch.serialization.add_safe_globals([dtyp])

    sn, ci, mw, dr = proc_params
    dfs = [pd.read_parquet(p).copy() for p in parquet_paths]
    df = pd.concat(dfs, axis=0, ignore_index=False).copy()
    df = postprocess_df(df, sn=sn, ci=ci, mw=mw, dr=dr).copy()

    num_ts = (
        num_timesteps
        or max(df.index.get_level_values(1).tolist())
    )
    train_ids, _ = split_df(df_w=df, ratio=0.9, random=False)
    mask = df.index.get_level_values(0).isin(train_ids)
    df_train = df.loc[mask].copy()

    hps = deepcopy(hps)
    if hps["dim_player"] % hps["num_heads"] != 0:
        for h in [8, 4, 2]:
            if hps["dim_player"] % h == 0:
                hps["num_heads"] = h
                break
    if hps["diff_eq_dim_middle"] > hps["final_dim"]:
        hps["diff_eq_dim_middle"] = hps["final_dim"]

    input_hps = dict(
        player_count=(shape_players // 2),
        trajectory_length=num_ts,
    )
    encoder_hps = get_encoder_hyperparameters(
        lags=([0, 15], [15, 30]),
        dim_play=hps["dim_play"],
        dim_player=hps["dim_player"],
        num_heads=hps["num_heads"],
        dropout=hps["dropout"],
        expansion=hps["expansion"],
    ) | input_hps
    diff_eq_hps = dict(
        drift_size=hps["drift_size"],
        diffusion_size=hps["diffusion_size"],
        dim_middle=hps["diff_eq_dim_middle"],
        steps=hps["sde_steps"],
        rtol=1e-5, atol=1e-5,
        scale=hps["sde_scale"],
        dt=1e-2,
    )
    decoder_hps = (
        get_decoder_hyperparameters(min_s=1e-5, max_s=1e-2)
        | input_hps
    )
    output_hps = dict(
        noise_floor=1e-6,
        noise_ceiling=1e-1,
        noise_gap=10,
        noise_exp=1.0,
        prior_means=((0.0, 0.0), (0.0, 0.0)),
        prior_stdvs=((1e-3, 1e-3), (1e-3, 1e-3)),
        device=learning_device,
    )
    shared = dict(dim_h=hps["final_dim"], pow_iters=1)
    base = dict(
        dim_start=state_dim,
        dim_hidden=shared["dim_h"],
        dim_end=action_dim,
        pow_iters=shared["pow_iters"],
        encoder_hps=encoder_hps,
        diff_eq_hps=diff_eq_hps,
        decoder_hps=decoder_hps,
        **output_hps,
        **input_hps,
    )
    pf_hps = dict(backwards=False, **base)
    pb_hps = dict(backwards=True, **base)

    _orig_bs = _ac.batch_size
    _ac.batch_size = hps["batch_size"]
    t0 = perf_counter()
    rets, _ = train_bagged_model(
        bag_ct=1, ratio=0.9,
        df_m=df_train,
        pf_cls=PF, pf_args=pf_hps,
        pb_cls=PF, pb_args=pb_hps,
        random=True,
        write_model=False,
        runner_device=learning_device,
    )
    _ac.batch_size = _orig_bs

    log_z = float(rets[0][0])
    return {
        "log_z": log_z,
        "yards_approx": float(np.exp(log_z) * x_field_max),
        "seconds": round(perf_counter() - t0, 1),
        "gpu_id": gpu_id,
    }


def _run_parallel_batch(
    proposals: list[dict],
    parquet_paths: list[str],
    proc_params: tuple,
    num_timesteps: int,
) -> list[dict]:
    """Dispatch trials across GPUs 0..len(proposals)-1."""
    k = len(proposals)
    ctx = mp.get_context("spawn")
    args_list = [
        (
            i,
            proposals[i],
            parquet_paths,
            proc_params,
            num_timesteps,
            sys.path[:],
        )
        for i in range(k)
    ]
    with ProcessPoolExecutor(
        max_workers=k, mp_context=ctx
    ) as pool:
        futures = {
            pool.submit(_trial_worker, a): i
            for i, a in enumerate(args_list)
        }
        results: list[dict | None] = [None] * k
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                results[idx] = fut.result()
            except Exception as exc:
                results[idx] = {
                    "log_z": float("-inf"),
                    "yards_approx": 0.0,
                    "seconds": 0.0,
                    "error": str(exc),
                }
    return results


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in deep learning hyperparameter"
    " optimisation for generative models of American"
    " football plays.\n\n"
    "TASK\n"
    "Search for the best hyperparameter configuration"
    " for a Trajectory-Balance GFlowNet that models"
    " NFL play trajectories (player positions at 30 Hz)."
    " Each trial trains one model bag and reports log_Z"
    " (the log partition function). HIGHER log_Z is"
    " better: the model assigns more probability mass"
    " to realistic play patterns, corresponding to"
    " higher expected yards.\n\n"
    "ARCHITECTURE SUMMARY\n"
    "- Encoder: GRU + multi-head attention over player"
    " sequences.\n"
    "  dim_play / dim_player: embedding sizes.\n"
    "  num_heads: must divide dim_player evenly\n"
    "  (e.g. 4 on 128 = OK; 8 on 64 = OK).\n"
    "  dropout: regularisation (~1800 plays;"
    " 0.10 is a safe prior).\n"
    "  expansion: MLP width multiplier"
    " (1=linear, 2=standard FFN).\n"
    "- SDE (diff-eq): continuous-time dynamics.\n"
    "  drift_size / diffusion_size: MLP depth.\n"
    "  diff_eq_dim_middle: hidden width"
    " (keep <= final_dim).\n"
    "  sde_steps: Euler steps (4 is usually fine).\n"
    "  sde_scale: noise magnitude (0.1 is robust).\n"
    "- Shared: final_dim is the bottleneck"
    " for SDE and decoder.\n"
    "- Optimiser: AdamW;"
    " lr and batch_size strongly affect convergence.\n\n"
    "STRATEGY\n"
    "1. Baseline: dim_play=128, dim_player=128,"
    " num_heads=4, dropout=0.10, expansion=2,\n"
    "   final_dim=64, drift=2, diffusion=2,"
    " dim_middle=32, sde_steps=4, sde_scale=0.1,\n"
    "   lr=1e-4, batch=8.\n"
    "2. Vary 1-2 axes per trial to isolate effects.\n"
    "3. Focus on dimensions that matter most.\n"
    "4. Constraints (silently fixed):\n"
    "   - num_heads not dividing dim_player"
    " -> snapped to nearest valid value.\n"
    "   - diff_eq_dim_middle > final_dim"
    " -> clamped.\n"
    "5. Do not repeat configurations in the history."
)


# ---------------------------------------------------------------------------
# Main agentic search loop
# ---------------------------------------------------------------------------

def agentic_hp_search(
    df_train: pd.DataFrame,
    num_timesteps: int,
    parquet_paths: list[str] | None = None,
    proc_params: tuple | None = None,
    n_trials: int = 12,
    n_parallel: int = 1,
    claude_model: str = "claude-opus-4-8",
) -> tuple[dict, list[dict]]:
    """
    Run the agentic HP search.

    Parameters
    ----------
    df_train      : pre-processed training DataFrame
    num_timesteps : max timestep index from the dataset
    parquet_paths : parquet paths for parallel workers
                    (required when n_parallel > 1)
    proc_params   : (reward_sign, ci, max_window,
                    downsampling) for workers
    n_trials      : total number of evaluations
    n_parallel    : 1=sequential; k>1=k-GPU batched
    claude_model  : Anthropic model ID

    Returns
    -------
    best    : {"hps": {...}, "log_z": float, ...}
    history : list of per-trial dicts
    """
    if n_parallel > 1 and parquet_paths is None:
        raise ValueError(
            "parquet_paths required for n_parallel > 1"
        )

    client = anthropic.Anthropic()
    is_parallel = n_parallel > 1
    n_rounds = (n_trials + n_parallel - 1) // n_parallel

    history: list[dict] = []
    best: dict | None = None

    mode_str = (
        "sequentially"
        if not is_parallel
        else f"{n_parallel} at a time in parallel"
    )
    prop_str = (
        f"Propose your first {n_parallel} configurations."
        if is_parallel
        else "Propose your first configuration."
    )
    messages = [
        {
            "role": "user",
            "content": (
                f"Begin the search. Budget: {n_trials}"
                f" trials, {mode_str}. {prop_str}"
            ),
        }
    ]

    trial_counter = 0

    for round_idx in range(n_rounds):
        configs_this_round = min(
            n_parallel, n_trials - trial_counter
        )
        active_tool = (
            _make_batch_tool(configs_this_round)
            if (is_parallel and configs_this_round > 1)
            else _SINGLE_TOOL
        )
        active_name = active_tool["name"]

        print(f"\n{'='*64}")
        print(
            f"Round {round_idx+1}/{n_rounds}  "
            f"(trials {trial_counter+1}-"
            f"{trial_counter+configs_this_round}"
            f"/{n_trials})"
        )
        print(f"{'='*64}")

        response = client.messages.create(
            model=claude_model,
            max_tokens=4096,
            thinking={"type": "adaptive"},
            system=SYSTEM_PROMPT,
            tools=[active_tool],
            tool_choice={
                "type": "tool",
                "name": active_name,
            },
            messages=messages,
        )

        tool_use_block = next(
            b for b in response.content
            if b.type == "tool_use"
        )
        raw_input = tool_use_block.input

        if active_name == "propose_hyperparameter_batch":
            proposals = raw_input["configurations"]
            batch_rat = raw_input.get(
                "batch_rationale", ""
            )
            print(f"Batch rationale: {batch_rat}")
        else:
            proposal = deepcopy(raw_input)
            print(f"Rationale: {proposal.pop('rationale', '')}")
            proposals = [proposal]

        if is_parallel and len(proposals) > 1:
            n_gpu = len(proposals)
            print(
                f"Dispatching {n_gpu} trials"
                f" across GPUs 0-{n_gpu - 1}..."
            )
            metrics_list = _run_parallel_batch(
                proposals,
                parquet_paths,
                proc_params,
                num_timesteps,
            )
        else:
            metrics_list = []
            for hp in proposals:
                hp = {
                    k: v for k, v in hp.items()
                    if k != "rationale"
                }
                print(f"  HPs: {json.dumps(hp)}")
                print(
                    "  Training...",
                    end="",
                    flush=True,
                )
                m = run_trial(hp, df_train, num_timesteps)
                metrics_list.append(m)
                lz = m["log_z"]
                ya = m["yards_approx"]
                print(
                    f" log_Z={lz:.4f}"
                    f"  yards~{ya:.1f}"
                    f"  ({m['seconds']}s)"
                )

        round_results = []
        for hp, m in zip(proposals, metrics_list):
            hp_clean = {
                k: v for k, v in hp.items()
                if k != "rationale"
            }
            is_best = (
                best is None
                or m["log_z"] > best["log_z"]
            )
            if is_best:
                best = {"hps": hp_clean, **m}
            entry = {
                "trial": trial_counter + 1,
                "hps": hp_clean,
                "rationale": hp.get("rationale", ""),
                **m,
                "is_best": is_best,
            }
            history.append(entry)
            round_results.append(entry)
            trial_counter += 1
            if is_best:
                lz = m["log_z"]
                print(f"  *** NEW BEST: log_Z={lz:.4f} ***")

        result_summary = "\n".join(
            f"  Config {i+1}:"
            f" log_Z={r['log_z']:.4f}"
            f"  yards~{r['yards_approx']:.1f}"
            f"  {r['seconds']}s"
            + ("  <- NEW BEST" if r["is_best"] else "")
            for i, r in enumerate(round_results)
        )

        messages.append({
            "role": "assistant",
            "content": response.content,
        })

        if trial_counter < n_trials:
            remaining = n_trials - trial_counter
            next_k = min(n_parallel, remaining)
            next_prompt = (
                f"Propose the next {next_k}"
                " configurations."
                if next_k > 1
                else "Propose the next configuration."
            )
            feedback = (
                f"Round {round_idx+1} complete.\n\n"
                "Full history:\n"
                f"{json.dumps(history, indent=2)}\n\n"
                f"Best so far: log_Z={best['log_z']:.4f}"
                f" (yards~{best['yards_approx']:.1f})\n\n"
                f"{next_prompt}"
            )
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": result_summary,
                    },
                    {
                        "type": "text",
                        "text": feedback,
                    },
                ],
            })
        else:
            messages.append({
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block.id,
                        "content": result_summary,
                    }
                ],
            })

    return best, history


# ---------------------------------------------------------------------------
# Optional trajectory-level filters
# ---------------------------------------------------------------------------

def slice_df(
    df: pd.DataFrame,
    max_score_diff: float | None = 0.14,
    max_down: float | None = 0.50,
    snap_x_range: tuple[float, float] | None = (0.20, 0.80),
    max_quarter: float | None = None,
) -> pd.DataFrame:
    """
    Filter trajectories by play-level conditions. Operates
    at trajectory granularity — either the whole play is
    kept or dropped.

    Parameters are in the same normalised units as the df:
      max_score_diff : off_score - def_score / 100
                       (0.14 = within 14 points)
      max_down       : play_down / 4
                       (0.50 = 1st or 2nd down only)
      snap_x_range   : (lo, hi) for play_snap_x in [0,1]
                       (0.20-0.80 = non-red-zone,
                        non-own-endzone)
      max_quarter    : play_quarter / 4
                       (0.75 = Q1-Q3 only)

    Reads play context from frame index 2 (first real
    frame after the start sentinel), where play-level
    features hold their true values.
    """
    frame2 = df.loc[df.index.get_level_values(1) == 2]

    mask = pd.Series(True, index=frame2.index)

    if max_score_diff is not None:
        mask &= (
            frame2["play_score_difference"].abs()
            <= max_score_diff
        )
    if max_down is not None:
        mask &= frame2["play_down"] <= max_down
    if snap_x_range is not None:
        lo, hi = snap_x_range
        mask &= (
            (frame2["play_snap_x"] >= lo) &
            (frame2["play_snap_x"] <= hi)
        )
    if max_quarter is not None:
        mask &= frame2["play_quarter"] <= max_quarter

    valid_ids = frame2.loc[mask].index.get_level_values(0)
    return df.loc[
        df.index.get_level_values(0).isin(valid_ids)
    ].copy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def _load_seasons(
    search_seasons: list[tuple[str, int]],
) -> tuple[pd.DataFrame, int, list[str]]:
    """Load pre-processed parquets for given seasons."""
    dfs, paths = [], []
    for team, year in search_seasons:
        path = os.path.join(
            processed_dir,
            f"u_df_{team}_{year}.parquet",
        )
        dfs.append(pd.read_parquet(path, engine="pyarrow").copy())
        paths.append(path)

    df_all = pd.concat(
        dfs, axis=0, ignore_index=False
    ).copy()
    df_all = postprocess_df(
        df_all,
        sn=reward_sign,
        ci=[0],
        mw=max_window,
        dr=downsampling,
    ).copy()

    num_timesteps = max(
        df_all.index.get_level_values(1).tolist()
    )
    train_ids, _ = split_df(
        df_w=df_all, ratio=0.9, random=False
    )
    mask = df_all.index.get_level_values(0).isin(train_ids)
    df_train = df_all.loc[mask].copy()
    return df_train, num_timesteps, paths


if __name__ == "__main__":
    import argparse

    _szn = RECOMMENDED_SEARCH_SEASONS[0]
    _szn_default = f"{_szn[0]}:{_szn[1]}"

    parser = argparse.ArgumentParser(
        description="Agentic HP search"
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=12,
        help="Total number of trial evaluations",
    )
    parser.add_argument(
        "--n-parallel",
        type=int,
        default=1,
        help=(
            "GPUs to use in parallel"
            " (1=sequential, k=k-GPU batched)"
        ),
    )
    parser.add_argument(
        "--seasons",
        type=str,
        default=_szn_default,
        help=(
            "Comma-separated TEAM:YEAR pairs,"
            " e.g. 'TB:2022' or 'NE:2018,TB:2021'"
        ),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="claude-opus-4-8",
    )
    args = parser.parse_args()

    search_seasons = []
    for s in args.seasons.split(","):
        team, year = s.strip().split(":")
        search_seasons.append((team.upper(), int(year)))

    print(f"\nHP Search -- Agentic")
    print(f"Seasons : {search_seasons}")
    print(
        f"Trials  : {args.n_trials}"
        f"  (parallel={args.n_parallel})"
    )
    print(f"Model   : {args.model}\n")

    torch.cuda.empty_cache()
    torch.serialization.add_safe_globals([dtyp])

    df_train, num_timesteps, parquet_paths = (
        _load_seasons(search_seasons)
    )
    n_plays = len(
        set(df_train.index.get_level_values(0).tolist())
    )
    print(
        f"Training trajectories: {n_plays}"
        f"  timesteps: {num_timesteps}"
    )

    proc_params = (reward_sign, [0], max_window, downsampling)

    best, history = agentic_hp_search(
        df_train=df_train,
        num_timesteps=num_timesteps,
        parquet_paths=(
            parquet_paths if args.n_parallel > 1 else None
        ),
        proc_params=(
            proc_params if args.n_parallel > 1 else None
        ),
        n_trials=args.n_trials,
        n_parallel=args.n_parallel,
        claude_model=args.model,
    )

    print("\n\n" + "="*64)
    print("SEARCH COMPLETE")
    print("="*64)
    lz = best["log_z"]
    ya = best["yards_approx"]
    print(f"Best log_Z : {lz:.4f}  (yards~{ya:.1f})")
    print(f"Best HPs   :\n{json.dumps(best['hps'], indent=2)}")

    out_path = os.path.join(
        os.path.dirname(processed_dir),
        "hp_search_"
        + "_".join(f"{t}{y}" for t, y in search_seasons)
        + ".json",
    )
    with open(out_path, "w") as f:
        json.dump(
            {"best": best, "history": history},
            f,
            indent=2,
            default=str,
        )
    print(f"\nResults saved -> {out_path}")
