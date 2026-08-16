
from __future__ import annotations

from dataclasses import dataclass, field

from data.constants import play_catgs
from run_config import DEFAULTS


@dataclass(frozen=True)
class HPField:
    key: str
    label: str
    group: str
    kind: str  # "int" | "float" | "str" | "bool"
    default: object
    description: str=""
    suggested_values: tuple=field(default_factory=tuple)

GROUPS: list[str] = [
    "Optimizer",
    "Model",
    "Encoder",
    "Decoder",
    "SDE",
    "Reward",
    "Data",
    "Search",
]

HP_FIELDS: list[HPField] = [
    HPField(
        key="dim_play", label="Play Embedding Dim",
        group="Encoder",
        kind="int", default=DEFAULTS["dim_play"],
        description="Play-level embedding dimension.",
        suggested_values=(64, 128, 256),
    ),
    HPField(
        key="dim_player", label="Player Embedding Dim",
        group="Encoder",
        kind="int", default=DEFAULTS["dim_player"],
        description="Per-player embedding dimension.",
        suggested_values=(64, 128, 256),
    ),
    HPField(
        key="lags_def_tm", label="Defense Teammate Lags",
        group="Encoder",
        kind="str", default=DEFAULTS["lags_def_tm"],
        description=(
            "Comma-separated frame-lag offsets used to build defensive "
            "players' history features relative to their teammates."
        ),
    ),
    HPField(
        key="lags_off_tm", label="Offense Teammate Lags",
        group="Encoder",
        kind="str", default=DEFAULTS["lags_off_tm"],
        description=(
            "Comma-separated frame-lag offsets used to build offensive "
            "players' history features relative to their teammates."
        ),
    ),
    HPField(
        key="lags_def_op", label="Defense Opponent Lags",
        group="Encoder",
        kind="str", default=DEFAULTS["lags_def_op"],
        description=(
            "Comma-separated frame-lag offsets used to build defensive "
            "players' history features relative to their opponents."
        ),
    ),
    HPField(
        key="lags_off_op", label="Offense Opponent Lags",
        group="Encoder",
        kind="str", default=DEFAULTS["lags_off_op"],
        description=(
            "Comma-separated frame-lag offsets used to build offensive "
            "players' history features relative to their opponents."
        ),
    ),
    HPField(
        key="expansion", label="MLP Expansion",
        group="Encoder",
        kind="int", default=DEFAULTS["expansion"],
        description=(
            "MLP width expansion factor (output = expansion^depth * dim)."
        ),
        suggested_values=(1, 2, 3),
    ),
    HPField(
        key="num_heads", label="Attention Heads",
        group="Encoder",
        kind="int", default=DEFAULTS["num_heads"],
        description="Count of attention heads.",
        suggested_values=(2, 4, 8),
    ),
    HPField(
        key="dropout", label="Dropout",
        group="Encoder",
        kind="float", default=DEFAULTS["dropout"],
        description=(
            "Dropout probability across all MLP and attention blocks."
        ),
        suggested_values=(0.0, 0.05, 0.10, 0.20),
    ),

    HPField(
        key="diff_eq_dim_middle", label="SDE Hidden Dim",
        group="SDE",
        kind="int", default=DEFAULTS["diff_eq_dim_middle"],
        description="Hidden dim of SDE networks.",
        suggested_values=(16, 32, 64),
    ),
    HPField(
        key="drift_size", label="Drift Network Depth",
        group="SDE",
        kind="int", default=DEFAULTS["drift_size"],
        description="Depth of the SDE drift network (must be 2-4).",
        suggested_values=(2, 3, 4),
    ),
    HPField(
        key="diffusion_size", label="Diffusion Network Depth",
        group="SDE",
        kind="int", default=DEFAULTS["diffusion_size"],
        description="Depth of the SDE diffusion network (must be 0-6).",
        suggested_values=(1, 2, 3, 4),
    ),
    HPField(
        key="dt", label="SDE Step Size",
        group="SDE",
        kind="float", default=DEFAULTS["dt"],
        description=(
            "Euler integration step size for the SDE solver "
            "(must be between 1e-13 and 1e-1)."
        ),
        suggested_values=(1e-3, 1e-2, 5e-2),
    ),
    HPField(
        key="final_dim", label="Shared Hidden Dim",
        group="Model",
        kind="int", default=DEFAULTS["final_dim"],
        description=(
            "Shared hidden dim: "
            "out of the encoder, through the SDE, into the decoder."
        ),
        suggested_values=(32, 64, 128),
    ),
    HPField(
        key="pow_iters", label="Power Iterations",
        group="Model",
        kind="int", default=DEFAULTS["pow_iters"],
        description=(
            "Spectral-norm power iterations per forward pass "
            "(must be 1-4)."
        ),
    ),
    HPField(
        key="noise_floor", label="PF Noise Floor",
        group="Model",
        kind="float", default=DEFAULTS["noise_floor"],
        description=(
            "Lower bound on the per-step noise scale added to the "
            "latent during training."
        ),
    ),
    HPField(
        key="noise_ceiling", label="PF Noise Ceiling",
        group="Model",
        kind="float", default=DEFAULTS["noise_ceiling"],
        description=(
            "Upper bound on the per-step noise scale added to the "
            "latent during training."
        ),
    ),
    HPField(
        key="noise_decay", label="PF Noise Decay",
        group="Model",
        kind="float", default=DEFAULTS["noise_decay"],
        description=(
            "Fraction of a trial's total gradient steps "
            "used as the latent noise schedule's decay gap."
        ),
    ),
    HPField(
        key="noise_exp", label="PF Noise Exponent",
        group="Model",
        kind="float", default=DEFAULTS["noise_exp"],
        description=(
            "Exponent controlling how fast the latent noise schedule "
            "decays with step count."
        ),
    ),

    HPField(
        key="batch_size", label="Batch Size",
        group="Optimizer",
        kind="int", default=DEFAULTS["batch_size"],
        description="Number of trajectories per training batch.",
        suggested_values=(4, 8, 16, 32),
    ),
    HPField(
        key="learning_rate", label="Learning Rate",
        group="Optimizer",
        kind="float", default=DEFAULTS["learning_rate"],
        description="AdamW learning rate.",
        suggested_values=(3e-5, 1e-4, 3e-4, 1e-3),
    ),
    HPField(
        key="weight_decay_rate", label="Weight Decay",
        group="Optimizer",
        kind="float", default=DEFAULTS["weight_decay_rate"],
        description="AdamW weight decay rate.",
    ),
    HPField(
        key="num_epochs", label="Epochs",
        group="Optimizer",
        kind="int", default=DEFAULTS["num_epochs"],
        description="Number of passes over the training set.",
        suggested_values=(1, 2, 3, 5),
    ),

    HPField(
        key="reward_scale", label="Reward Scale",
        group="Reward",
        kind="float", default=DEFAULTS["reward_scale"],
        description="Multiplier on the sigmoid reward.",
        suggested_values=(0.5, 1.0, 2.0, 5.0),
    ),
    HPField(
        key="reward_beta", label="Reward Beta",
        group="Reward",
        kind="float", default=DEFAULTS["reward_beta"],
        description="Steepness of the reward sigmoid."
    ),
    HPField(
        key="reward_sign", label="Reward Sign",
        group="Reward",
        kind="bool", default=DEFAULTS["reward_sign"],
        description=(
            "Flips the sign convention used inside the reward sigmoid."
        ),
    ),
    HPField(
        key="max_score_diff", label="Max Score Difference",
        group="Reward",
        kind="float", default=DEFAULTS["max_score_diff"],
        description=(
            "Normalized max score-differential cutoff used when "
            "filtering plays."
        ),
    ),

    HPField(
        key="decoder_min_stdv", label="Decoder Min Standard Deviation",
        group="Decoder",
        kind="float", default=DEFAULTS["decoder_min_stdv"],
        description=(
            "Lower bound of the decoder's predicted output standard "
            "deviation (must be >= 1e-5)."
        ),
    ),
    HPField(
        key="decoder_max_stdv", label="Decoder Max Standard Deviation",
        group="Decoder",
        kind="float", default=DEFAULTS["decoder_max_stdv"],
        description=(
            "Upper bound of the decoder's predicted output standard "
            "deviation (must be <= 1e-1)."
        ),
    ),

    HPField(
        key="seasons", label="Seasons (TEAM:YEAR, comma-separated)",
        group="Data",
        kind="str", default=DEFAULTS["seasons"],
        description=(
            "Comma-separated TEAM:YEAR pairs, e.g. 'TB:2022' or "
            "'NE:2018,TB:2021'."
        ),
    ),
    HPField(
        key="match_count", label="Matches per Season",
        group="Data",
        kind="int", default=DEFAULTS["match_count"],
        description="Cap on matches loaded per season.",
    ),
    HPField(
        key="train_ratio", label="Train Split Ratio",
        group="Data",
        kind="float", default=DEFAULTS["train_ratio"],
        description=(
            "Fraction of trajectories assigned to the training split."
        ),
    ),
    HPField(
        key="max_window", label="Max Window",
        group="Data",
        kind="int", default=DEFAULTS["max_window"],
        description="Maximum number of frames kept per play window.",
    ),
    HPField(
        key="cut_two_min", label="Cut Two-Minute Drill",
        group="Data",
        kind="bool", default=DEFAULTS["cut_two_min"],
        description="Exclude two-minute-drill situations from the dataset.",
    ),
    HPField(
        key="catg_idxs", label="Play Categories",
        group="Data",
        kind="str", default=DEFAULTS["catg_idxs"],
        description=(
            "Comma-separated play categories to include, case-insensitive "
            "\n-- e.g. 'Pass' or 'Pass, Rush'."
        ),
    ),
    HPField(
        key="snap_x_low", label="Snap X Low",
        group="Data",
        kind="float", default=DEFAULTS["snap_x_low"],
        description=(
            "Lower bound of the normalized field-x range plays must "
            "snap within to be kept."
        ),
    ),
    HPField(
        key="snap_x_high", label="Snap X High",
        group="Data",
        kind="float", default=DEFAULTS["snap_x_high"],
        description=(
            "Upper bound of the normalized field-x range plays must "
            "snap within to be kept."
        ),
    ),
    HPField(
        key="km_num_clusters", label="Formation KMeans Clusters",
        group="Data",
        kind="int", default=DEFAULTS["km_num_clusters"],
        description=(
            "Number of clusters for the formation-clustering feature."
        ),
    ),
    HPField(
        key="km_alpha_decay", label="Formation KMeans Alpha",
        group="Data",
        kind="float", default=DEFAULTS["km_alpha_decay"],
        description="Alpha decay for the formation-clustering feature.",
    ),

    HPField(
        key="n_trials", label="Total Trials",
        group="Search",
        kind="int", default=DEFAULTS["n_trials"],
        description=(
            "Total number of trial evaluations for the agentic HP search."
        ),
    ),
    HPField(
        key="n_parallel", label="Parallel Trials (GPUs)",
        group="Search",
        kind="int", default=DEFAULTS["n_parallel"],
        description=(
            "1 = sequential loop; k>1 = k-GPU batched proposals per round."
        ),
    ),
    HPField(
        key="claude_model", label="Claude Model",
        group="Search",
        kind="str", default=DEFAULTS["claude_model"],
        description=(
            "Anthropic model ID for proposing hyperparameter configurations."
        ),
    ),
]

def fields_by_group() -> dict[str, list[HPField]]:
    out: dict[str, list[HPField]] = {g: [] for g in GROUPS}
    for f in HP_FIELDS:
        out.setdefault(f.group, []).append(f)
    return out

def defaults_as_strings() -> dict[str, str]:
    return {f.key: str(f.default) for f in HP_FIELDS}

_CATG_LABEL_OVERRIDES = {"Two Point Conversion": "2P"}
CATG_LABELS: dict[int, str] = {
    i: _CATG_LABEL_OVERRIDES.get(name, name)
    for i, name in enumerate(play_catgs)
}
_CATG_LABEL_TO_IDX: dict[str, int] = {
    label.lower(): idx for idx, label in CATG_LABELS.items()
}

def catg_idxs_to_labels(raw: str) -> str:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:  continue
        try:
            out.append(CATG_LABELS.get(int(tok), tok))
        except ValueError:
            out.append(tok)
    return ", ".join(out)

def catg_labels_to_idxs(raw: str) -> str:
    out = []
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:  continue
        if tok.lower() in _CATG_LABEL_TO_IDX:
            out.append(str(_CATG_LABEL_TO_IDX[tok.lower()]))
        else:
            out.append(tok)
    return ",".join(out)

def catg_label_tokens_valid(raw: str) -> bool:
    for tok in raw.split(","):
        tok = tok.strip()
        if not tok:  continue
        if tok.lower() in _CATG_LABEL_TO_IDX:  continue
        try:
            int(tok)
        except ValueError:
            return False
    return True
