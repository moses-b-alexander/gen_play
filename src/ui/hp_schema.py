
from __future__ import annotations

from dataclasses import dataclass, field

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
    "Encoder",
    "Diffusion / SDE",
    "Model (Shared)",
    "Optimizer",
    "Reward",
    "Decoder / Output Noise",
    "Data & Season",
    "Search Control",
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
        description=(
            "Per-player embedding dimension. num_heads must evenly "
            "divide this."
        ),
        suggested_values=(64, 128, 256),
    ),
    HPField(
        key="num_heads", label="Attention Heads",
        group="Encoder",
        kind="int", default=DEFAULTS["num_heads"],
        description="Attention heads. Must divide dim_player evenly.",
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
        key="expansion", label="MLP Expansion",
        group="Encoder",
        kind="int", default=DEFAULTS["expansion"],
        description=(
            "MLP width expansion factor (output = expansion^depth * dim)."
        ),
        suggested_values=(1, 2, 3),
    ),

    HPField(
        key="drift_size", label="Drift Network Depth",
        group="Diffusion / SDE",
        kind="int", default=DEFAULTS["drift_size"],
        description="Depth of the SDE drift network.",
        suggested_values=(1, 2, 3, 4),
    ),
    HPField(
        key="diffusion_size", label="Diffusion Network Depth",
        group="Diffusion / SDE",
        kind="int", default=DEFAULTS["diffusion_size"],
        description="Depth of the SDE diffusion network.",
        suggested_values=(1, 2, 3, 4),
    ),
    HPField(
        key="diff_eq_dim_middle", label="SDE Hidden Width",
        group="Diffusion / SDE",
        kind="int", default=DEFAULTS["diff_eq_dim_middle"],
        description="Hidden width of SDE networks. Should be <= final_dim.",
        suggested_values=(16, 32, 64),
    ),
    HPField(
        key="sde_steps", label="Euler Steps",
        group="Diffusion / SDE",
        kind="int", default=DEFAULTS["sde_steps"],
        description=(
            "Number of output evaluation points passed to sdeint "
            "(ts=linspace(0,1,steps)); actual integration granularity "
            "is set by dt."
        ),
        suggested_values=(2, 4, 6, 8),
    ),

    HPField(
        key="final_dim", label="Shared Hidden Dim",
        group="Model (Shared)",
        kind="int", default=DEFAULTS["final_dim"],
        description=(
            "Shared hidden dim (dim_h) feeding into the SDE and decoder."
        ),
        suggested_values=(32, 64, 128),
    ),
    HPField(
        key="pow_iters", label="Power Iterations",
        group="Model (Shared)",
        kind="int", default=DEFAULTS["pow_iters"],
        description="Spectral-norm power iterations per forward pass.",
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
        description=(
            "Multiplier on the sigmoid reward r(yg) = scale * sigmoid(yg). "
            "Acts as inverse temperature: higher values sharpen the reward "
            "signal toward high-yardage plays."
        ),
        suggested_values=(0.5, 1.0, 2.0, 5.0),
    ),
    HPField(
        key="reward_beta", label="Reward Beta",
        group="Reward",
        kind="float", default=DEFAULTS["reward_beta"],
        description="Beta shaping term applied alongside reward_scale.",
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
        key="max_score_diff", label="Max Score Diff",
        group="Reward",
        kind="float", default=DEFAULTS["max_score_diff"],
        description=(
            "Normalized max score-differential cutoff used when "
            "filtering plays."
        ),
    ),

    HPField(
        key="decoder_min_stdv", label="Decoder Min Stdv",
        group="Decoder / Output Noise",
        kind="float", default=DEFAULTS["decoder_min_stdv"],
        description=(
            "Lower bound of the decoder's predicted output stdev "
            "(must be >= 1e-6)."
        ),
    ),
    HPField(
        key="decoder_max_stdv", label="Decoder Max Stdv",
        group="Decoder / Output Noise",
        kind="float", default=DEFAULTS["decoder_max_stdv"],
        description=(
            "Upper bound of the decoder's predicted output stdev "
            "(must be <= 1e-2)."
        ),
    ),
    HPField(
        key="noise_floor", label="PF Noise Floor",
        group="Decoder / Output Noise",
        kind="float", default=DEFAULTS["noise_floor"],
        description=(
            "Lower bound on the per-step noise scale added to the "
            "PF's hidden state during training."
        ),
    ),
    HPField(
        key="noise_ceiling", label="PF Noise Ceiling",
        group="Decoder / Output Noise",
        kind="float", default=DEFAULTS["noise_ceiling"],
        description=(
            "Upper bound on the per-step noise scale added to the "
            "PF's hidden state during training."
        ),
    ),
    HPField(
        key="noise_decay", label="PF Noise Decay",
        group="Decoder / Output Noise",
        kind="float", default=DEFAULTS["noise_decay"],
        description=(
            "Fraction of a trial's total gradient steps "
            "(batches_per_epoch * num_epochs) used as the PF noise "
            "schedule's decay gap."
        ),
    ),
    HPField(
        key="noise_exp", label="PF Noise Exponent",
        group="Decoder / Output Noise",
        kind="float", default=DEFAULTS["noise_exp"],
        description=(
            "Exponent controlling how fast the PF noise schedule "
            "decays with step count."
        ),
    ),

    HPField(
        key="seasons", label="Seasons (TEAM:YEAR, comma-sep)",
        group="Data & Season",
        kind="str", default=DEFAULTS["seasons"],
        description=(
            "Comma-separated TEAM:YEAR pairs, e.g. 'TB:2022' or "
            "'NE:2018,TB:2021'."
        ),
    ),
    HPField(
        key="match_count", label="Matches per Season",
        group="Data & Season",
        kind="int", default=DEFAULTS["match_count"],
        description=(
            "Cap on matches loaded per season. Only used when "
            "main.py re-processes raw data (saved=False)."
        ),
    ),
    HPField(
        key="train_ratio", label="Train Split Ratio",
        group="Data & Season",
        kind="float", default=DEFAULTS["train_ratio"],
        description=(
            "Fraction of trajectories assigned to the training split."
        ),
    ),
    HPField(
        key="max_window", label="Max Window",
        group="Data & Season",
        kind="int", default=DEFAULTS["max_window"],
        description="Maximum number of frames kept per play window.",
    ),
    HPField(
        key="cut_two_min", label="Cut Two-Minute Drill",
        group="Data & Season",
        kind="bool", default=DEFAULTS["cut_two_min"],
        description="Exclude two-minute-drill situations from the dataset.",
    ),
    HPField(
        key="catg_idxs", label="Category Indices",
        group="Data & Season",
        kind="str", default=DEFAULTS["catg_idxs"],
        description=(
            "Comma-separated play-category indices to include, "
            "e.g. '0' or '0,1'."
        ),
    ),
    HPField(
        key="km_alpha_decay", label="Formation KMeans Alpha Decay",
        group="Data & Season",
        kind="float", default=DEFAULTS["km_alpha_decay"],
        description="Alpha decay for the formation-clustering feature.",
    ),
    HPField(
        key="km_num_clusters", label="Formation KMeans Clusters",
        group="Data & Season",
        kind="int", default=DEFAULTS["km_num_clusters"],
        description=(
            "Number of clusters for the formation-clustering feature."
        ),
    ),
    HPField(
        key="snap_x_range", label="Snap X Range (lo,hi)",
        group="Data & Season",
        kind="str", default=DEFAULTS["snap_x_range"],
        description=(
            "Normalized field-x range (comma-separated lo,hi) plays "
            "must snap within to be kept."
        ),
    ),

    HPField(
        key="n_trials", label="Total Trials",
        group="Search Control",
        kind="int", default=DEFAULTS["n_trials"],
        description=(
            "Total number of trial evaluations for the agentic HP search."
        ),
    ),
    HPField(
        key="n_parallel", label="Parallel Trials (GPUs)",
        group="Search Control",
        kind="int", default=DEFAULTS["n_parallel"],
        description=(
            "1 = sequential loop; k>1 = k-GPU batched proposals per round."
        ),
    ),
    HPField(
        key="claude_model", label="Claude Model",
        group="Search Control",
        kind="str", default=DEFAULTS["claude_model"],
        description=(
            "Anthropic model ID used to propose hyperparameter "
            "configurations."
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
