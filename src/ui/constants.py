
from common.constants import fps


max_configs = 10
max_name_length = 24

active_slug = "current_config"

groups = [
    "Optimizer", "Model",
    "Encoder", "Decoder", "SDE",
    "Reward", "Data",
    "Search",
]

nav_items = [
    ("Hyperparameters", "tune", True),
    ("Search Runs", "science", False),
    ("Training", "play_circle", False),
    ("Results", "insights", False),
    ("Settings", "settings", False),
]

exclusive_eps = 1e-6

clamp_bounds = {
    "dropout": (0.0, 1.0, False),
    "dim_play": (1, float("inf"), False),
    "dim_player": (1, float("inf"), False),
    "final_dim": (1, float("inf"), False),
    "diff_eq_dim_middle": (1, float("inf"), False),
    "drift_size": (2, 4, False),
    "diffusion_size": (0, 6, False),
    "pow_iters": (1, 4, False),
    "batch_size": (1, float("inf"), False),
    "num_epochs": (1, float("inf"), False),
    "n_parallel": (1, float("inf"), False),
    "n_trials": (1, float("inf"), False),
    "decoder_min_stdv": (1e-5, 1e-1, False),
    "decoder_max_stdv": (1e-5, 1e-1, False),
    "dt": (1e-13, 1e-1, False),
}

paired_bounds = {
    "decoder_min_stdv": ("decoder_max_stdv", "lower"),
    "decoder_max_stdv": ("decoder_min_stdv", "upper"),
}

list_clamp_bounds = {
    "lags_def_tm": (0, 1 + (fps * 30)), "lags_off_tm": (0, 1 + (fps * 30)),
    "lags_def_op": (0, 1 + (fps * 30)), "lags_off_op": (0, 1 + (fps * 30)),
}

group_icons = {
    "Encoder": "cable",
    "SDE": "waves",
    "Model": "hub",
    "Optimizer": "trending_up",
    "Reward": "military_tech",
    "Decoder": "graphic_eq",
    "Data": "calendar_month",
    "Search": "smart_toy",
}

colors = {
    "bg": "#14151b",
    "bg_alt": "#191a22",
    "surface": "#1d1f29",
    "surface_hover": "#242631",
    "border": "#2b2e3a",
    "accent": "#7c5cff",
    "accent_soft": "#7c5cff26",
    "success": "#34d399",
    "danger": "#f87171",
    "warning": "#fbbf24",
    "text": "#e7e8ee",
    "text_muted": "#9297ab",
}
