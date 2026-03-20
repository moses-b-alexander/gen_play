
import torch

from common.constants import fps, team_players
from data.constants import x_field_max, y_bnd


torch.backends.cudnn.deterministic = True

train_ratio = 0.90

learning_rate, weight_decay_rate = 1e-4, 1e-5

max_window = 1000
downsampling = 10

batch_size = 8

final_dim = 64

downsample_rate = fps // downsampling
if downsample_rate < 2 or downsample_rate > fps // 2:  downsample_rate = 1
max_deltas = [
    (2.000 / x_field_max) * downsample_rate,
    (1.000 / y_bnd) * downsample_rate
]
max_dx, max_dy = max_deltas[0], max_deltas[1]

global_dim, agent_dim = 28, 21
state_dim = global_dim + agent_dim

action_dim = 2

play_tensor_indices = {
    "game": [0, 1],
    "def_count": [2, 3, 6],
    "def_status": [4, 5, 7],
    "down": [8, 26],
    "off_count": [9, 10, 13],
    "off_status": [11, 12, 14],
    "time": [15, 23],
    "score": [16, 17],
    "snap": [18, 19, 20, 21],
    "when": [24, 25],
    "snap_y": [22],
    "length": [27],
}

player_tensor_indices = {
    "role": [30, 31, 32, 33, 35, 36, 37, 38, 39, 40],
    "acceleration": [28, 29],
    "position": [45, 46],
    "velocity": [43, 44],
    "x": [28, 43, 45],
    "y": [29, 44, 46],
    "team": [47, 48],
    "snap": [41, 42],
}
player_tensor_indices = {
    k: list(map(lambda i: i - global_dim, v))
    for k, v in player_tensor_indices.items()
}

model_name = "pf"
