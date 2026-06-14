
import torch

from common.constants import fps
from data.constants import downsampling, x_field_max, y_bnd


torch.backends.cudnn.deterministic = True

train_ratio = 0.90

learning_rate, weight_decay_rate = 1e-4, 1e-5

batch_size = 8

final_dim = 64

downsample_rate = fps // downsampling
if downsample_rate < 2 or downsample_rate > fps // 2:  downsample_rate = 1
max_deltas = [
    (4.000 / x_field_max) * downsample_rate,
    (2.000 / abs(y_bnd)) * downsample_rate
]
max_dx, max_dy = max_deltas[0], max_deltas[1]

global_dim, agent_dim = 28, 20
state_dim = global_dim + agent_dim

action_dim = 2

play_tensor_indices = {
    "game": [0, 1],
    "def_count": [2, 3, 6],
    "def_status": [4, 5, 7],
    "down": [8, 26],
    "off_count": [11, 12, 15],
    "off_status": [13, 14, 16],
    "time": [17, 25],
    "score": [18, 19],
    "snap": [20, 21, 22, 23],
    "snap_y": [24],
    "length": [27],
}

player_tensor_indices = {
    "role": [30, 31, 32, 33, 34, 35, 36, 37, 38],
    "acceleration": [28, 29],
    "position": [43, 44],
    "velocity": [41, 42],
    "x": [28, 41, 43],
    "y": [29, 42, 44, 45],
    "team": [46, 47],
    "snap": [39, 40],
}
player_tensor_indices = {
    k: list(map(lambda i: i - global_dim, v))
    for k, v in player_tensor_indices.items()
}

model_name = "pf"
