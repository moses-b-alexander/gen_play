
import torch


torch.backends.cudnn.deterministic = True

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
