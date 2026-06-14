
from math import log
import numpy as np

from common.constants import all_players, dtyp, fps


train_index, test_index = 15, 1

x_field_min, x_field_max = 0, 100
x_endzone_min, x_endzone_max = x_field_min - 10, x_field_max + 10

y_mid, y_bnd = 0.000000, 26.666667
y_hash = 23.583333 - y_bnd + y_mid

num_games = 20
game_time = 4 * 15 * 60
num_drives = game_time // 4

max_play_frames = (fps * 60) / 4.00

catg_idxs = [0, ]
# catg_idxs = [0, 1, ]

max_window = 180
downsampling = 1

km_alpha_decay = 0.15
km_num_clusters = 16

kmnc = float(km_num_clusters)

reward_threshold = 1.00
reward_threshold = abs(reward_threshold)
if reward_threshold > x_field_max or reward_threshold < 1 / x_field_max:
    reward_threshold = 1.00
if reward_threshold > 1.00:  reward_threshold /= x_field_max
reward_threshold += 1e-6

reward_sign = False
reward_scale = 1.00
reward_beta = 10.00

xgb_num_estimators = 100
xgb_max_depth = 4
xgb_learning_rate = 0.1
xgb_subsample = 0.8
xgb_colsample_bytree = 0.8

max_score_diff = 21 / 100

snap_x_range = (20 / x_field_max, 80 / x_field_max)

cut_two_min = True

fstv, lstv = dtyp(-1.0), dtyp(+1.0)

col_types = {
    "str": str,
    "int": np.int32,
    "float": np.float32,
    "bool": np.bool_,
}

col_fillnas = {
    "str": col_types["str"]("00000"),
    "int": col_types["int"](0),
    "float": col_types["float"](np.nan),
    "bool": col_types["bool"](False),
}

shape_players = all_players * 1

play_catgs = ["Pass", "Rush", "Two Point Conversion"]

state_mapping = {
    0: ("float", "game_season", 0.0, 1.0, "uniform"),
    1: ("float", "game_week", 0.0, 1.0, "uniform"),
    2: ("float", "play_defense_drive", 0.0, 1.0, "uniform"),
    3: ("float", "play_defense_game", 0.0, 1.0, "uniform"),
    4: ("bool", "play_defense_in_lead", False, True, "bernoulli"),
    5: ("bool", "play_defense_is_home_team", False, True, "bernoulli"),
    6: ("float", "play_defense_play", 0.0, 1.0, "uniform"),
    7: ("float", "play_defense_timeouts", 0.0, 1.0, "uniform"),
    8: ("float", "play_down", 0.0, 1.0, "uniform"),
    9: ("float", "play_formation_distance", 0.0, kmnc, "uniform"),
    10: ("float", "play_formation_entropy", 0.0, log(kmnc), "uniform"),
    11: ("float", "play_offense_drive", 0.0, 1.0, "uniform"),
    12: ("float", "play_offense_game", 0.0, 1.0, "uniform"),
    13: ("bool", "play_offense_in_lead", False, True, "bernoulli"),
    14: ("bool", "play_offense_is_home_team", False, True, "bernoulli"),
    15: ("float", "play_offense_play", 0.0, 1.0, "uniform"),
    16: ("float", "play_offense_timeouts", 0.0, 1.0, "uniform"),
    17: ("float", "play_quarter", 0.0, 1.0, "uniform"),
    18: ("float", "play_score_difference", -1.0, +1.0, "normal"),
    19: ("float", "play_score_equal", False, True, "bernoulli"),
    20: ("bool", "play_snap_center", False, True, "bernoulli"),
    21: ("bool", "play_snap_left", False, True, "bernoulli"),
    22: ("bool", "play_snap_right", False, True, "bernoulli"),
    23: ("float", "play_snap_x", 0.0, 1.0, "uniform"),
    24: ("float", "play_snap_y", -1.0, +1.0, "normal"),
    25: ("float", "play_time", 0.0, 1.0, "uniform"),
    26: ("float", "play_yards_needed", 0.0, 1.0, "uniform"),
    27: ("float", "play_true_length", 0.0, 10000.0, "uniform"),
    28: ("float", "player_ax", -1.0, +1.0, "normal"),
    29: ("float", "player_ay", -1.0, +1.0, "normal"),
    30: ("bool", "player_position_db", False, True, "bernoulli"),
    31: ("bool", "player_position_dl", False, True, "bernoulli"),
    32: ("bool", "player_position_lb", False, True, "bernoulli"),
    33: ("bool", "player_position_na", False, True, "bernoulli"),
    34: ("bool", "player_position_ol", False, True, "bernoulli"),
    35: ("bool", "player_position_qb", False, True, "bernoulli"),
    36: ("bool", "player_position_rb", False, True, "bernoulli"),
    37: ("bool", "player_position_te", False, True, "bernoulli"),
    38: ("bool", "player_position_wr", False, True, "bernoulli"),
    39: ("bool", "player_post_snap", False, True, "bernoulli"),
    40: ("bool", "player_pre_snap", False, True, "bernoulli"),
    41: ("float", "player_vx", -1.0, +1.0, "normal"),
    42: ("float", "player_vy", -1.0, +1.0, "normal"),
    43: ("float", "player_x", -0.2, +1.2, "uniform"),
    44: ("float", "player_y", -1.1, +1.1, "normal"),
    45: ("float", "player_yabs", 0.0, 1.1, "uniform"),
    46: ("bool", "player_defense", False, True, "bernoulli"),
    47: ("bool", "player_offense", False, True, "bernoulli"),
}

season_count = 1
match_count = 2
