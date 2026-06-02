
from math import log
import numpy as np

from common.constants import all_players, dtyp, fps, team_players


train_index, test_index = 15, 1

x_field_min, x_field_max = 0, 100
x_endzone_min, x_endzone_max = x_field_min - 10, x_field_max + 10

y_mid, y_bnd = 0.000000, 26.666667
y_hash = 23.583333 - y_bnd + y_mid

num_games = 20
game_time = 4 * 15 * 60
num_drives = game_time // 4

max_play_frames = (fps * 60) / 4.00

reward_sign = False

reward_threshold = 1.00
reward_threshold = abs(reward_threshold)
if reward_threshold > x_field_max or reward_threshold < 1 / x_field_max:
    reward_threshold = 1.00
if reward_threshold > 1.00:  reward_threshold /= x_field_max

reward_scale = 1.00

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
    9: ("float", "play_offense_drive", 0.0, 1.0, "uniform"),
    10: ("float", "play_offense_game", 0.0, 1.0, "uniform"),
    11: ("bool", "play_offense_in_lead", False, True, "bernoulli"),
    12: ("bool", "play_offense_is_home_team", False, True, "bernoulli"),
    13: ("float", "play_offense_play", 0.0, 1.0, "uniform"),
    14: ("float", "play_offense_timeouts", 0.0, 1.0, "uniform"),
    15: ("float", "play_quarter", 0.0, 1.0, "uniform"),
    16: ("float", "play_score_difference", -1.0, +1.0, "normal"),
    17: ("float", "play_score_equal", False, True, "bernoulli"),
    18: ("bool", "play_snap_center", False, True, "bernoulli"),
    19: ("bool", "play_snap_left", False, True, "bernoulli"),
    20: ("bool", "play_snap_right", False, True, "bernoulli"),
    21: ("float", "play_snap_x", 0.0, 1.0, "uniform"),
    22: ("float", "play_snap_y", -1.0, +1.0, "normal"),
    23: ("float", "play_time", 0.0, 1.0, "uniform"),
    24: ("float", "play_time_after_snap", 0.0, 1.0, "uniform"),
    25: ("float", "play_time_before_snap", 0.0, 1.0, "uniform"),
    26: ("float", "play_yards_needed", 0.0, 1.0, "uniform"),
    27: ("float", "play_qb_pressure", 0.0, 2.0, "uniform"),
    28: ("float", "play_tgt_sep", 0.0, 2.0, "uniform"),
    29: ("float", "play_tgt_depth", -0.2, +1.2, "uniform"),
    30: ("float", "play_true_length", 0.0, 10000.0, "uniform"),
    31: ("float", "player_ax", -1.0, +1.0, "normal"),
    32: ("float", "player_ay", -1.0, +1.0, "normal"),
    33: ("bool", "player_position_db", False, True, "bernoulli"),
    34: ("bool", "player_position_dl", False, True, "bernoulli"),
    35: ("bool", "player_position_lb", False, True, "bernoulli"),
    36: ("bool", "player_position_ls", False, True, "bernoulli"),
    37: ("bool", "player_position_na", False, True, "bernoulli"),
    38: ("bool", "player_position_ol", False, True, "bernoulli"),
    39: ("bool", "player_position_pt", False, True, "bernoulli"),
    40: ("bool", "player_position_qb", False, True, "bernoulli"),
    41: ("bool", "player_position_rb", False, True, "bernoulli"),
    42: ("bool", "player_position_te", False, True, "bernoulli"),
    43: ("bool", "player_position_wr", False, True, "bernoulli"),
    44: ("bool", "player_post_snap", False, True, "bernoulli"),
    45: ("bool", "player_pre_snap", False, True, "bernoulli"),
    46: ("float", "player_vx", -1.0, +1.0, "normal"),
    47: ("float", "player_vy", -1.0, +1.0, "normal"),
    48: ("float", "player_x", -0.2, +1.2, "uniform"),
    49: ("float", "player_y", -1.0, +1.0, "normal"),
    50: ("float", "player_sep", 0.0, 2.0, "uniform"),
    51: ("float", "player_yabs", 0.0, 1.0, "uniform"),
    52: ("bool", "player_defense", False, True, "bernoulli"),
    53: ("bool", "player_offense", False, True, "bernoulli"),
}

season_count = 1
match_count = 2
