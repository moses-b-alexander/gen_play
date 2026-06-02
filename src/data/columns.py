
from common.constants import all_players
from data.constants import col_fillnas, col_types


play_rename_cols = {
    "drive_uuid": "play_drive_uuid",
    "game_id": "play_game_uuid",
    "play_quarter_clock_remaining": "play_clock",
    "play_yards_to_go": "play_yards_needed",
    "play_yardline": "play_snap_x",
    "play_home_timeouts_remaining": "play_home_timeouts",
    "play_away_timeouts_remaining": "play_away_timeouts",
    "play_pass_yards_air": "play_pass_air_yards",
    "play_pass_yards_after_catch": "play_pass_ground_yards",
    "play_yards_run": "play_rush_yards",
    "play_offense_team_short_name": "play_offense_team",
    "play_defense_team_short_name": "play_defense_team",
    "szn": "play_season",
}
track_rename_cols = {
    "player_id": "player_uuid",
    "x": "player_x",
    "y": "player_y",
    "timestamp": "now_time",
    "time_since_snap": "frame_time_snap",
}

play_cols = {
    "play_defense_is_home_team": "bool",
    "play_defense_score": "int",
    "play_defense_team": "str",
    "play_defense_timeouts": "int",
    "play_down": "int",
    "play_drive_uuid": "str",
    "play_game_uuid": "str",
    "play_offense_is_home_team": "bool",
    "play_offense_score": "int",
    "play_offense_team": "str",
    "play_offense_timeouts": "int",
    "play_quarter": "int",
    "play_season": "int",
    "play_snap_center": "bool",
    "play_snap_left": "bool",
    "play_snap_right": "bool",
    "play_snap_x": "float",
    "play_snap_y": "float",
    "play_time": "float",
    "play_uuid": "str",
    "play_yards_gained": "float",
    "play_yards_needed": "float",
}
track_cols = {
    "frame_time": "float",
    "frame_time_snap": "float",
    "play_day": "int",
    "play_month": "int",
    "play_playoffs": "bool",
    "play_year": "int",
    "play_uuid": "str",
    "player_defense": "bool",
    "player_uuid": "str",
    "player_offense": "bool",
    "player_position_db": "bool",
    "player_position_dl": "bool",
    "player_position_lb": "bool",
    "player_position_ls": "bool",
    "player_position_na": "bool",
    "player_position_ol": "bool",
    "player_position_pt": "bool",
    "player_position_qb": "bool",
    "player_position_rb": "bool",
    "player_position_te": "bool",
    "player_position_wr": "bool",
    "player_post_snap": "bool",
    "player_ax": "float",
    "player_ay": "float",
    "player_dx": "float",
    "player_dy": "float",
    "player_vx": "float",
    "player_vy": "float",
    "player_x": "float",
    "player_y": "float",
    "player_sep": "float",
    "player_yabs": "float",
}

date_cols = {
    "play_day": "int",
    "play_month": "int",
    "play_year": "int",
}

player_cols = {
    "player_defense": "bool",
    "player_uuid": "str",
    "player_offense": "bool",
    "player_position_db": "bool",
    "player_position_dl": "bool",
    "player_position_lb": "bool",
    "player_position_ls": "bool",
    "player_position_na": "bool",
    "player_position_ol": "bool",
    "player_position_pt": "bool",
    "player_position_qb": "bool",
    "player_position_rb": "bool",
    "player_position_te": "bool",
    "player_position_wr": "bool",
    "player_post_snap": "bool",
    "player_ax": "float",
    "player_ay": "float",
    "player_dx": "float",
    "player_dy": "float",
    "player_vx": "float",
    "player_vy": "float",
    "player_x": "float",
    "player_y": "float",
    "player_sep": "float",
    "player_yabs": "float",
}

pivot_cols = {
    f"{pc}-{n:02d}": t
    for pc, t in player_cols.items() for n in range(1, all_players + 1)
}

play_col_fillnas = {c: col_fillnas[t] for c, t in play_cols.items()}
track_col_fillnas = {
    c: col_fillnas[t]
    for c, t in track_cols.items() if c not in player_cols
}
for n in range(1, all_players + 1):
    track_col_fillnas |= {
        f"{c}-{n:02d}": col_fillnas[t]
        for c, t in track_cols.items() if c in player_cols
    }

play_col_types = {c: col_types[t] for c, t in play_cols.items()}
track_col_types = {
    c: col_types[t]
    for c, t in track_cols.items() if c not in player_cols
}
for n in range(1, all_players + 1):
    track_col_types |= {
        f"{c}-{n:02d}": col_types[t]
        for c, t in track_cols.items() if c in player_cols
    }
