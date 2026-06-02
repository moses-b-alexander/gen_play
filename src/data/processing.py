
import gzip
import json
import numpy as np
import os
import pandas as pd

from ai.constants import max_dx, max_dy
from common.constants import (
    all_players, dtyp, fps, seed, team_players
)
from common.dirs import plays_path, tracking_path
from data.columns import (
    date_cols, pivot_cols,
    play_col_fillnas, play_col_types, play_rename_cols,
    player_cols,
    track_col_types, track_rename_cols
)
from data.constants import (
    col_fillnas, col_types, fstv, game_time, lstv, num_drives, num_games,
    play_catgs, reward_scale, train_index, test_index,
    x_field_min, x_field_max, y_bnd, y_hash
)
from data.names import (
    action_flattened_colnames,
    frame_colnames, metadata_colnames, num_pos_colnames,
    old_colnames, play_remove_colnames,
    play_fill_colnames, play_state_colnames,
    player_bool_flattened_colnames, player_float_flattened_colnames,
    player_location_flattened_colnames, player_state_flattened_colnames,
    player_uuid_colnames,
    reward_colnames,
    track_remove_colnames
)


def make_empty(
    df_id: str, first_row: pd.Series, pad_size: int, pad_val: bool
) -> pd.DataFrame:
    pad_index = [df_id] * pad_size

    play_state_colnames_1 = ["game_playoffs"] + [
        c for c in play_state_colnames if c != "play_true_length"
    ] + ["play_yards_gained"]
    empty_play_array = np.full(
        (pad_size, len(play_state_colnames_1)), fill_value=None)
    for i, c in enumerate(play_state_colnames_1):
        empty_play_array[:, i] = col_types["float"](first_row[c])
    empty_play_df = pd.DataFrame(
        empty_play_array, index=pad_index, columns=play_state_colnames_1)
    empty_play_df["play_type"] = first_row["play_type"]

    empty_bool_array = np.full(
        (pad_size, len(player_bool_flattened_colnames)),
        fill_value=col_types["float"](np.nan)
    )
    empty_bool_df = pd.DataFrame(
        empty_bool_array, index=pad_index,
        columns=player_bool_flattened_colnames
    )

    empty_float_array = np.full(
        (pad_size, len(player_float_flattened_colnames)),
        fill_value=col_types["float"](np.nan)
    )
    empty_float_df = pd.DataFrame(
        empty_float_array, index=pad_index,
        columns=player_float_flattened_colnames
    )

    empty_df = pd.concat(
        [empty_play_df, empty_bool_df, empty_float_df], axis=1)
    empty_df["play_padded"] = col_types["float"](pad_val)
    empty_df["play_real"] = col_types["float"](True)
    empty_df = empty_df.copy()

    return empty_df

def get_data(
    tm: str, szn: str, mct: int, sd: str=""
) -> list[pd.DataFrame]:

    play_file = \
        os.path.join(plays_path, f"tb12_plays_dataset_{szn}_{(szn+1)}.csv")
    play = pd.read_csv(play_file)
    play = play.copy()

    num_plays = play.shape[0]
    play.loc[play.play_defense_team_short_name == "LA",
        "play_defense_team_short_name"] = "LAR"
    play.loc[play.play_offense_team_short_name == "LA",
        "play_offense_team_short_name"] = "LAR"
    play["szn"] = szn

    play = play.reset_index(drop=False)
    play = play.drop(index=
        play.loc[(play.play_offense_team_short_name != tm) &
            (play.play_defense_team_short_name != tm),].index)
    play = play.copy()

    play = play.drop(columns=old_colnames, errors="ignore")
    play = play.drop(columns=play_remove_colnames, errors="ignore")

    play = play.drop(index=play.loc[play.play_uuid.isna(),].index)
    play = play.drop(index=play.loc[play.week.isna(),].index)
    play = play.drop(index=play.loc[play.play_quarter == 5,].index)
    play = play.drop(index=
        play.loc[play.play_yards_gained < -x_field_max - 1e-6,].index)
    play = play.drop(index=
        play.loc[play.play_yards_gained > +x_field_max + 1e-6,].index)
    play = play.drop(index=
        play.loc[play.play_quarter_clock_remaining.isna(),].index)
    play = play.drop(index=play.loc[
        play.play_down_negated.astype("boolean").fillna(False),
    ].index)
    play = play.drop(index=play.loc[
        play.play_fumbled.astype("boolean").fillna(False),
    ].index)
    play = play.drop(index=play.loc[
        play.play_pass_dropped.astype("boolean").fillna(False),
    ].index)
    play = play.drop(index=play.loc[
        play.play_pass_intercepted.astype("boolean").fillna(False),
    ].index)
    play = play.drop(index=play.loc[
        play.play_pass_interception_dropped.astype("boolean").fillna(False),
    ].index)
    play = play.drop(index=play.loc[
        play.play_pass_batted.astype("boolean").fillna(False),
    ].index)
    play = play.drop(index=play.loc[
        play.play_pass_thrownaway.astype("boolean").fillna(False),
    ].index)
    play = play.drop(index=play.loc[
        play.play_challenged.astype("boolean").fillna(False),
    ].index)
    play = play.drop(index=play.loc[
        play.play_lateral_success.astype("boolean").fillna(False),
    ].index)
    play = play.drop(index=play.loc[play.play_turnover_type.notna(),].index)
    play = play.drop(index=play.loc[
        (play.play_type == "Pass") & ~(play.play_pass_made.astype(bool)),
    ].index)
    play = play.drop(index=play.loc[play.play_penalty_types != "{}",].index)
    play = play.drop(index=play.loc[~play.play_type.isin(play_catgs),].index)

    play = play.rename(columns=play_rename_cols)
    play = play.drop(columns=[
        "play_pass_air_yards", "play_pass_ground_yards", "play_rush_yards",
    ])

    play["play_season"] = play.play_season.astype(int)
    play["play_quarter"] = play.play_quarter.astype(int)
    play["play_time"] = play.play_clock.astype(float)
    play["play_time"] = play.play_time.div(900000.000)
    play = play.drop(index=play.loc[play.play_time < 0.000 - 1e-6,].index)
    play = play.drop(index=play.loc[play.play_time > 1.000 + 1e-6,].index)

    play = play.sort_values(by=["play_quarter", "play_time"])

    play.loc[play.play_offense_is_home_team == False,
        "play_defense_is_home_team"] = True
    play.loc[play.play_offense_is_home_team == True,
        "play_defense_is_home_team"] = False
    play.loc[play.play_offense_is_home_team.isna(),
        ["play_offense_is_home_team", "play_defense_is_home_team"]] = False

    play.loc[play.play_offense_is_home_team, "play_offense_score"] = \
        play.loc[play.play_offense_is_home_team, "play_home_score"]
    play.loc[play.play_defense_is_home_team, "play_offense_score"] = \
        play.loc[play.play_defense_is_home_team, "play_away_score"]
    play.loc[play.play_offense_is_home_team, "play_defense_score"] = \
        play.loc[play.play_offense_is_home_team, "play_away_score"]
    play.loc[play.play_defense_is_home_team, "play_defense_score"] = \
        play.loc[play.play_defense_is_home_team, "play_home_score"]

    play.loc[play.play_offense_is_home_team, "play_offense_timeouts"] = \
        play.loc[play.play_offense_is_home_team, "play_home_timeouts"]
    play.loc[play.play_defense_is_home_team, "play_offense_timeouts"] = \
        play.loc[play.play_defense_is_home_team, "play_away_timeouts"]
    play.loc[play.play_offense_is_home_team, "play_defense_timeouts"] = \
        play.loc[play.play_offense_is_home_team, "play_away_timeouts"]
    play.loc[play.play_defense_is_home_team, "play_defense_timeouts"] = \
        play.loc[play.play_defense_is_home_team, "play_home_timeouts"]

    play["play_snap_y"] = play.play_snap_y.sub(np.abs(y_bnd))

    play[["play_snap_left", "play_snap_center", "play_snap_right"]] = False
    play.loc[play.play_snap_y < -np.abs(y_hash), "play_snap_left"] = True
    play.loc[
        (play.play_snap_y >= -np.abs(y_hash)) &
        (play.play_snap_y <= +np.abs(y_hash)),
    "play_snap_center"] = True
    play.loc[play.play_snap_y > +np.abs(y_hash), "play_snap_right"] = True

    for c in num_pos_colnames:
        play.loc[play[c] < 0, c] = 0

    play = play.drop(columns=[
        "week", "play_clock", "play_penalty_types",
        "play_fumbled", "play_turnover_type",
        "play_challenged", "play_down_negated",
        "play_pass_made", "play_lateral_success",
        "play_pass_intercepted", "play_pass_interception_dropped",
        "play_pass_batted", "play_pass_thrownaway", "play_pass_dropped",
        "play_home_score", "play_away_score",
        "play_home_timeouts", "play_away_timeouts",
    ])

    play = play.drop_duplicates(subset=["play_uuid"])
    play = play.drop_duplicates(
        subset=["play_game_uuid", "play_quarter", "play_time"])

    play = play.reset_index(drop=True)
    play = play.infer_objects(copy=False)
    play = play.fillna(play_col_fillnas).astype(play_col_types)
    play = play.copy()

    season_tracking_files = list(sorted([
        f for f in os.listdir(tracking_path)
        if tm in f[20:] and
        (int(f[12:16]) == (szn) and (int(f[16:18]) >= 7) or
        (int(f[12:16]) == (szn + 1) and (int(f[16:18]) < 7)))
    ]))[:mct]

    track_dfs = []
    for tracking_index in range(len(season_tracking_files)):
        tracking_file = season_tracking_files[tracking_index]
        local_file = os.path.join(tracking_path, tracking_file)
        with gzip.open(local_file, "r") as f:
            sb_tracking = json.loads(f.read().decode("utf-8"))
        tracking = pd.json_normalize(
            sb_tracking,
            ["plays", "tracks", "steps"],
            meta=[
                "game_id",
                "frequency",
                ["plays", "play_uuid"],
                ["plays", "play_offense_team_id"],
                ["plays", "offense_left_to_right"],
                ["plays", "tracks", "team_id"],
                ["plays", "tracks", "nfl_team_id"],
                ["plays", "tracks", "player", "player_id"],
                ["plays", "tracks", "player", "name"],
                ["plays", "tracks", "player", "position_code"],
            ],
        )
        tracking.columns = [x.split(".")[-1] for x in tracking.columns]
        tracking = tracking.copy()
        if "calibration_fault" in tracking.columns:
            tracking = tracking.drop(columns=["calibration_fault"])
        if tracking_index < 0 + train_index:
            tracking["play_playoffs"] = False
        elif tracking_index >= len(season_tracking_files) - test_index:
            tracking["play_playoffs"] = True
        tracking["play_year"] = int(tracking_file[12:16])
        tracking["play_month"] = int(tracking_file[16:18])
        tracking["play_day"] = int(tracking_file[18:20])
        min_timestamp = tracking.timestamp.min()
        tracking["timestamp"] = tracking["timestamp"] - min_timestamp
        freq = \
            int(list(pd.factorize(tracking.frequency.astype(np.int16))[1])[0])
        tracking = tracking.iloc[::(freq // fps), :]
        if tracking_index < 0 + train_index:
            track_dfs.append(tracking)
        elif tracking_index >= len(season_tracking_files) - test_index:
            track_dfs.append(tracking)

    track = pd.concat(track_dfs)
    track = track.reset_index(drop=True)
    track = track.copy()

    invalid_plays_1 = track.loc[
        (track.play_uuid.isna()) | (track.timestamp.isna()) |
        (track.time_since_snap.isna()) | (track.player_id.isna()),
    "play_uuid"].unique()
    track = track.loc[~track.play_uuid.isin(invalid_plays_1),]

    track[["player_offense", "player_defense"]] = False
    track.loc[track.team_id == track.play_offense_team_id,
        "player_offense"] = True
    track.loc[track.team_id != track.play_offense_team_id,
        "player_defense"] = True

    track.loc[
        ~track.offense_left_to_right.astype("boolean").fillna(False), "x"
    ] = track.loc[
        ~track.offense_left_to_right.astype("boolean").fillna(False), "x"
    ].rsub(x_field_max)
    track["y"] = track.y.sub(np.abs(y_bnd))
    track.loc[
        ~track.offense_left_to_right.astype("boolean").fillna(False), "y"
    ] = track.loc[
        ~track.offense_left_to_right.astype("boolean").fillna(False), "y"
    ].mul(-1)

    invalid_plays_2_defense = track.loc[
        (track.player_defense & (
            (track.position_code == "OL") |
            (track.position_code == "TE") | (track.position_code == "WR") |
            (track.position_code == "RB") | (track.position_code == "QB") |
            (track.position_code == "P") | (track.position_code == "LS")
        )),
    "play_uuid"].unique()
    invalid_plays_2_offense = track.loc[
        (track.player_offense & (
            (track.position_code == "DB") | (track.position_code == "LB") |
            (track.position_code == "DL")
        )),
    "play_uuid"].unique()
    track = track.loc[~track.play_uuid.isin(invalid_plays_2_defense),]
    track = track.loc[~track.play_uuid.isin(invalid_plays_2_offense),]

    track["player_post_snap"] = False
    track.loc[track.time_since_snap >= 0.0, "player_post_snap"] = True

    track[[
        "player_position_ol", "player_position_te", "player_position_wr",
        "player_position_rb", "player_position_qb",
        "player_position_pt", "player_position_ls",
        "player_position_db", "player_position_lb", "player_position_dl",
    ]] = False
    track["player_position_na"] = True
    track.loc[track.position_code == "OL", "player_position_ol"] = True
    track.loc[track.position_code == "TE", "player_position_te"] = True
    track.loc[track.position_code == "WR", "player_position_wr"] = True
    track.loc[track.position_code == "RB", "player_position_rb"] = True
    track.loc[track.position_code == "QB", "player_position_qb"] = True
    track.loc[track.position_code == "P", "player_position_pt"] = True
    track.loc[track.position_code == "LS", "player_position_ls"] = True
    track.loc[track.position_code == "DB", "player_position_db"] = True
    track.loc[track.position_code == "LB", "player_position_lb"] = True
    track.loc[track.position_code == "DL", "player_position_dl"] = True
    track["player_position_not_na"] = track[[
        "player_position_ol", "player_position_te", "player_position_wr",
        "player_position_rb", "player_position_qb",
        "player_position_pt", "player_position_ls",
        "player_position_db", "player_position_lb", "player_position_dl",
    ]].any(axis=1)
    track["player_position_na"] = ~track["player_position_not_na"]

    track = track.drop(columns=track_remove_colnames, errors="ignore")
    track = track.rename(columns=track_rename_cols)

    track["now_time"] = track["now_time"].astype(np.uint64)
    track["min_time"] = \
        track.groupby("play_uuid", sort=False)["now_time"].transform("min")
    track["max_time"] = \
        track.groupby("play_uuid", sort=False)["now_time"].transform("max")
    track["frame_time"] = (track.now_time - track.min_time) / 1e6
    track["frame_time"] = track["frame_time"].div(100)
    track = track.drop(columns=["min_time", "now_time", "max_time"])

    track = play[["play_uuid", "play_snap_x", "play_snap_y"]].merge(
        track, on=["play_uuid"], how="inner", suffixes=("", "_track"))
    track = track.copy()

    track = track.drop_duplicates(
        subset=["play_uuid", "frame_time", "player_uuid"])

    track["player_x"] = \
        (track.player_x - track.play_snap_x) / x_field_max
    track["player_y"] = (track.player_y - track.play_snap_y) / np.abs(y_bnd)

    track[["player_dx", "player_dy"]] = col_fillnas["float"]
    track["player_dx"] = track.groupby(
        ["play_uuid", "player_uuid"], sort=False
    )["player_x"].diff(periods=+1)
    track["player_dy"] = track.groupby(
        ["play_uuid", "player_uuid"], sort=False
    )["player_y"].diff(periods=+1)

    invalid_plays_3 = track.loc[
        ((track.player_dx > +max_dx) | (track.player_dx < -max_dx)) |
        ((track.player_dy > +max_dy) | (track.player_dy < -max_dy)),
    "play_uuid"].unique()
    track = track.loc[~track.play_uuid.isin(invalid_plays_3),]

    track[[
        "player_vx", "player_vy", "player_ax", "player_ay",
    ]] = col_fillnas["float"]
    track["player_vx"] = track.groupby(
        ["play_uuid", "player_uuid"], sort=False
    )["player_x"].diff(periods=+1) * fps
    track["player_vy"] = track.groupby(
        ["play_uuid", "player_uuid"], sort=False
    )["player_y"].diff(periods=+1) * fps
    track["player_ax"] = track.groupby(
        ["play_uuid", "player_uuid"], sort=False
    )["player_vx"].diff(periods=+1) * fps
    track["player_ay"] = track.groupby(
        ["play_uuid", "player_uuid"], sort=False
    )["player_vy"].diff(periods=+1) * fps

    track["player_vx"] = track.player_vx.div(fps)
    track["player_vy"] = track.player_vy.div(fps)
    track["player_ax"] = track.player_ax.div((fps ** 2))
    track["player_ay"] = track.player_ay.div((fps ** 2))

    track["abs_player_y"] = track["player_y"].abs()
    track["abs_player_y"] = \
        track["abs_player_y"].astype(col_types["float"])
    track = track.sort_values(by=(
        frame_colnames[:-1] + ["player_offense", "player_x", "abs_player_y"]
    ))
    track = track.drop(columns=["abs_player_y"])

    track["player_rank"] = \
        track.groupby(frame_colnames[:-1], sort=False).cumcount() + 1
    track["player_rank"] = track["player_rank"].astype(np.uint8)
    track_pivot = track.pivot(
        index=frame_colnames,
        columns="player_rank",
        values=list(player_cols.keys())
    )
    track_pivot.columns = \
        [f"{c[0]}-{c[1]:02d}" for c in track_pivot.columns]
    track_pivot = track_pivot.reset_index(drop=False)

    track_play = track[
        frame_colnames + list(date_cols.keys()) + ["play_playoffs"]
    ]
    track_play = track_play.drop_duplicates(subset=frame_colnames[:-1])

    track_merged = \
        track_play.merge(track_pivot, on=frame_colnames, how="inner")
    track_merged = track_merged.copy()

    track_new = track_merged.reindex(
        columns=(
            frame_colnames + \
            list(date_cols.keys()) + list(pivot_cols.keys()) + \
            ["play_playoffs"]
        ),
        fill_value=np.nan
    )

    track_new = track_new.drop(
        index=track_new.loc[track_new.play_uuid.isna(),].index)
    track_new = track_new.drop(
        index=track_new.loc[track_new.frame_time.isna(),].index)

    track_new = track_new.drop(
        index=track_new.loc[track_new.frame_time < 0.000 - 1e-6,].index)
    track_new = track_new.drop(
        index=track_new.loc[track_new.frame_time > 1.000 + 1e-6,].index)

    track_new = track_new.drop(
        index=track_new.loc[track_new.frame_time_snap.isna(),].index)
    track_new = track_new.sort_values(by=frame_colnames[:-1])
    track_new = track_new.reset_index(drop=True)

    track_new["defense_ct"] = \
        track_new[[
            f"player_defense-{n:02d}" for n in range(1, all_players + 1)
        ]].astype("boolean").fillna(False).astype(int).sum(axis=1)
    track_new["offense_ct"] = \
        track_new[[
            f"player_offense-{n:02d}" for n in range(1, all_players + 1)
        ]].astype("boolean").fillna(False).astype(int).sum(axis=1)
    track_new["defense_ct"] = track_new["defense_ct"].astype(int)
    track_new["offense_ct"] = track_new["offense_ct"].astype(int)
    track_new["all_ct"] = track_new.defense_ct + track_new.offense_ct
    track_new["no_ct"] = track_new.all_ct.rsub(all_players)

    track_new["offense_fill"] = track_new[
        ["all_ct", "defense_ct", "offense_ct", "no_ct"]
    ].apply(lambda r: [
        n + 1 for n in range(
            r["all_ct"],
            r["all_ct"] + team_players - r["offense_ct"]
        )], axis=1
    )
    track_new["defense_fill"] = track_new[
        ["all_ct", "defense_ct", "offense_ct", "no_ct"]
    ].apply(lambda r: [
        n + 1 for n in range(
            r["all_ct"] + team_players - r["offense_ct"],
            all_players
        )], axis=1
    )

    track_new[
        [f"player_offense-{n:02d}" for n in range(1, all_players + 1)]
    ] = pd.DataFrame(track_new.apply(lambda r: [
            True if n in r["offense_fill"]
            else False if n in r["defense_fill"]
            else r[f"player_offense-{n:02d}"]
            for n in range(1, all_players + 1)
    ], axis=1).tolist(), index=track_new.index)
    track_new[
        [f"player_defense-{n:02d}" for n in range(1, all_players + 1)]
    ] = pd.DataFrame(track_new.apply(lambda r: [
            True if n in r["defense_fill"]
            else False if n in r["offense_fill"]
            else r[f"player_defense-{n:02d}"]
            for n in range(1, all_players + 1)
    ], axis=1).tolist(), index=track_new.index)

    track_new[
        [f"player_position_na-{n:02d}" for n in range(1, all_players + 1)]
    ] = pd.DataFrame(track_new.apply(lambda r: [
            True if (n in r["defense_fill"] or n in r["offense_fill"])
            else r[f"player_position_na-{n:02d}"]
            for n in range(1, all_players + 1)
    ], axis=1).tolist(), index=track_new.index)

    track_new[
        [f"player_post_snap-{n:02d}" for n in range(1, all_players + 1)]
    ] = pd.DataFrame(track_new.apply(lambda r: [
        True if r["frame_time_snap"] >= 0 else False
        for n in range(1, all_players + 1)
    ], axis=1).tolist(), index=track_new.index)
    track_new = track_new.copy()
    for n in range(1, all_players + 1):
        track_new[f"player_pre_snap-{n:02d}"] = \
            ~track_new[f"player_post_snap-{n:02d}"]
        track_new = track_new.copy()

    track_new = track_new.copy()
    track_new = track_new.drop(columns=[
        "all_ct",
        "defense_ct", "offense_ct", "defense_fill", "offense_fill",
        "no_ct",
    ])
    track_new = track_new.astype(track_col_types)
    track_new = track_new.copy()

    _xs = track_new[
        [f"player_x-{n:02d}" for n in range(1, all_players + 1)]
    ].values.astype(np.float32)
    _ys = track_new[
        [f"player_y-{n:02d}" for n in range(1, all_players + 1)]
    ].values.astype(np.float32)
    _off = track_new[
        [f"player_offense-{n:02d}" for n in range(1, all_players + 1)]
    ].values.astype(np.float32)
    _opp_mask = (
        (_off[:, :, np.newaxis] != _off[:, np.newaxis, :]) &
        (~(np.isnan(_xs) | np.isnan(_ys)))[:, np.newaxis, :]
    )
    _dist_all = np.sqrt(
        (_xs[:, :, np.newaxis] - _xs[:, np.newaxis, :]) ** 2 +
        (_ys[:, :, np.newaxis] - _ys[:, np.newaxis, :]) ** 2
    )
    _sep_all = np.where(_opp_mask, _dist_all, np.inf).min(axis=2)
    _sep_all = np.where(
        np.isinf(_sep_all), np.nan, _sep_all
    ).astype(np.float32)
    for _n in range(all_players):
        track_new[f"player_sep-{_n + 1:02d}"] = _sep_all[:, _n]
    for _n in range(1, all_players + 1):
        track_new[f"player_yabs-{_n:02d}"] = np.abs(
            _ys[:, _n - 1]
        ).astype(np.float32)

    df = play.merge(
        track_new, on=["play_uuid"], how="inner", suffixes=("", "_track"))
    df = df.copy()

    df = df.rename(columns=
        {"play_year": "year", "play_month": "month", "play_day": "day"})
    df["date"] = pd.to_datetime(df[["year", "month", "day"]])
    min_date = df["date"].min()
    days_to_subtract = (min_date.weekday() - 1) % 7
    start_of_week = min_date - pd.Timedelta(days=days_to_subtract)
    diff_days = (df["date"] - start_of_week).dt.days
    df["week_of_year"] = (diff_days // 7) + 1
    df["play_week"] = df.week_of_year.astype(col_types["int"])

    df["play_snap_x"] = \
        np.floor(df["play_snap_x"]).astype(col_types["int"])

    df.loc[df.play_type == play_catgs[-1], "play_yards_needed"] = \
        col_types["float"](2.00)
    df["play_yards_needed"] = \
        df["play_yards_needed"].fillna(col_types["float"](x_field_max))
    df["play_yards_needed"] = \
        np.ceil(df["play_yards_needed"]).astype(col_types["int"])

    df["play_defense_timeouts"] = df["play_defense_timeouts"].div(3)
    df["play_offense_timeouts"] = df["play_offense_timeouts"].div(3)

    df["play_defense_in_lead"] = False
    df.loc[df.play_defense_score > df.play_offense_score,
        "play_defense_in_lead"] = True
    df["play_defense_in_lead"] = \
        df["play_defense_in_lead"].astype(col_types["bool"])

    df["play_offense_in_lead"] = False
    df.loc[df.play_offense_score > df.play_defense_score,
        "play_offense_in_lead"] = True
    df["play_offense_in_lead"] = \
        df["play_offense_in_lead"].astype(col_types["bool"])

    df["play_defense_score"] = \
        df["play_defense_score"].astype(col_types["float"])

    df["play_offense_score"] = \
        df["play_offense_score"].astype(col_types["float"])

    df["play_score_difference"] = \
        df.play_offense_score - df.play_defense_score
    df["play_score_difference"] = \
        df["play_score_difference"].astype(col_types["float"])

    df.loc[df.play_score_difference.abs() <= 1e-2,
        "play_score_equal"] = True
    df.loc[df.play_score_difference.abs() > 1e-2,
        "play_score_equal"] = False

    _off_vals = np.column_stack([
        df[f"player_offense-{n:02d}"].values.astype(np.float32)
        for n in range(1, all_players + 1)
    ])
    _sep_vals = np.column_stack([
        df[f"player_sep-{n:02d}"].values.astype(np.float32)
        for n in range(1, all_players + 1)
    ])
    _x_vals = np.column_stack([
        df[f"player_x-{n:02d}"].values.astype(np.float32)
        for n in range(1, all_players + 1)
    ])
    _qb_vals = np.column_stack([
        df[f"player_position_qb-{n:02d}"].values.astype(np.float32)
        for n in range(1, all_players + 1)
    ])
    _is_off_qb = (_qb_vals == 1) & (_off_vals == 1)
    with np.errstate(all="ignore"):
        df["play_qb_pressure"] = np.nanmax(
            np.where(_is_off_qb, _sep_vals, np.nan), axis=1
        ).astype(col_types["float"])
    _wr_vals = np.column_stack([
        df[f"player_position_wr-{n:02d}"].values.astype(np.float32)
        for n in range(1, all_players + 1)
    ])
    _te_vals = np.column_stack([
        df[f"player_position_te-{n:02d}"].values.astype(np.float32)
        for n in range(1, all_players + 1)
    ])
    _rb_vals = np.column_stack([
        df[f"player_position_rb-{n:02d}"].values.astype(np.float32)
        for n in range(1, all_players + 1)
    ])
    _is_rec = (
        (_wr_vals == 1) | (_te_vals == 1) | (_rb_vals == 1)
    ) & (_off_vals == 1) & (_x_vals > 0)
    _rec_sep = np.where(_is_rec, _sep_vals, np.nan)
    _rec_x = np.where(_is_rec, _x_vals, np.nan)
    _rec_sep_f = np.where(np.isnan(_rec_sep), -np.inf, _rec_sep)
    _no_rec = np.all(~np.isfinite(_rec_sep_f), axis=1)
    _tgt_idx = np.argmax(_rec_sep_f, axis=1)
    _rows = np.arange(len(df))
    _tgt_sep = np.where(_no_rec, np.nan, _rec_sep[_rows, _tgt_idx])
    _tgt_depth = np.where(_no_rec, np.nan, _rec_x[_rows, _tgt_idx])
    df["play_tgt_sep"] = _tgt_sep.astype(col_types["float"])
    df["play_tgt_depth"] = _tgt_depth.astype(col_types["float"])

    df["play_time_since_start"] = \
        df["frame_time"].astype(col_types["float"])
    df["play_time_since_snap"] = \
        df["frame_time_snap"].astype(col_types["float"])

    df["play_time_before_snap"] = df.groupby(
        "play_uuid", group_keys=False, sort=False
    )["play_time_since_snap"].transform("min").abs()
    df["play_time_before_snap"] = \
        df["play_time_before_snap"].div(100.000).astype(col_types["float"])

    df["play_time_after_snap"] = df.groupby(
        "play_uuid", group_keys=False, sort=False
    )["play_time_since_snap"].transform("max").abs()
    df["play_time_after_snap"] = \
        df["play_time_after_snap"].div(100.000).astype(col_types["float"])

    df["play_season"] = \
        df.play_season.sub(1970).add(1).astype(col_types["int"])

    df = df.copy()
    df = df.drop(columns=[
        "date", "year", "month", "day", "week_of_year",
        "play_defense_score", "play_offense_score",
        "frame_time", "frame_time_snap",
    ])
    df = df.rename(columns={
        "play_season": "game_season",
        "play_week": "game_week",
        "play_playoffs": "game_playoffs",
    })
    df = df.copy()

    df["play_down"] = df.play_down.div(4).astype(col_types["float"])

    df["play_quarter"] = df.play_quarter.div(4).astype(col_types["float"])

    df["play_snap_x"] = \
        df.play_snap_x.div(x_field_max).astype(col_types["float"])
    df["play_snap_y"] = \
        df.play_snap_y.div(np.abs(y_bnd)).astype(col_types["float"])

    df["play_yards_needed"] = \
        df.play_yards_needed.div(x_field_max).astype(col_types["float"])
    df["play_yards_gained"] = \
        df.play_yards_gained.div(x_field_max).astype(col_types["float"])

    df["play_score_difference"] = \
        df.play_score_difference.div(100).astype(col_types["float"])

    df["play_time"] = df.play_time.div(1).astype(col_types["float"])
    df["play_time_since_start"] = \
        df.play_time_since_start.div(1).astype(col_types["float"])
    df["play_time_since_snap"] = \
        df.play_time_since_snap.div(1).astype(col_types["float"])

    df["game_week"] = df.game_week.div(25).astype(col_types["float"])

    df["game_season"] = df.game_season.div(1000).astype(col_types["float"])

    df = df.copy()
    df = df.reset_index(drop=True)
    df = df.sort_values(by=[
        "game_season", "game_week",
        "play_quarter", "play_time",
        "play_time_since_start",
    ])

    def_drive_dfs = []
    defense_teams = \
        list(dict.fromkeys(pd.factorize(df.play_defense_team)[1]))
    for dt in defense_teams:
        def_df = df.loc[df.play_defense_team == dt,].copy()
        def_df["play_defense_game"] = \
            def_df.groupby("play_game_uuid", sort=False).ngroup() + 1
        def_game_uuids = \
            list(dict.fromkeys(pd.factorize(def_df.play_game_uuid)[1]))
        for dgi in range(len(def_game_uuids)):
            dgu = def_game_uuids[dgi]
            def_gdf = \
                def_df.loc[def_df.play_game_uuid == dgu,].copy()
            def_gdf["play_defense_drive"] = def_gdf\
                .groupby("play_drive_uuid", sort=False).ngroup() + 1
            def_drive_uuids = list(
                dict.fromkeys(pd.factorize(def_gdf.play_drive_uuid)[1])
            )
            for ddi in range(len(def_drive_uuids)):
                ddu = def_drive_uuids[ddi]
                def_ddf = \
                    def_gdf.loc[def_gdf.play_drive_uuid == ddu,].copy()
                def_ddf["play_defense_play"] = \
                    def_ddf.groupby("play_uuid", sort=False).ngroup() + 1
                def_drive_dfs.append(def_ddf)
    defense_df = pd.concat(def_drive_dfs)[[
        "play_defense_game", "play_defense_drive", "play_defense_play"
    ]]

    off_drive_dfs = []
    offense_teams = \
        list(dict.fromkeys(pd.factorize(df.play_offense_team)[1]))
    for ot in offense_teams:
        off_df = df.loc[df.play_offense_team == ot,].copy()
        off_df["play_offense_game"] = \
            off_df.groupby("play_game_uuid", sort=False).ngroup() + 1
        off_game_uuids = \
            list(dict.fromkeys(pd.factorize(off_df.play_game_uuid)[1]))
        for ogi in range(len(off_game_uuids)):
            ogu = off_game_uuids[ogi]
            off_gdf = \
                off_df.loc[off_df.play_game_uuid == ogu,].copy()
            off_gdf["play_offense_drive"] = off_gdf\
                .groupby("play_drive_uuid", sort=False).ngroup() + 1
            off_drive_uuids = list(
                dict.fromkeys(pd.factorize(off_gdf.play_drive_uuid)[1])
            )
            for odi in range(len(off_drive_uuids)):
                odu = off_drive_uuids[odi]
                off_ddf = \
                    off_gdf.loc[off_gdf.play_drive_uuid == odu,].copy()
                off_ddf["play_offense_play"] = \
                    off_ddf.groupby("play_uuid", sort=False).ngroup() + 1
                off_drive_dfs.append(off_ddf)
    offense_df = pd.concat(off_drive_dfs)[[
        "play_offense_game", "play_offense_drive", "play_offense_play"
    ]]

    team_df = pd.concat([defense_df, offense_df], axis=1)
    df = pd.concat([df, team_df], axis=1)
    df = df.copy()

    df = df.drop(columns=["play_defense_team", "play_offense_team"])
    df[[
        c for c in df.columns if "uuid" not in c and c != "play_type"
    ]] = df[[
        c for c in df.columns if "uuid" not in c and c != "play_type"
    ]].astype(col_types["float"])
    df = df.drop(columns=player_uuid_colnames)
    df = df.copy()

    df["play_padded"] = col_types["float"](False)
    df["play_real"] = col_types["float"](True)
    df = df.copy()

    df["play_defense_game"] = \
        df.play_defense_game.div(num_games).astype(col_types["float"])
    df["play_defense_drive"] = \
        df.play_defense_drive.div(num_drives).astype(col_types["float"])
    df["play_defense_play"] = \
        df.play_defense_play.div(game_time).astype(col_types["float"])
    df["play_offense_game"] = \
        df.play_offense_game.div(num_games).astype(col_types["float"])
    df["play_offense_drive"] = \
        df.play_offense_drive.div(num_drives).astype(col_types["float"])
    df["play_offense_play"] = \
        df.play_offense_play.div(game_time).astype(col_types["float"])

    df = df.reset_index(drop=True)
    df = df.set_index("play_uuid", drop=False)

    df = df.rename(columns={"play_uuid": "play_uuid0"})
    df = df.drop(columns=[
        "play_drive_uuid", "play_game_uuid",
    ])
    df = df.sort_values(by=[
        "game_season", "game_week",
        "play_quarter", "play_time",
        "play_time_since_start",
    ])
    df = df.drop(columns=["play_time_since_start"])

    df = df.copy()

    return df

def pad_plays(
    df: pd.DataFrame, pad_ct: int
) -> pd.DataFrame:

    def pad_fn(sdf: pd.DataFrame) -> pd.DataFrame:
        sdf = sdf.copy()
        dfid = str(sdf.index.min())
        pad_size = pad_ct - sdf.shape[0]
        fstrow = sdf.iloc[0]

        gp = dtyp(fstrow.game_playoffs)
        typ = fstrow.play_type
        yg = dtyp(fstrow.play_yards_gained)

        zerrow_df = make_empty(dfid, fstrow, 1, False)
        zerrow_df[play_state_colnames] = col_types["float"](0.0)
        zerrow_df = zerrow_df.copy()
        zerrow_df[player_state_flattened_colnames] = col_types["float"](0.0)
        zerrow_df = zerrow_df.copy()
        zerrow_df[action_flattened_colnames] = col_types["float"](0.0)
        zerrow_df = zerrow_df.copy()
        zerrow_df[metadata_colnames] = col_types["float"](0.0)
        zerrow_df = zerrow_df.copy()

        zerrow_df = zerrow_df.copy()
        zerrow_df[[
            f"player_offense-{n:02d}" for n in range (1, team_players + 1)
        ]] = col_types["float"](1.0)
        zerrow_df = zerrow_df.copy()
        zerrow_df[[
            f"player_offense-{n:02d}"
            for n in range (team_players + 1, all_players + 1)
        ]] = col_types["float"](0.0)
        zerrow_df = zerrow_df.copy()
        zerrow_df[[
            f"player_defense-{n:02d}"
            for n in range (team_players + 1, all_players + 1)
        ]] = col_types["float"](1.0)
        zerrow_df = zerrow_df.copy()
        zerrow_df[[
            f"player_defense-{n:02d}" for n in range (1, team_players + 1)
        ]] = col_types["float"](0.0)
        zerrow_df = zerrow_df.copy()
        zerrow_df["play_true_length"] = col_types["float"]((sdf.shape[0] + 2))
        zerrow_df = zerrow_df.copy()

        sdf["play_true_length"] = col_types["float"]((sdf.shape[0] + 2))
        sdf = sdf.copy()

        empty_df = make_empty(dfid, fstrow, pad_size, True)
        empty_df = empty_df.copy()
        empty_df[[
            f"player_offense-{n:02d}" for n in range (1, team_players + 1)
        ]] = col_types["float"](1.0)
        empty_df = empty_df.copy()
        empty_df[[
            f"player_offense-{n:02d}"
            for n in range (team_players + 1, all_players + 1)
        ]] = col_types["float"](0.0)
        empty_df = empty_df.copy()
        empty_df[[
            f"player_defense-{n:02d}"
            for n in range (team_players + 1, all_players + 1)
        ]] = col_types["float"](1.0)
        empty_df = empty_df.copy()
        empty_df[[
            f"player_defense-{n:02d}" for n in range (1, team_players + 1)
        ]] = col_types["float"](0.0)
        empty_df = empty_df.copy()
        empty_df[[
            f"player_position_na-{n:02d}" for n in range(1, all_players + 1)
        ]] = col_types["float"](1.0)
        empty_df = empty_df.copy()
        empty_df[[
            f"player_post_snap-{n:02d}" for n in range(1, all_players + 1)
        ]] = col_types["float"](1.0)
        empty_df = empty_df.copy()
        empty_df[[
            f"player_pre_snap-{n:02d}" for n in range(1, all_players + 1)
        ]] = col_types["float"](0.0)
        empty_df = empty_df.copy()
        empty_df["play_true_length"] = \
            col_types["float"]((empty_df.shape[0] - 2))
        empty_df = empty_df.copy()

        lstrow_df = make_empty(dfid, fstrow, 1, False)
        lstrow_df[play_state_colnames] = col_types["float"](lstv)
        lstrow_df = lstrow_df.copy()
        lstrow_df[player_state_flattened_colnames] = col_types["float"](lstv)
        lstrow_df = lstrow_df.copy()
        lstrow_df[action_flattened_colnames] = col_types["float"](lstv)
        lstrow_df = lstrow_df.copy()
        lstrow_df[metadata_colnames] = col_types["float"](0.0)
        lstrow_df = lstrow_df.copy()
        lstrow_df["play_true_length"] = col_types["float"]((sdf.shape[0] + 2))
        lstrow_df = lstrow_df.copy()

        padded_df = pd.concat(
            [zerrow_df, sdf, empty_df, lstrow_df], axis=0, ignore_index=True
        )
        padded_df = padded_df.copy()

        padded_df["play_real"] = col_types["float"](True)
        padded_df["play_is_first"] = (
            [col_types["float"](1.0)] +
            ([col_types["float"](0.0)] * (padded_df.shape[0] - 1))
        )
        padded_df["play_is_last"] = (
            ([col_types["float"](0.0)] * (padded_df.shape[0] - 1)) +
            [col_types["float"](1.0)]
        )
        padded_df["game_playoffs"] = col_types["float"](gp)
        padded_df = padded_df.copy()

        padded_df["play_yards_gained"] = col_types["float"](0.0)
        padded_df.loc[
            (padded_df.play_is_last == col_types["float"](1.0)) & \
            (padded_df.play_yards_gained.notna()),
        "play_yards_gained"] = col_types["float"](yg)
        padded_df.loc[
            (padded_df.play_is_last == col_types["float"](1.0)) & \
            (padded_df.play_yards_gained.isna()),
        "play_yards_gained"] = col_types["float"](0.0)
        padded_df = padded_df.copy()

        padded_df.loc[(padded_df.play_is_first == col_types["float"](1.0)),
            "play_true_length"] = col_types["float"](fstv)
        padded_df.loc[(padded_df.play_is_last == col_types["float"](1.0)),
            "play_true_length"] = col_types["float"](lstv)
        padded_df = padded_df.copy()

        padded_df["play_uuid1"] = dfid
        padded_df["play_idx"] = (
            padded_df.groupby("play_uuid1", sort=False)
            .transform("cumcount") + 1
        )
        padded_df["play_type"] = typ
        padded_df = padded_df.copy()

        return padded_df

    p_df = df.groupby(
        "play_uuid0", group_keys=False, sort=False
    ).apply(pad_fn, include_groups=False)
    p_df = p_df.drop(columns=["play_time_since_snap"])
    p_df = p_df.copy()

    p_cols = [
        c for c in p_df.columns
        if c != "play_uuid0" and c != "play_uuid1" and \
            c != "play_idx" and c != "play_type"
    ]
    p_df[p_cols] = p_df[p_cols].astype(dtyp)
    p_df = p_df.copy()

    p_df["play_type"] = p_df["play_type"].astype("object")
    p_df["play_idx"] = \
        p_df["play_idx"].astype(col_types["float"]).astype(int)
    p_df["play_uuid1"] = p_df["play_uuid1"].astype("object")
    p_df = p_df.copy()

    p_df = p_df.reset_index(drop=True)
    p_df = p_df.set_index(["play_uuid1", "play_idx"], drop=True)
    p_df = p_df.copy()

    remove_ids = list(p_df.loc[
        (
            (p_df.index.get_level_values(1) == 2) |
            (p_df.index.get_level_values(1) == 3)
        ) &
        (
            (p_df[
                player_location_flattened_colnames
            ].fillna(0.0).abs().le(+1e-3).all(axis=1))
        )
    ].index.get_level_values(0))
    p_df = p_df.loc[~p_df.index.get_level_values(0).isin(remove_ids)]
    p_df = p_df.copy()

    return p_df

def set_reward(df: pd.DataFrame, sn: bool=False) -> pd.DataFrame:
    df = df.copy()

    if not sn:  sgn = -1
    else:  sgn = +1
    rwd = lambda yg: (1 / (1 + (np.exp(sgn * reward_scale * yg))))
    df["play_yards_gained"] = df["play_yards_gained"].apply(rwd)
    df["play_yards_gained"] = df["play_yards_gained"].astype(dtyp)

    df = df.copy()

    return df

def set_zero_play(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df.loc[df.play_padded == col_types["float"](1.0), play_state_colnames] = \
        col_types["float"](0.0)
    df.loc[df.play_padded == col_types["float"](1.0), "play_true_length"] = \
        col_types["float"](0.0)

    df = df.copy()

    return df

def set_zero_player(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df[player_state_flattened_colnames] = \
        df[player_state_flattened_colnames].fillna(0.0)
    df[action_flattened_colnames] = df[action_flattened_colnames].fillna(0.0)

    df.loc[df.play_padded == col_types["float"](1.0),
        [f"player_position_na-{n:02d}" for n in range(1, all_players + 1)]
    ] = col_types["float"](0.0)
    df.loc[df.play_padded == col_types["float"](1.0),
        [f"player_post_snap-{n:02d}" for n in range(1, all_players + 1)]
    ] = col_types["float"](0.0)

    df = df.copy()

    return df

def set_category(df: pd.DataFrame, catg_idxs: list[int]) -> pd.DataFrame:
    df = df.copy()

    catgs = [
        play_catgs[ci] for ci in catg_idxs
        if ci in list(range(len(play_catgs)))
    ]
    df = df.drop(index=df.loc[~df.play_type.isin(catgs),].index)

    df = df.copy()

    return df

def set_window(df: pd.DataFrame, window_len: int, downsample_rate: int=1):
    df = df.copy()

    num_t = max(df.index.get_level_values(1).tolist())
    if window_len > 1 and window_len < num_t - 1:
        df = df[
            (df.index.get_level_values(1) < window_len + 1) | \
            (df.index.get_level_values(1) == num_t)
        ].copy()

        lvl1a_w = df.index.get_level_values(1)
        lvl1b_w = lvl1a_w.where(lvl1a_w != num_t, window_len + 1)
        df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values(0), lvl1b_w], names=df.index.names)

        df.loc[
            (df.play_is_first != col_types["float"](1.0)) &
            (df.play_is_last != col_types["float"](1.0)), "play_true_length"
        ] = df.loc[
            (df.play_is_first != col_types["float"](1.0)) &
            (df.play_is_last != col_types["float"](1.0)), "play_true_length"
        ].clip(upper=(window_len + 1))

    df = df.copy()

    num_t = max(df.index.get_level_values(1).tolist())
    downsample = fps // downsample_rate
    if downsample >= 2 and downsample <= fps // 2:
        ds_keys = [df.index.names[0], "ds"]

        df["ds"] = ((df.index.get_level_values(1) - 1) // downsample) + 1
        df.loc[df.index.get_level_values(1) == num_t, "ds"] = df.ds.max() + 1
        dlt = df.groupby(ds_keys, sort=False)[action_flattened_colnames].sum()

        fst = df.groupby(
            "ds", group_keys=False, sort=False
        ).apply(
            lambda g: g.index.get_level_values(1).min(), include_groups=False
        ).tolist()
        df = df.loc[df.index.get_level_values(1).isin(fst),].copy()

        jnd = df.join(dlt, on=ds_keys, rsuffix="__rr")
        df[dlt.columns] = jnd[[(c + "__rr") for c in dlt.columns]].to_numpy()

        df.index = pd.MultiIndex.from_arrays(
            [df.index.get_level_values(0), df.ds], names=df.index.names)

        df.loc[
            (df.play_is_first != col_types["float"](1.0)) &
            (df.play_is_last != col_types["float"](1.0)), "play_true_length"
        ] = df.loc[
            (df.play_is_first != col_types["float"](1.0)) &
            (df.play_is_last != col_types["float"](1.0)), "play_true_length"
        ].sub(1)

        df.loc[
            (df.play_is_first != col_types["float"](1.0)) &
            (df.play_is_last != col_types["float"](1.0)), "play_true_length"
        ] = df.loc[
            (df.play_is_first != col_types["float"](1.0)) &
            (df.play_is_last != col_types["float"](1.0)), "play_true_length"
        ].floordiv(downsample)

        df.loc[
            (df.play_is_first != col_types["float"](1.0)) &
            (df.play_is_last != col_types["float"](1.0)), "play_true_length"
        ] = df.loc[
            (df.play_is_first != col_types["float"](1.0)) &
            (df.play_is_last != col_types["float"](1.0)), "play_true_length"
        ].add(1)

        df = df.drop(columns=["ds"])

    df = df.copy()

    return df

def postprocess_df(
    df: pd.DataFrame,
    sn: bool=False,
    ci: list[int]=list(range(len(play_catgs))),
    mw: int=0, dr: int=1
) -> pd.DataFrame:
    df = df.copy()

    df = set_reward(df, sn=sn)

    df = set_zero_play(df)

    df = set_zero_player(df)

    df = set_category(df, catg_idxs=ci)

    df = set_window(df, window_len=mw, downsample_rate=dr)

    df = df.drop(columns=["index"])

    df = df.copy()

    return df

def split_df(
    df_w: pd.DataFrame, ratio: float=1.0, random: bool=False
) -> pd.DataFrame:
    rtio = 1.0 if ratio <= 0.0 or ratio >= 1.0 else ratio
    traj_ids_w = sorted(list(set(list(pd.factorize(
        df_w.index.get_level_values(0)
    )[1]))))

    if random:  rng = np.random.default_rng(None)
    else:  rng = np.random.default_rng(seed)

    if rtio == 1.0:
        traj_ids = [t for t in traj_ids_w]
        traj_ids_n = []
    else:
        traj_ids = list(rng.choice(
            traj_ids_w, size=int(len(traj_ids_w) * rtio), replace=False
        ))
        traj_ids_n = [t for t in traj_ids_w if t not in traj_ids]

    return (traj_ids, traj_ids_n)
