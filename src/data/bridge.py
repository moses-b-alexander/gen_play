
import numpy as np
import pandas as pd
import torch

from common.constants import dtyp, team_players
from data.constants import shape_players, x_field_max, y_mid, y_bnd
from data.names import (
    action_colnames,
    metadata_colnames,
    play_state_colnames,
    player_location_flattened_colnames,
    player_state_colnames,
    reward_colname
)


def tensorize_df(
    df: pd.DataFrame, play_index: int,
    tensorization_device: torch.device
) -> tuple[torch.Tensor]:
    play_tensor = torch.tensor(
        df[play_state_colnames].values, dtype=torch.float32)
    play_tensor = play_tensor.unsqueeze(1).expand(-1, shape_players, -1)
    player_tensors = [
        torch.tensor(df[sl].values, dtype=torch.float32).unsqueeze(1)
        for sl in player_state_colnames[:shape_players]
    ]
    player_tensor = torch.cat(player_tensors, dim=1)
    state_tensor = torch.cat([play_tensor, player_tensor], dim=-1)
    state_tensor = state_tensor.to(tensorization_device)

    action_tensors = [
        torch.tensor(df[al].values, dtype=torch.float32).unsqueeze(1)
        for al in action_colnames[:shape_players]
    ]
    action_tensor = torch.cat(action_tensors, dim=1)
    action_tensor = action_tensor.to(tensorization_device)

    reward_tensor = torch.tensor(
        df[[reward_colname]].values, dtype=torch.float32)
    reward_tensor = reward_tensor.to(tensorization_device)

    metadata_tensor = torch.tensor(
        df[metadata_colnames].values, dtype=torch.float32)
    metadata_tensor = metadata_tensor.to(tensorization_device)

    index_tensor = torch.full((df.shape[0],), play_index, dtype=torch.int64)
    index_tensor = index_tensor.to(tensorization_device)

    return (
        state_tensor, action_tensor, reward_tensor,
        metadata_tensor, index_tensor
    )

def tensorize_xy(
    df: pd.DataFrame, play_index: str
) -> tuple[tuple]:
    sdf = df.loc[
        (df.index.get_level_values(0) == play_index) &
        (df.play_is_last != 1.0)
    ].copy() if isinstance(play_index, str) else df.copy()
    te = torch.zeros((sdf.shape[0], team_players))

    xs = torch.tensor(
        sdf[player_location_flattened_colnames[0::2]].astype(dtyp).values)
    ys = torch.tensor(
        sdf[player_location_flattened_colnames[1::2]].astype(dtyp).values)

    sx, sy = dtyp(sdf.play_snap_x.iloc[1]), dtyp(sdf.play_snap_y.iloc[1])
    xs, ys = (
        np.round((((xs + sx) * x_field_max)), 3),
        np.round((((ys + sy) * np.abs(y_bnd)) + y_mid), 3),
    )

    xsf = xs.clone() if xs.size(1) == shape_players else te.clone()
    ysf = ys.clone() if ys.size(1) == shape_players else te.clone()

    return (
        (xsf[:, :team_players], xsf[:, team_players:]),
        (ysf[:, :team_players], ysf[:, team_players:]),
        (
            np.round(sx * x_field_max, 0),
            np.round((sy * np.abs(y_bnd) + y_mid), 3)
        )
    )
