
from __future__ import annotations

import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
# os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from copy import deepcopy
from functools import partial
import numpy as np
import random
import sys
from time import localtime, perf_counter, sleep, strftime
import torch
from uuid import uuid4

from ai.aggregate import *
from ai.constants import *
from ai.decoder import *
from ai.diff_eq import *
from ai.encoder import *
from ai.hyperparameters import *
from ai.pf import *
from ai.train import *
from ai.utils import *
from common.constants import *
from common.dirs import *
from data.bridge import *
from data.constants import *
from data.names import *
from data.processing import *
from data.utils import *
from visualization.constants import *
from visualization.plots import *


print("\n\nMain.\n\n")
print(strftime("%Y-%m-%d | %H:%M:%S", localtime()))
t0 = perf_counter()

saved = True
from_start = False

torch.cuda.empty_cache()
torch.serialization.add_safe_globals([dtyp])

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

tsts = seasons[-season_count::] if not from_start else seasons[:season_count:]

unified_dfs, offense_dfs, defense_dfs, _ = [], [], [], []
unified_dfs_f, offense_dfs_f, defense_dfs_f, _ = [], [], [], []
if not saved:
    for ts in tsts:
        unified_df = get_data(ts[0], ts[1], match_count)
        unified_df = unified_df.copy()
        unified_dfs.append(unified_df)

    max_size = np.max([
        (df.groupby("play_uuid0", sort=False).count().max().iloc[0] + 2)
        for df in unified_dfs
    ])
    for i in range(len(unified_dfs)):
        ts = tsts[i]

        udf_f = pad_plays(unified_dfs[i], max_size)
        udf_f = udf_f.copy()
        udf_f.to_parquet(
            os.path.join(processed_dir, f"u_df_{ts[0]}_{ts[1]}.parquet"),
            engine="pyarrow"
        )
        unified_dfs_f.append(udf_f)

else:
    for ts in tsts:
        udf_f = pd.read_parquet(
            os.path.join(processed_dir, f"u_df_{ts[0]}_{ts[1]}.parquet"),
            engine="pyarrow"
        )
        udf_f = udf_f.copy()
        unified_dfs_f.append(udf_f)

df_u_f = pd.concat(unified_dfs_f, axis=0, ignore_index=False)
df_u_f = df_u_f.copy()

df_u_f = postprocess_df(
    df_u_f, sn=reward_sign, ci=[0, ], mw=max_window, dr=downsampling
)
df_u_f = df_u_f.copy()

num_timesteps = max(df_u_f.index.get_level_values(1).tolist())

train_ids, test_ids = split_df(df_w=df_u_f, ratio=train_ratio)
df_u_f_train = df_u_f.loc[df_u_f.index.get_level_values(0).isin(train_ids),]
df_u_f_test = df_u_f.loc[df_u_f.index.get_level_values(0).isin(test_ids),]

df_shape = df_u_f_train.shape
get_num_plays = lambda df: (len(
    sorted(list(set(list(pd.factorize(df.index.get_level_values(0))[1]))))
))
num_plays = get_num_plays(df_u_f_train)
df_mbs = np.round(
    (df_u_f_train.memory_usage(deep=True).sum()) / (1024 * 1024), 3
)

print("\n\n")
print(f"{df_shape} (Rows, Columns)")
print(f"Total Number of Trajectories: {num_plays}")
print(f"Number of Trajectories per Batch: {batch_size}")
print(f"Number of Timesteps per Trajectory: {num_timesteps}")
print(f"Number of Players per Timestep: {shape_players}")
print(f"Number of Input|Output Features per Player: {state_dim}|{action_dim}")
print(f"~{df_mbs // 2} MB Used for Storage")
# print(asdf)
print("\n\n")

prototype_hps = dict(
    dim_global=global_dim, dim_agent=agent_dim, dim_output=action_dim,
    dim_prj_play=32, dim_lin_play=64, dropout_play=0.01,
    dim_prj_player=32, dim_lin_player=64, dropout_player=0.01,
    dim_prj_fused=32, dim_lin_fused=64, dropout_fused=0.05,
    beta_sp=1e1, epsilon_sp=1e-8,
    device=learning_device,
)

input_hps = dict(
    player_count=(shape_players//2),
    trajectory_length=(num_timesteps*1),
)
shared_hps = dict(
    dim_h=final_dim,
    pow_iters=1,
)
encoder_hps = get_encoder_hyperparameters(
    lags=([0, 15], [15, 30]),
    dim_play=128, dim_player=128,
    num_heads=4, dropout=0.10,
    expansion=2,
) | input_hps
diffeq_hps = dict(
    drift_size=2, diffusion_size=2,
    dim_middle=32,
    steps=2,
    rtol=1e-5, atol=1e-5,
    scale=1e-1,
    dt=1e-2,
) | {}
decoder_hps = get_decoder_hyperparameters(
    min_s=1e-5, max_s=1e-2,
) | input_hps
output_hps = dict(
    noise_floor=1e-6, noise_ceiling=1e-1, noise_gap=10, noise_exp=1.0,
    prior_means=((0.0000, 0.0000), (0.0000, 0.0000)),
    prior_stdvs=((0.0010, 0.0010), (0.0010, 0.0010)),
    device=learning_device,
)

pf_hps = dict(
    **input_hps,
    backwards=False,
    dim_start=state_dim, dim_hidden=shared_hps["dim_h"], dim_end=action_dim,
    pow_iters=shared_hps["pow_iters"],
    encoder=partial(Encoder, **encoder_hps),
    diff_eq=partial(DiffEq, **diffeq_hps),
    decoder=partial(Decoder, **decoder_hps),
    **output_hps,
)
pb_hps = dict(
    **input_hps,
    dim_start=state_dim, dim_hidden=shared_hps["dim_h"], dim_end=action_dim,
    pow_iters=shared_hps["pow_iters"],
    encoder=partial(Encoder, **encoder_hps),
    diff_eq=partial(DiffEq, **diffeq_hps),
    decoder=partial(Decoder, **decoder_hps),
    backwards=True,
    **output_hps,
)

pf_flat_hps = {
    **input_hps,
    "dim_start": state_dim,
    "dim_hidden": shared_hps["dim_h"],
    "dim_end": action_dim,
    "pow_iters": shared_hps["pow_iters"],
    **encoder_hps,
    **diffeq_hps,
    **decoder_hps,
    "backwards": False,
    **output_hps,
}
config_hps = {
    "dtype": str(dtyp),
    "seasons": season_count, "matches": match_count,
    "reward_threshold": reward_threshold, "reward_scale": reward_log_scale,
    "reward_sign": reward_sign,
    "split": train_ratio,
    "lr": learning_rate, "wd": weight_decay_rate,
    "trajectories": num_plays,
    "batch": batch_size,
    "fps": downsample_rate, "window": max_window, "timesteps": num_timesteps,
    "players": shape_players,
    "input_dim": state_dim, "hidden_dim": final_dim, "output_dim": action_dim,
    "dx_threshold": max_dx, "dy_threshold": max_dy,
    "screen": screen_mode, "color": "#999999",
}

retsu0 = train_bagged_model(
    bag_ct=1, ratio=0.999,
    df_m=df_u_f_train,
    pf_cls=PF, pf_args=pf_hps,
    # pb_cls=PF, pb_args=pb_hps,
    pb_cls=None, pb_args={},
    random=False,
    write_model=True,
    hp_dict=pf_flat_hps,
    cfg_dict=config_hps,
    runner_device=learning_device
)

run_id = ""
retsu0 = [
    (r[0], ((PF(**pf_hps)).load_state_dict(r[-1])).eval())
    for r in (load_models(run_id, learning_device))
]

num_eval_traj = batch_size * 1
eval_states, eval_ids, eval_df = produce_evaluation_states(
    num=num_eval_traj, df_e=df_u_f_test, random=False
)
eval_states_r = PlayStates(permute_batch_first(eval_states.tensor))

print("===================================================================\n")
avgs = aggregate_samples(retsu0, eval_states)

orig = [tensorize_xy(eval_df, eval_id) for eval_id in eval_ids]
gend = [
    accumulate_xy(avg.to(eval_state_tensor), eval_state_tensor)
    for avg, eval_state_tensor
    in zip(avgs.unbind(dim=1), eval_states.tensor.unbind(dim=1))
]

ips = list(range(num_eval_traj))
for ip in ips[::3]:
    a00, s00 = plot_play_2(None, orig[ip], gend[ip], num_timesteps, 1)
    plt.show()
    plt.close("all")

t1 = perf_counter()
print(f"\n\nRan for {round((t1 - t0), 3)} Seconds.\n\n")
print("\n\nDone.\n\n")
