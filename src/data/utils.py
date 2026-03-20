
import numpy as np
import pandas as pd
import torch

from ai.constants import batch_size, state_dim
from ai.utils import permute_batch_first
from data.constants import reward_threshold
from data.play_gfn import (
    PlayActions, PlayContainer, PlayEnv, PlayPreprocessor, PlayStates
)
from data.processing import split_df
from gflownet.trajectories import Trajectories


def subset_containers(
    env: PlayEnv, idxs: int | list[int] | None=None
) -> list[PlayContainer]:
    if idxs is None:  idx_l = list(range(len(env.containers)))
    elif isinstance(idxs, int):
        if idxs >= 0 and idxs < len(env.containers):
            idx_l = [idxs]
    elif isinstance(idxs, list):
        idx_l = [i for i in idxs if i >= 0 and i < len(env.containers)]
    else:  idx_l = list(range(len(env.containers)))

    ctrs = [env.containers[ci] for ci in idx_l]

    return ctrs

def produce_evaluation_states(
    num: int | float,
    df_e: pd.DataFrame,
    random: bool=False
) -> PlayStates:
    eval_ids, _ = split_df(df_w=df_e, ratio=1.0)
    num_plays_eval = len(eval_ids)

    if isinstance(num, int):  num_n = num
    elif isinstance(num, float):  num_n = int(num * num_plays_eval)
    else:  num_n = 1
    if num_n <= 0 or num_n >= num_plays_eval:  num_n = 1

    if random:
        final_eval_ids = rng.choice(eval_ids, size=num_n, replace=False)
    else:
        final_eval_ids = eval_ids[:num_n]

    df_ee = pd.concat([
        df_e.loc[df_e.index.get_level_values(0) == fei,]
        for fei in final_eval_ids
    ])
    env_e = PlayEnv(
        dfs=[df_ee], preprocessor=PlayPreprocessor(output_dim=state_dim))
    eval_ctrs = [c for c in env_e.containers]

    eval_states = PlayStates(
        torch.stack([ec.states.tensor for ec in eval_ctrs], dim=0))
    eval_states = \
        PlayStates(permute_batch_first(eval_states.tensor[:, :-1, ...]))

    return (eval_states, final_eval_ids, (df_e.copy()))

def build_trajectories_from_batch(
    batch: tuple[torch.Tensor], tensor_device: torch.device
) -> Trajectories:
    states, actions, rewards = batch

    states = (states.permute(1, 0, 2, 3).contiguous()).clone().detach()
    states_traj = PlayStates(states).to_device(tensor_device)

    actions = (actions.permute(1, 0, 2, 3).contiguous()).clone().detach()
    actions_traj = PlayActions(actions).to_device(tensor_device)

    rewards_clamped = \
        rewards.clamp(min=-reward_threshold, max=+reward_threshold)
    log_rewards = torch.log(rewards_clamped).to(tensor_device)
    log_rewards_traj = log_rewards

    trajs = Trajectories(
        states=states_traj, actions=actions_traj, log_rewards=log_rewards_traj
    )

    return trajs
