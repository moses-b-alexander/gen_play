
import torch
from typing import Tuple

from ai.constants import action_dim
from gflownet.estimators import Estimator
from gflownet.trajectories import Trajectories


def get_trajectory_pfs_and_pbs(
    pf: Estimator, pb: Estimator | None,
    trajectories: Trajectories, fill_value: float=0.0
) -> Tuple[torch.Tensor, torch.Tensor]:
    log_pf_trajectories = \
        get_trajectory_pfs(pf, trajectories, fill_value=fill_value)
    log_pb_trajectories = \
        get_trajectory_pbs(pb, trajectories, fill_value=fill_value)

    return (log_pf_trajectories, log_pb_trajectories)

def get_trajectory_pfs(
    pf: Estimator, trajectories: Trajectories, fill_value: float=0.0
) -> torch.Tensor:
    device = trajectories.states.device

    state_mask = ~trajectories.states.is_sink_state
    action_mask = ~trajectories.actions.is_dummy

    valid_states = trajectories.states[state_mask].to_device(device)
    valid_actions = trajectories.actions[action_mask].to_device(device)

    log_pf_trajectories = torch.full_like(
        trajectories.actions.tensor[..., 0],
        fill_value=fill_value, dtype=torch.get_default_dtype()
    )

    if len(valid_states) == 0:  return log_pf_trajectories

    valid_log_pf_actions = pf.to_probability_distribution(
        valid_states, None, False
    ).log_prob(valid_actions.tensor)

    log_pf_trajectories[action_mask] = valid_log_pf_actions

    assert log_pf_trajectories.shape[:2] == (
        trajectories.max_length, trajectories.n_trajectories,
    )

    return log_pf_trajectories

def get_trajectory_pbs(
    pb: Estimator | None, trajectories: Trajectories, fill_value: float=0.0
) -> torch.Tensor:
    device = trajectories.states.device

    state_mask = ~trajectories.states.is_initial_state
    state_mask[0, :] = False
    action_mask = (
        ~trajectories.actions.is_dummy & ~trajectories.actions.is_exit)

    valid_states = trajectories.states[state_mask].to_device(device)
    valid_actions = trajectories.actions[action_mask].to_device(device)

    log_pb_trajectories = torch.full_like(
        trajectories.actions.tensor[..., 0],
        fill_value=fill_value, dtype=torch.get_default_dtype()
    )

    if len(valid_states) == 0:  return log_pb_trajectories

    if pb is not None:
        valid_log_pb_actions = pb.to_probability_distribution(
            valid_states, None, False
        ).log_prob(valid_actions.tensor)
    else:
        valid_log_pb_actions = torch.zeros_like(valid_actions.tensor[..., 0])

    log_pb_trajectories[action_mask] = valid_log_pb_actions

    assert log_pb_trajectories.shape[:2] == (
        trajectories.max_length, trajectories.n_trajectories
    )

    return log_pb_trajectories
