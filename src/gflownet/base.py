
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from typing import Any, cast

from ai.constants import global_dim
from common.constants import dtyp
from data.constants import max_play_frames
from gflownet.estimators import Estimator
from gflownet.prob_calculations import get_trajectory_pfs_and_pbs
from gflownet.trajectories import Trajectories


class GFlowNet(ABC, nn.Module):
    # [GFlowNet Foundations](https://arxiv.org/pdf/2111.09266)

    @abstractmethod
    def to_training_samples(self, trajectories: Trajectories) -> Trajectories:

        pass

    @abstractmethod
    def loss(self, training_objects: Any, inference: bool) -> torch.Tensor:

        pass

class TrajectoryBasedGFlowNet(GFlowNet):
    def __init__(
        self, pf: Estimator, pb: Estimator | None
    ) -> None:
        super().__init__()

        self.pf = pf
        self.pb = pb

    def get_scores(
        self,
        trajectories: Trajectories, inference: bool, fill_value: float
    ) -> torch.Tensor:
        log_pf_trajectories, log_pb_trajectories = get_trajectory_pfs_and_pbs(
            self.pf, self.pb, trajectories, inference, fill_value)

        feats = trajectories.states.tensor[:, :, :, :-2]
        feat_sums = feats.sum(dim=(-2, -1))
        pad_mask_inv = torch.isclose(
            feat_sums, torch.zeros_like(feat_sums), rtol=1e-6, atol=1e-6)
        pad_mask_inv_r = \
            pad_mask_inv.unsqueeze(-1).expand(-1, -1, feats.size(-2))
        pad_mask = ~pad_mask_inv_r

        pad_mask_pf = pad_mask[:-1, ...].float()
        pad_mask_pb = pad_mask[1:, ...].float()

        ret_log_pf_trajectories = \
            (log_pf_trajectories * pad_mask_pf).sum(dim=0).sum(dim=-1)
        ret_log_pb_trajectories = \
            (log_pb_trajectories * pad_mask_pb).sum(dim=0).sum(dim=-1)

        log_rewards = trajectories.log_rewards

        assert ret_log_pf_trajectories.shape == (trajectories.n_trajectories,)
        assert ret_log_pb_trajectories.shape == (trajectories.n_trajectories,)

        return (
            ret_log_pf_trajectories - ret_log_pb_trajectories - log_rewards)

    def to_training_samples(self, trajectories: Trajectories) -> Trajectories:

        return trajectories

class TBGFlowNet(TrajectoryBasedGFlowNet):
    def __init__(
        self,
        pf: Estimator, pb: Estimator | None,
        logZ: dtyp | nn.Parameter | None=None
    ) -> None:
        self.constant_pb = True if pb is None else False
        super().__init__(pf, pb)

        if logZ is None:  self.logZ = nn.Parameter(torch.tensor(0.0))
        else:  self.logZ = logZ

    def loss(
        self, trajectories: Trajectories, inference: bool
    ) -> torch.Tensor:
        # [Trajectory Balance loss](https://arxiv.org/abs/2201.13259)

        scores = self.get_scores(trajectories, inference, 0.00)
        log_Z = cast(torch.Tensor, self.logZ)

        stt = trajectories.states.tensor
        inv_t = 1 / stt.size(0)
        total_len = stt.size(1) * stt.size(0)
        norm_lens = stt[1, :, 0, global_dim-1] * inv_t

        norm_scores = (
            scores * (
                norm_lens / norm_lens.mean(dim=0)
            ).clamp(min=(1 / max_play_frames), max=(max_play_frames * 1))
        )
        z_scores = norm_scores + log_Z.squeeze()
        final_scores = nn.functional.huber_loss(
            input=z_scores, target=torch.zeros_like(z_scores),
            delta=(total_len + 1), reduction="none"
        )

        loss = (
            final_scores.mean(dim=0) * (
                (stt[1, :, 0, global_dim-1]).sum(dim=0) / total_len
            ).clamp(min=(4.00 * inv_t), max=1.00)
        ) + 1e-12

        return loss
