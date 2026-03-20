
import torch
from torch.utils.data import Dataset

from data.play_gfn import PlayContainer


class OfflineTrajectoryDataset(Dataset):
    def __init__(self, containers: list[PlayContainer]) -> None:
        self.containers = containers

    def __len__(self) -> int:
        return len(self.containers)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor]:
        ctr = self.containers[idx]
        states = ctr.states.tensor.clone().detach()
        actions = ctr.actions[~ctr.is_terminal.bool()].tensor.clone().detach()
        reward = ctr.rewards[ctr.is_terminal.bool()].sum()

        return (states, actions, reward)

class PairedOfflineTrajectoryDataset(Dataset):
    def __init__(
        self,
        defense_containers: list[PlayContainer],
        offense_containers: list[PlayContainer],
    ) -> None:
        self.defense_containers = defense_containers
        self.offense_containers = offense_containers

    def __len__(self) -> int:
        return \
            (len(self.defense_containers) + len(self.offense_containers)) // 2

    def __getitem__(self, idx: int) -> dict[str, tuple[torch.Tensor]]:
        d_ctr = self.defense_containers[idx]
        o_ctr = self.offense_containers[idx]

        d_states = d_ctr.states.tensor.clone().detach()
        o_states = o_ctr.states.tensor.clone().detach()
        d_actions = \
            d_ctr.actions[~d_ctr.is_terminal.bool()].tensor.clone().detach()
        o_actions = \
            o_ctr.actions[~o_ctr.is_terminal.bool()].tensor.clone().detach()
        d_reward = d_ctr.rewards[d_ctr.is_terminal.bool()].sum()
        o_reward = o_ctr.rewards[o_ctr.is_terminal.bool()].sum()

        return {
            "defense": (d_states, d_actions, d_reward),
            "offense": (o_states, o_actions, o_reward),
        }
