
from math import log
import torch
import torch.nn as nn

from ai.constants import action_dim
from ai.utils import collect_distributions
from common.constants import dtyp
from data.play_gfn import PlayActions, PlayStates


def aggregate_logz(tups: list[tuple[dtyp, nn.Module]]) -> dtyp:
    z = dtyp((torch.tensor([t[0] for t in tups])).mean(dim=0))

    return z

def aggregate_distributions(
    tups: list[tuple[dtyp, nn.Module]],
    states: PlayStates, min_stdev: float=1e-5
) -> torch.distributions.independent.Independent:
    dists = collect_distributions(tups=tups, states=states)

    if min_stdev < 1e-5 or min_stdev > 1e0:  min_std = 1e-5
    else:  min_std = min_stdev

    means = torch.stack([p.mean for p in dists], dim=0)
    stds = torch.stack([p.stddev for p in dists], dim=0)

    mean = means.mean(dim=0)
    var = (stds ** 2).mean(dim=0) + ((means - mean) ** 2).mean(dim=0)
    std = var.clamp_min(min_std).sqrt()

    d = torch.distributions.Independent(
        torch.distributions.Normal(mean, std), 1)

    return d

def aggregate_samples(
    tups: list[tuple[dtyp, nn.Module]], states: PlayStates,
) -> torch.Tensor:
    dists = collect_distributions(tups=tups, states=states)

    samples = torch.stack([p.sample() for p in dists], dim=0)
    r = samples.mean(dim=0).reshape((*states.tensor.shape[:-1], action_dim))

    return r

def aggregate_logprobs(
    tups: list[tuple[dtyp, nn.Module]],
    states: PlayStates, actions: PlayActions,
) -> torch.Tensor:
    dists = collect_distributions(tups=tups, states=states)

    lp = torch.stack([p.log_prob(actions) for p in dists], dim=0)
    q = torch.logsumexp(lp, dim=0) - log(len(tups))

    return q
