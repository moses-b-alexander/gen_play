
from copy import deepcopy
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Type
from uuid import uuid4
import wandb # pyright: ignore[reportMissingImports]

from ai.constants import action_dim, state_dim
from ai.utils import seed_worker
from common.constants import dtyp, s_char, s_str, seed
from common.devices import learning_device
from common.dirs import output_dir
from data.constants import x_field_max
from data.datasets import OfflineTrajectoryDataset
from data.play_gfn import PlayEnv, PlayPreprocessor
from data.processing import split_df
from data.utils import build_trajectories_from_batch, subset_containers
from gflownet.base import GFlowNet, TBGFlowNet
from gflownet.estimators import Estimator


def initialize_model(
    env: PlayEnv,
    fwd_model: Estimator, bwd_model: Estimator,
    learning_rate: float, weight_decay_rate: float,
    model_device: torch.device
) -> tuple[GFlowNet, Estimator, Estimator, torch.optim.Optimizer]:

    p_f_model = fwd_model.to(model_device)
    p_b_model = bwd_model.to(model_device) if bwd_model is not None else None

    gf_net = TBGFlowNet(pf=p_f_model, pb=p_b_model, logZ=env.log_z)

    pf_params = list(p_f_model.parameters())
    pb_params = list(p_b_model.parameters()) if p_b_model is not None else []

    optm = torch.optim.AdamW(
        params=pf_params + pb_params,
        lr=learning_rate, weight_decay=weight_decay_rate,
        betas=(0.900, 0.999), eps=1e-8,
    )

    return (gf_net, p_f_model, p_b_model, optm)

def training_step(
    env: PlayEnv, batch: tuple[torch.Tensor],
    gfnet: GFlowNet, opt: torch.optim.Optimizer,
    step_num: int,
    step_device: torch.device
) -> tuple[GFlowNet, torch.optim.Optimizer, dtyp]:
    trajs = \
        build_trajectories_from_batch(batch=batch, tensor_device=step_device)
    gfnet.pf.set_step(step_num)
    if gfnet.pb is not None:  gfnet.pb.set_step(step_num)

    loss = gfnet.loss(trajs)

    opt.zero_grad(set_to_none=True)
    loss.backward(retain_graph=False)
    opt.step()

    loss_val = dtyp(loss.item())

    return (gfnet, opt, loss_val)

def train_model(
    return_aux: bool,
    env: PlayEnv,
    p_f: Estimator, p_b: Estimator | None,
    optm: torch.optim.Optimizer | None,
    bs: int,
    lr: float, wdr: float,
    ne: int,
    idxs: int | list[int] | None,
    write_loss: int,
    random: bool,
    training_device: torch.device,
    use_wandb: bool=False
) -> tuple[dtyp, nn.Module]:
    ctrs = subset_containers(env=env, idxs=idxs)

    dataset = OfflineTrajectoryDataset(ctrs)

    gfn, pf, pb, opt = initialize_model(
        env=env, fwd_model=p_f, bwd_model=p_b,
        learning_rate=lr, weight_decay_rate=wdr,
        model_device=training_device
    )
    if optm is not None:  opt = optm

    if random:
        loader = DataLoader(
            dataset,
            batch_size=bs, shuffle=False,
            pin_memory=False, drop_last=True,
            worker_init_fn=None, generator=None
        )
    else:
        gen = torch.Generator()
        gen.manual_seed(seed)
        loader = DataLoader(
            dataset,
            batch_size=bs, shuffle=True,
            pin_memory=False, drop_last=True,
            worker_init_fn=seed_worker, generator=gen
        )

    cur = 0
    for epoch in range(ne):
        for batch in loader:
            cur += 1
            if cur > ((2 ** 31) - 2):  break

            gfn, opt, lossv = training_step(
                env=env, batch=batch,
                gfnet=gfn, opt=opt,
                step_num=cur,
                step_device=training_device
            )

            if write_loss > 0 and cur % write_loss == 0:
                print(
                    s_str,
                    f"Epoch {(epoch+1):03d} Step {cur:04d} Loss: {lossv:.3f}",
                    s_str
                )
                if use_wandb:
                    wandb.log({"loss": lossv, "epoch": epoch+1, "step": cur})
        else:
            continue
        break

    if return_aux:  return (env.log_z, gfn.pf, gfn.pb, opt)
    else:  return (env.log_z, gfn.pf)

def train_bagged_model(
    bag_ct: int,
    df_m: pd.DataFrame,
    pf_cls: Type[Estimator], pb_cls: Type[Estimator],
    pf_args: dict, pb_args: dict,
    opt_args: dict,
    ratio: float=1.000,
    random: bool=False,
    write_model: bool=False,
    cfg_dict: dict={},
    runner_device: torch.device=learning_device,
    use_wandb: bool=False,
    wandb_project: str="gen_play"
) -> list[tuple[dtyp, Estimator]]:
    rets = []
    uid = uuid4().hex

    if ratio == 0.999:  ratio = 1.000 # for convenience
    if ratio == 0.000:  ratio = 0.001 # just in case

    ct = len(sorted(list(set(list(pd.factorize(
        df_m.index.get_level_values(0)
    )[1]))))) * ratio

    pf_args |= {"noise_gap": int(
        pf_args["noise_decay"] * ct * opt_args["ne"] // opt_args["bs"]
    )}
    pf_args.pop("noise_decay")
    if pb_cls is not None:
        pb_args |= {"noise_gap": int(
            pb_args["noise_decay"] * ct * opt_args["ne"] // opt_args["bs"]
        )}
        pb_args.pop("noise_decay")

    for i in range(bag_ct):
        if use_wandb:
            wandb.init(
                project=wandb_project,
                group=uid, name=f"{uid}_{i+1}",
                config={
                    **deepcopy(cfg_dict),
                    **deepcopy(pf_args),
                    **deepcopy(opt_args),
                    "bag_idx": (i + 1)
                },
                reinit=True
            )

        ids_train, _ = split_df(df_w=df_m, ratio=ratio, random=random)
        df_mm = df_m.loc[df_m.index.get_level_values(0).isin(ids_train),]
        env_train = PlayEnv(
            dfs=[df_mm], preprocessor=PlayPreprocessor(output_dim=state_dim))

        models = (
            (pf_cls(**deepcopy(pf_args)), pb_cls(**deepcopy(pb_args)))
            if pb_cls is not None else (pf_cls(**deepcopy(pf_args)), None)
        )

        mt = train_model(
            return_aux=False,
            env=env_train,
            p_f=models[0], p_b=models[1],
            optm=None,
            bs=opt_args["bs"],
            lr=opt_args["lr"], wdr=opt_args["wdr"],
            ne=opt_args["ne"],
            idxs=None,
            write_loss=2,
            random=random,
            training_device=runner_device,
            use_wandb=use_wandb
        )
        rets.append(mt)

        if use_wandb:
            wandb.log({
                "log_z": dtyp(mt[0]),
                "total_yards": dtyp(np.round((np.exp(mt[0])*x_field_max), 3))
            })
            wandb.finish()

        if write_model:
            wd = os.path.join(output_dir, uid)
            if not os.path.exists(wd):  os.makedirs(wd)
            torch.save(
                {
                    "log_z": dtyp(mt[0]),
                    "hyperparameters": deepcopy(pf_args),
                    "config": cfg_dict.copy(),
                    "pf": mt[1].state_dict(),
                },
                os.path.join(wd, f"model_{i+1}.pt")
            )
            print(
                (s_str[:-1] * 2).replace(s_char, "*"), "\n",
                f"Total Yards for {uid}_{i+1}# ",
                float(np.round((np.exp(mt[0])*x_field_max), 3)),
                (s_str[:-1] * 2).replace(s_char, "*"), "\n",
            )

    return (rets, uid)
