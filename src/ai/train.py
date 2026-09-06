
from copy import deepcopy
import numpy as np
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Type
from uuid import uuid4
import wandb

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

    opto = torch.optim.AdamW(
        params=(pf_params + pb_params),
        lr=learning_rate, weight_decay=weight_decay_rate,
        betas=(0.900, 0.999), eps=1e-8
    ) # defaults

    return (gf_net, p_f_model, p_b_model, opto)

def training_step(
    batch: tuple[torch.Tensor],
    gfnet: GFlowNet, opt: torch.optim.Optimizer,
    step_num: int, step_device: torch.device
) -> tuple[GFlowNet, torch.optim.Optimizer, dtyp]:
    trajs = \
        build_trajectories_from_batch(batch=batch, tensor_device=step_device)
    gfnet.pf.set_step(step_num)
    if gfnet.pb is not None:  gfnet.pb.set_step(step_num)

    loss = gfnet.loss(trajs, False)

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
    env_val: PlayEnv | None=None,
    patience_frac: float=0.10, eval_every: int=-1, min_delta: float=0.01,
    use_wandb: bool=False
) -> tuple[dtyp, Estimator, dtyp]:
    ctrs = subset_containers(env=env, idxs=idxs)
    dataset = OfflineTrajectoryDataset(ctrs)

    gfn, pf, pb, opti = initialize_model(
        env=env, fwd_model=p_f, bwd_model=p_b,
        learning_rate=lr, weight_decay_rate=wdr,
        model_device=training_device
    )
    if optm is not None:  opti = optm

    if random:
        loader = DataLoader(
            dataset,
            batch_size=bs, shuffle=True,
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

    if min_delta < 1e-6 or min_delta > (1 - 1e-6):  min_delta = 0.01
    if patience_frac <= 0.0 or patience_frac >= 1.0:  patience_frac = 0.10

    steps_per_epoch = len(dataset) // bs
    total_steps_planned = ne * steps_per_epoch

    val_loader = None
    if env_val is not None and len(env_val.containers) > 0:
        val_dataset = OfflineTrajectoryDataset(subset_containers(env=env_val))
        val_loader = DataLoader(
            val_dataset,
            batch_size=bs, shuffle=False,
            pin_memory=False, drop_last=False,
            worker_init_fn=None, generator=None
        )

    if val_loader is not None:
        if eval_every <= 1:
            eval_every = max(2, steps_per_epoch)
            eval_every = max(2, min(eval_every, total_steps_planned // 2))

        total_checks = max(1, total_steps_planned // eval_every)
        patience = max(1, round(total_checks * patience_frac))

        print(
            s_str.replace("=", "~"),
            f"Evaluate every {eval_every} steps for {total_checks} checks. "
            f"Wait ~{(patience * eval_every)} steps for improvement of "
            f"at least {min_delta * 100:.2f}% relative loss. ",
            s_str.replace("=", "~")
        )
        if use_wandb:
            wandb.config.update({
                "eval_every_effective": eval_every,
                "patience_effective": patience,
                "min_delta_effective": min_delta
            })

    best_val_loss = float("inf")
    best_state: tuple[dict, dict | None] | None = None
    checks_no_improve = 0
    stopped_early = False

    cur_g = 0
    for epc in range(1, ne + 1):
        cur_e = 0
        hit_step_cap = False
        for batch in loader:
            cur_g += 1
            cur_e += 1

            if cur_g > ((2 ** 15) - 2):
                hit_step_cap = True
                break

            gfn, opti, lossv = training_step(
                batch=batch, gfnet=gfn, opt=opti, step_num=cur_g,
                step_device=training_device
            )

            if write_loss > 0 and cur_e % write_loss == 0:
                print(
                    s_str,
                    f"Epoch {epc:03d} Step {cur_e:04d} Loss: {lossv:.3f}",
                    s_str
                )
                if use_wandb:
                    wandb.log({"loss": lossv, "epoch": epc, "step": cur_e})

            if val_loader is not None and cur_g % eval_every == 0:
                gfn.eval()
                val_losses = []
                with torch.no_grad():
                    for vbatch in val_loader:
                        vtrajs = build_trajectories_from_batch(
                            batch=vbatch, tensor_device=training_device)
                        val_losses.append(dtyp(gfn.loss(vtrajs, True).item()))
                gfn.train()

                val_loss = dtyp(np.mean(val_losses)) \
                    if len(val_losses) > 0 else float("inf")
                print(
                    s_str,
                    f"Epoch {epc:03d} Step {cur_e:04d} "
                    f"Val Loss: {val_loss:.3f}",
                    s_str
                )
                if use_wandb:
                    wandb.log(
                        {"val_loss": val_loss, "epoch": epc, "step": cur_e})

                if (best_val_loss * (1.0 - min_delta)) > val_loss:
                    best_val_loss = val_loss
                    checks_no_improve = 0
                    best_state = (
                        deepcopy(gfn.pf.state_dict()),
                        deepcopy(gfn.pb.state_dict())
                            if gfn.pb is not None else None
                    )
                else:
                    checks_no_improve += 1
                    if checks_no_improve > patience:
                        print(
                            s_str,
                            f"Early stopping at step {cur_g:05d} "
                            f"(no validation improvement for "
                            f"{patience} checks)",
                            s_str
                        )
                        stopped_early = True
                        break

        if hit_step_cap or stopped_early:  break

    if best_state is not None:
        gfn.pf.load_state_dict(best_state[0])
        if gfn.pb is not None:  gfn.pb.load_state_dict(best_state[1])

    if return_aux:  return (env.log_z, gfn.pf, gfn.pb, opti, best_val_loss)
    else:  return (env.log_z, gfn.pf, best_val_loss)

def train_bagged_model(
    bag_count: int, validation_ratio: float,
    df_m: pd.DataFrame,
    pf_cls: Type[Estimator], pb_cls: Type[Estimator],
    pf_args: dict, pb_args: dict,
    bs: int, lr: float, wdr: float, ne: int,
    random: bool=False,
    write_model: bool=False,
    cfg_dict: dict={},
    runner_device: torch.device=learning_device,
    patience_frac: float=0.10, eval_every: int=-1, min_delta: float=0.01,
    use_wandb: bool=False,
    wandb_project: str="gen_play"
) -> list[tuple[dtyp, Estimator, dtyp]]:
    rets = []
    uid = uuid4().hex

    ratio = 1 - validation_ratio
    nt = len(sorted(list(set(list(pd.factorize(
        df_m.index.get_level_values(0)
    )[1])))))
    er = 1 / (10 ** (int(np.log10(nt)) + 1))
    if ratio >= (1 - er):  ratio = 1.000 # for convenience
    if ratio <= (0 + er):  ratio = (1 / nt) + (er * 1e-1) # just in case
    ct = nt * ratio

    total_steps = max(bs, int(ct * ne // bs))
    pf_args |= {"total_steps": total_steps}
    if pb_cls is not None:
        pb_args |= {"total_steps": total_steps}

    for i in range(bag_count):
        if use_wandb:
            wandb.init(
                project=wandb_project,
                group=uid, name=f"{uid}_{i+1}",
                config={
                    **deepcopy(cfg_dict),
                    **deepcopy(pf_args),
                    "validation_ratio_effective": (1.0 - ratio),
                    "bag_idx": (i + 1)
                },
                reinit=True
            )

        ids_train, ids_val = split_df(df_w=df_m, ratio=ratio, random=random)
        df_mm = df_m.loc[df_m.index.get_level_values(0).isin(ids_train),]
        env_train = PlayEnv(
            dfs=[df_mm], preprocessor=PlayPreprocessor(output_dim=state_dim))

        env_val = None
        if len(ids_val) > 0:
            df_vv = df_m.loc[df_m.index.get_level_values(0).isin(ids_val),]
            env_val = PlayEnv(
                dfs=[df_vv],
                preprocessor=PlayPreprocessor(output_dim=state_dim)
            )

        models = (
            (pf_cls(**deepcopy(pf_args)), pb_cls(**deepcopy(pb_args)))
            if pb_cls is not None else (pf_cls(**deepcopy(pf_args)), None)
        )

        mt = train_model(
            return_aux=False,
            env=env_train,
            p_f=models[0], p_b=models[1],
            optm=None,
            bs=bs,
            lr=lr, wdr=wdr,
            ne=ne,
            idxs=None,
            write_loss=2,
            random=random,
            training_device=runner_device,
            env_val=env_val,
            patience_frac=patience_frac, eval_every=eval_every,
            min_delta=min_delta,
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
