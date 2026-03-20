
import torch
import torch.nn as nn

from ai.attender import Attender
from ai.constants import (
    agent_dim, global_dim, max_deltas, player_tensor_indices
)
from ai.custom_linear import LLinear
from ai.index_encoder import IndexEncoder
from ai.mlp_encoder import MLPEncoder
from ai.normalizer import Normalizer
from ai.play_encoder import PlayEncoder
from ai.player_normalizer import PlayerNormalizer
from ai.timestep_encoder import TimestepEncoder
from data.constants import shape_players


class Encoder(nn.Module):
    def __init__(
        self,
        player_count: int,
        trajectory_length: int,
        lags: tuple[list[int]],
        num_mlps_play: int,
        dim_projection_play: int, dim_start_play: int, dim_end_play: int,
        dropout_play: float,
        dim_stp: int, tau_stp: float,
        dim_snap: int,
        dim_idx: int,
        dim_start_player_initial_proj: int, dim_end_player_initial_proj: int,
        num_mlps_player_pre_attn_mlp: int,
        dim_start_player_pre_attn_mlp: int, dim_end_player_pre_attn_mlp: int,
        dropout_player_pre_attn_mlp: float,
        num_heads_player_attn: int,
        dim_middle_player_attn: int,
        tau_player_attn: float, tau_stdev_player_attn: float,
        dropout_player_attn: float,
        num_mlps_player_post_attn_mlp: int,
        dim_start_player_post_attn_mlp: int,
        dim_end_player_post_attn_mlp: int,
        dropout_player_post_attn_mlp: float,
        num_mlps_player_kin_mlp: int,
        dim_start_player_kin_mlp: int, dim_end_player_kin_mlp: int,
        dim_start_player_gate: int,
        num_mlps_player_pre_gate_attn_mlp: int,
        dim_start_player_pre_gate_attn_mlp: int,
        dim_end_player_pre_gate_attn_mlp: int,
        dropout_player_pre_gate_attn_mlp: float,
        num_heads_player_gate_attn: int,
        dim_middle_player_gate_attn: int,
        tau_player_gate_attn: float, tau_stdev_player_gate_attn: float,
        dropout_player_gate_attn: float,
        num_mlps_player_post_gate_attn_mlp: int,
        dim_start_player_post_gate_attn_mlp: int,
        dim_end_player_post_gate_attn_mlp: int,
        dropout_player_post_gate_attn_mlp: float,
        rnn_type: str,
        dim_start_rnn_tm: int, dim_start_rnn_op: int,
        dim_end_rnn_tm: int, dim_end_rnn_op: int,
        num_layers_rnn_tm: int, num_layers_rnn_op: int,
        dropout_rnn_tm: float, dropout_rnn_op: float,
        tau_softmax: float,
        dim_embedding_softmax: int,
        num_mlps_player_pre_fusion_mlp: int,
        dim_start_player_pre_fusion_mlp: int,
        dim_end_player_pre_fusion_mlp: int,
        dropout_player_pre_fusion_mlp: float,
        dim_start_player_fusion: int, dim_end_player_fusion: int,
        num_mlps_player_post_fusion_mlp: int,
        dim_start_player_post_fusion_mlp: int,
        dim_end_player_post_fusion_mlp: int,
        dropout_player_post_fusion_mlp: float,
        dim_start_conditioning: int, dim_end_conditioning: int,
        dim_output: int,
        pow_iters: int,
        backwards: bool
    ) -> None:
        super().__init__()

        self.checkpoints = True

        self.backwards = backwards

        assert isinstance(player_count, int)
        assert player_count == (shape_players // 2)
        self.player_count = player_count

        assert isinstance(trajectory_length, int) and trajectory_length > 3
        self.trajectory_length = trajectory_length

        self.register_buffer("max_delta", torch.tensor(max_deltas))

        self.player_keys_d = {}
        lags_k = ["def_tm", "off_tm", "def_op", "off_op"]
        lags_0, lags_1 = {k: [] for k in lags_k}, {k: [] for k in lags_k}
        assert len(lags) == 1 or len(lags) == 2 or len(lags) == 4
        for i in range(len(lags)):
            assert isinstance(lags[i], list)
            for lag in lags[i]:
                if lag < (self.trajectory_length - 2) - 1 and lag >= 0:
                    if len(lags) == 1:
                        for ki in lags_k:  lags_0[ki].append(lag)
                    elif len(lags) == 2 and i == 0:
                        lags_0["def_tm"].append(lag)
                        lags_0["off_tm"].append(lag)
                    elif len(lags) == 2 and i == 1:
                        lags_0["def_op"].append(lag)
                        lags_0["off_op"].append(lag)
                    else:
                        lags_0[lags_k[i]].append(lag)
        for ki in lags_k:  lags_1[ki] = sorted(list(set(lags_0[ki])))
        for ki in lags_k:
            self.player_keys_d[ki] = [f"{ki}_{lag:04d}" for lag in lags_1[ki]]
        self.player_keys = sorted(
            sum([l for l in self.player_keys_d.values()], []),
            key=lambda f: int(f[len("xxf_xx_"):])
        )

        self.include_idx = True

        self.play_enc = PlayEncoder(
            num_mlps=num_mlps_play,
            dim_projection=dim_projection_play,
            dim_embedding_start=dim_start_play,
            dim_embedding_end=dim_end_play,
            dropout=dropout_play,
            pow_iters=pow_iters
        )

        self.timestep_enc = TimestepEncoder(
            trajectory_length=self.trajectory_length-2,
            dim_stp=dim_stp, tau_stp=tau_stp
        )

        self.snap_enc = LLinear(
            dim_start=1, dim_end=dim_snap,
            pow_iters=pow_iters
        )

        self.index_enc = IndexEncoder(
            dim_idx=dim_idx,
            pow_iters=pow_iters
        )

        self.player_feature_slices = sorted([
            k for k in player_tensor_indices.keys()
            if k != "team" and k != "snap"
        ])
        player_initial_proj_args = dict(
            dim_start=dim_start_player_initial_proj,
            dim_end=dim_end_player_initial_proj,
            pow_iters=pow_iters
        )
        self.player_initial_projs = nn.ModuleDict({
            k: nn.ModuleDict({
                s: PlayerNormalizer(**(
                    {"type_feature": s} | player_initial_proj_args
                ))
                for s in self.player_feature_slices
            })
            for k in self.player_keys
        })

        if dim_end_player_initial_proj != dim_start_player_pre_attn_mlp:
            self.player_pre_attn_mlp_projs = {
                k: {
                    s: LLinear(
                        dim_start=dim_end_player_initial_proj,
                        dim_end=dim_start_player_pre_attn_mlp,
                        pow_iters=pow_iters
                    )
                    for s in self.player_feature_slices
                }
                for k in self.player_keys
            }
        else:
            self.player_pre_attn_mlp_projs = nn.ModuleDict({
                k: nn.ModuleDict({
                    s: nn.Identity() for s in self.player_feature_slices
                })
                for k in self.player_keys
            })

        player_pre_attn_mlp_args = dict(
            residual=False, num_mlps=num_mlps_player_pre_attn_mlp,
            dim_start=dim_start_player_pre_attn_mlp,
            dim_end=dim_end_player_pre_attn_mlp,
            norms=[""]*num_mlps_player_pre_attn_mlp,
            dropouts=[0.0]*num_mlps_player_pre_attn_mlp,
            biases=[(False, False)]*num_mlps_player_pre_attn_mlp,
            pow_iters=pow_iters
        )
        self.player_pre_attn_mlp_encs = nn.ModuleDict({
            k: nn.ModuleDict({
                s: MLPEncoder(**player_pre_attn_mlp_args)
                for s in self.player_feature_slices
            })
            for k in self.player_keys
        })
        self.player_pre_attn_mlp_drops = nn.ModuleDict({
            k: nn.ModuleDict({
                s: nn.Dropout(dropout_player_pre_attn_mlp)
                for s in self.player_feature_slices
            })
            for k in self.player_keys
        })

        self.player_pre_attn_blocks = nn.ModuleDict({
            k: nn.ModuleDict({
                s: nn.Sequential(
                    self.player_pre_attn_mlp_projs[k][s],
                    self.player_pre_attn_mlp_encs[k][s],
                    self.player_pre_attn_mlp_drops[k][s]
                )
                for s in self.player_feature_slices
            })
            for k in self.player_keys
        })

        player_pre_attn_proj_args = dict(
            type_norm="rms",
            dim_start=dim_end_player_pre_attn_mlp,
            dim_middle=dim_middle_player_attn,
            dim_end=dim_middle_player_attn,
            biases=(False, False), norm_bias=False,
            pow_iters=pow_iters
        )
        self.player_pre_attn_projs = nn.ModuleDict({
            k: nn.ModuleDict({
                s: Normalizer(**player_pre_attn_proj_args)
                for s in self.player_feature_slices
            })
            for k in self.player_keys
        })

        player_attn_args = dict(
            dim_embedding=dim_middle_player_attn,
            num_heads=num_heads_player_attn,
            tau=tau_player_attn,
            tau_stdev=tau_stdev_player_attn,
            dropout=dropout_player_attn,
            pow_iters=pow_iters
        )
        self.player_attn_encs = nn.ModuleDict({
            k: nn.ModuleDict({
                s: Attender(**player_attn_args)
                for s in self.player_feature_slices
            })
            for k in self.player_keys
        })

        player_post_attn_proj_args = dict(
            type_norm="rms",
            dim_start=dim_middle_player_attn,
            dim_middle=dim_middle_player_attn,
            dim_end=dim_start_player_post_attn_mlp,
            biases=(False, False), norm_bias=False,
            pow_iters=pow_iters
        )
        self.player_post_attn_projs = nn.ModuleDict({
            k: nn.ModuleDict({
                s: Normalizer(**player_post_attn_proj_args)
                for s in self.player_feature_slices
            })
            for k in self.player_keys
        })

        player_post_attn_mlp_args = dict(
            residual=False, num_mlps=num_mlps_player_post_attn_mlp,
            dim_start=dim_start_player_post_attn_mlp,
            dim_end=dim_end_player_post_attn_mlp,
            norms=[""]*num_mlps_player_post_attn_mlp,
            dropouts=[0.0]*num_mlps_player_post_attn_mlp,
            biases=[(False, False)]*num_mlps_player_post_attn_mlp,
            pow_iters=pow_iters
        )
        self.player_post_attn_mlp_encs = nn.ModuleDict({
            k: nn.ModuleDict({
                s: MLPEncoder(**player_post_attn_mlp_args)
                for s in self.player_feature_slices
            })
            for k in self.player_keys
        })
        self.player_post_attn_mlp_drops = nn.ModuleDict({
            k: nn.ModuleDict({
                s: nn.Dropout(dropout_player_post_attn_mlp)
                for s in self.player_feature_slices
            })
            for k in self.player_keys
        })

        self.player_post_attn_blocks = nn.ModuleDict({
            k: nn.ModuleDict({
                s: nn.Sequential(
                    self.player_post_attn_mlp_encs[k][s],
                    self.player_post_attn_mlp_drops[k][s]
                )
                for s in self.player_feature_slices
            })
            for k in self.player_keys
        })

        if (2 * dim_end_player_post_attn_mlp) != dim_start_player_kin_mlp:
            self.player_kin_mlp_projs = nn.ModuleDict({
                k: LLinear(
                    dim_start=(2 * dim_end_player_post_attn_mlp),
                    dim_end=dim_start_player_kin_mlp,
                    pow_iters=pow_iters
                )
                for k in self.player_keys
            })
        else:
            self.player_kin_mlp_projs = nn.ModuleDict({
                k: nn.Identity() for k in self.player_keys
            })

        player_kin_mlp_args = dict(
            residual=False, num_mlps=num_mlps_player_kin_mlp,
            dim_start=dim_start_player_kin_mlp,
            dim_end=dim_end_player_kin_mlp,
            norms=[""]*num_mlps_player_kin_mlp,
            dropouts=[0.0]*num_mlps_player_kin_mlp,
            biases=[(False, False)]*num_mlps_player_kin_mlp,
            pow_iters=pow_iters
        )
        self.player_kin_mlp_encs = nn.ModuleDict({
            k: MLPEncoder(**player_kin_mlp_args)
            for k in self.player_keys
        })

        self.player_kin_mlp_blocks = nn.ModuleDict({
            k: nn.Sequential(
                self.player_kin_mlp_projs[k], self.player_kin_mlp_encs[k]
            )
            for k in self.player_keys
        })

        if dim_end_player_kin_mlp != dim_start_player_gate:
            self.player_kin_gate_projs = nn.ModuleDict({
                k: LLinear(
                    dim_start=dim_end_player_kin_mlp,
                    dim_end=dim_start_player_gate,
                    pow_iters=pow_iters
                )
                for k in self.player_keys
            })
        else:
            self.player_kin_gate_projs = nn.ModuleDict({
                k: nn.Identity() for k in self.player_keys
            })

        if dim_end_player_post_attn_mlp != dim_start_player_gate:
            self.player_role_gate_projs = nn.ModuleDict({
                k: LLinear(
                    dim_start=dim_end_player_post_attn_mlp,
                    dim_end=dim_start_player_gate,
                    pow_iters=pow_iters
                )
                for k in self.player_keys
            })
        else:
            self.player_role_gate_projs = nn.ModuleDict({
                k: nn.Identity() for k in self.player_keys
            })

        self.player_gate_scales = nn.ModuleDict({
            k: LLinear(
                dim_start=dim_start_player_gate,
                dim_end=dim_start_player_gate,
                pow_iters=pow_iters
            )
            for k in self.player_keys
        })

        if dim_start_player_gate != \
            dim_start_player_pre_gate_attn_mlp:
            self.player_pre_gate_attn_mlp_projs = nn.ModuleDict({
                k: LLinear(
                    dim_start=dim_start_player_gate,
                    dim_end=dim_start_player_pre_gate_attn_mlp,
                    pow_iters=pow_iters
                )
                for k in self.player_keys
            })
        else:
            self.player_pre_gate_attn_mlp_projs = nn.ModuleDict({
                k: nn.Identity() for k in self.player_keys
            })

        player_pre_gate_attn_mlp_args = dict(
            residual=False, num_mlps=num_mlps_player_pre_gate_attn_mlp,
            dim_start=dim_start_player_pre_gate_attn_mlp,
            dim_end=dim_end_player_pre_gate_attn_mlp,
            norms=[""]*num_mlps_player_pre_gate_attn_mlp,
            dropouts=[0.0]*num_mlps_player_pre_gate_attn_mlp,
            biases=[(False, False)]*num_mlps_player_pre_gate_attn_mlp,
            pow_iters=pow_iters
        )
        self.player_pre_gate_attn_mlp_encs = nn.ModuleDict({
            k: MLPEncoder(**player_pre_gate_attn_mlp_args)
            for k in self.player_keys
        })
        self.player_pre_gate_attn_mlp_drops = nn.ModuleDict({
            k: nn.Dropout(dropout_player_pre_gate_attn_mlp)
            for k in self.player_keys
        })

        player_pre_gate_attn_proj_args = dict(
            type_norm="rms",
            dim_start=dim_end_player_pre_gate_attn_mlp,
            dim_middle=dim_middle_player_gate_attn,
            dim_end=dim_middle_player_gate_attn,
            biases=(False, False), norm_bias=False,
            pow_iters=pow_iters
        )
        self.player_pre_gate_attn_projs = nn.ModuleDict({
            k: Normalizer(**player_pre_gate_attn_proj_args)
            for k in self.player_keys
        })

        self.player_pre_gate_attn_blocks = nn.ModuleDict({
            k: nn.Sequential(
                self.player_pre_gate_attn_mlp_projs[k],
                self.player_pre_gate_attn_mlp_encs[k],
                self.player_pre_gate_attn_mlp_drops[k]
            )
            for k in self.player_keys
        })

        player_gate_attn_args = dict(
            dim_embedding=dim_middle_player_gate_attn,
            num_heads=num_heads_player_gate_attn,
            tau=tau_player_gate_attn,
            tau_stdev=tau_stdev_player_gate_attn,
            dropout=dropout_player_gate_attn,
            pow_iters=pow_iters
        )
        self.player_gate_attn_encs = nn.ModuleDict({
            k: Attender(**player_gate_attn_args)
            for k in self.player_keys
        })

        player_post_gate_attn_proj_args = dict(
            type_norm="rms",
            dim_start=dim_middle_player_gate_attn,
            dim_middle=dim_middle_player_gate_attn,
            dim_end=dim_start_player_post_gate_attn_mlp,
            biases=(False, False), norm_bias=False,
            pow_iters=pow_iters
        )
        self.player_post_gate_attn_projs = nn.ModuleDict({
            k: Normalizer(**player_post_gate_attn_proj_args)
            for k in self.player_keys
        })

        player_post_gate_attn_mlp_args = dict(
            residual=False, num_mlps=num_mlps_player_post_gate_attn_mlp,
            dim_start=dim_start_player_post_gate_attn_mlp,
            dim_end=dim_end_player_post_gate_attn_mlp,
            norms=[""]*num_mlps_player_post_gate_attn_mlp,
            dropouts=[0.0]*num_mlps_player_post_gate_attn_mlp,
            biases=[(False, False)]*num_mlps_player_post_gate_attn_mlp,
            pow_iters=pow_iters
        )
        self.player_post_gate_attn_mlp_encs = nn.ModuleDict({
            k: MLPEncoder(**player_post_gate_attn_mlp_args)
            for k in self.player_keys
        })
        self.player_post_gate_attn_mlp_drops = nn.ModuleDict({
            k: nn.Dropout(dropout_player_post_gate_attn_mlp)
            for k in self.player_keys
        })

        player_pre_rnn_proj_args = dict(
            type_norm="rms",
            dim_start=dim_end_player_post_gate_attn_mlp,
            biases=(False, False), norm_bias=False,
            pow_iters=pow_iters
        )
        player_pre_rnn_projs_tm = {
            k: Normalizer(**(
                player_pre_rnn_proj_args | \
                {"dim_middle": dim_start_rnn_tm, "dim_end": dim_start_rnn_tm}
            ))
            for k in self.player_keys if "_tm" in k
        }
        player_pre_rnn_projs_op = {
            k: Normalizer(**(
                player_pre_rnn_proj_args | \
                {"dim_middle": dim_start_rnn_op, "dim_end": dim_start_rnn_op}
            ))
            for k in self.player_keys if "_op" in k
        }
        self.player_pre_rnn_projs = nn.ModuleDict(
            (player_pre_rnn_projs_tm | player_pre_rnn_projs_op))

        self.player_post_gate_attn_blocks = nn.ModuleDict({
            k: nn.Sequential(
                self.player_post_gate_attn_mlp_encs[k],
                self.player_post_gate_attn_mlp_drops[k],
                self.player_pre_rnn_projs[k]
            )
            for k in self.player_keys
        })

        if rnn_type.lower() == "gru":  RNN = nn.GRU
        elif rnn_type.lower() == "lstm":  RNN = nn.LSTM
        else:  RNN = nn.RNN
        dim_start_rnn_d = {
            k: dim_start_rnn_tm if "_tm" in k else dim_start_rnn_op
            for k in self.player_keys_d.keys()
        }
        dim_end_rnn_d = {
            k: dim_end_rnn_tm if "_tm" in k else dim_end_rnn_op
            for k in self.player_keys_d.keys()
        }
        num_layers_rnn_d = {
            k: num_layers_rnn_tm if "_tm" in k else num_layers_rnn_op
            for k in self.player_keys_d.keys()
        }
        dropout_rnn_d = {
            k: dropout_rnn_tm if "_tm" in k else dropout_rnn_op
            for k in self.player_keys_d.keys()
        }

        self.player_rnns = nn.ModuleDict({
            k: RNN(
                input_size=dim_start_rnn_d[k],
                hidden_size=dim_end_rnn_d[k],
                num_layers=num_layers_rnn_d[k],
                bias=False,
                batch_first=False,
                dropout=dropout_rnn_d[k],
                bidirectional=False
            )
            for k in self.player_keys_d.keys()
        })

        player_post_rnn_proj_args = dict(
            type_norm="rms",
            dim_middle=dim_embedding_softmax,
            dim_end=dim_embedding_softmax,
            biases=(False, False), norm_bias=False,
            pow_iters=pow_iters
        )
        player_post_rnn_projs_tm = {
            k: Normalizer(**(
                player_post_rnn_proj_args | {"dim_start": dim_end_rnn_tm}
            ))
            for k in ["def_tm", "off_tm"]
        }
        player_post_rnn_projs_op = {
            k: Normalizer(**(
                player_post_rnn_proj_args | {"dim_start": dim_end_rnn_op}
            ))
            for k in ["def_op", "off_op"]
        }
        self.player_post_rnn_projs = nn.ModuleDict(
            (player_post_rnn_projs_tm | player_post_rnn_projs_op))

        self.player_softmax_softmaxes = nn.ModuleDict({
            k: nn.Softmax(dim=0) for k in self.player_keys_d.keys()
        })
        self.tau_softmax = tau_softmax
        self.player_softmax_scales = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(1)) for k in self.player_keys_d.keys()
        })
        self.player_softmax_mixes = nn.ParameterDict({
            k: nn.Parameter(torch.zeros(1)) for k in self.player_keys_d.keys()
        })

        if dim_embedding_softmax != dim_start_player_pre_fusion_mlp:
            self.player_pre_fusion_mlp_projs = {
                k: LLinear(
                    dim_start=dim_embedding_softmax,
                    dim_end=dim_start_player_pre_fusion_mlp,
                    pow_iters=pow_iters
                )
                for k in self.player_keys_d.keys()
            }
        else:
            self.player_pre_fusion_mlp_projs = nn.ModuleDict({
                k: nn.Identity() for k in self.player_keys_d.keys()
            })

        player_pre_fusion_mlp_args = dict(
            residual=False, num_mlps=num_mlps_player_pre_fusion_mlp,
            dim_start=dim_start_player_pre_fusion_mlp,
            dim_end=dim_end_player_pre_fusion_mlp,
            norms=[""]*num_mlps_player_pre_fusion_mlp,
            dropouts=[0.0]*num_mlps_player_pre_fusion_mlp,
            biases=[(False, False)]*num_mlps_player_pre_fusion_mlp,
            pow_iters=pow_iters
        )
        self.player_pre_fusion_mlp_encs = nn.ModuleDict({
            k: MLPEncoder(**player_pre_fusion_mlp_args)
            for k in self.player_keys_d.keys()
        })
        self.player_pre_fusion_mlp_drops = nn.ModuleDict({
            k: nn.Dropout(dropout_player_pre_fusion_mlp)
            for k in self.player_keys_d.keys()
        })

        self.player_pre_fusion_blocks = nn.ModuleDict({
            k: nn.Sequential(
                self.player_pre_fusion_mlp_projs[k],
                self.player_pre_fusion_mlp_encs[k],
                self.player_pre_fusion_mlp_drops[k]
            )
            for k in self.player_keys_d.keys()
        })

        player_fusion_proj_args = dict(
            type_norm="",
            dim_start=dim_end_player_pre_fusion_mlp,
            dim_middle=dim_start_player_fusion,
            dim_end=dim_end_player_fusion,
            biases=(False, False), norm_bias=False,
            pow_iters=pow_iters
        )
        self.player_fusion_projs = nn.ModuleDict({
            k: Normalizer(**player_fusion_proj_args)
            for k in self.player_keys_d.keys()
        })

        self.player_post_fusion_mlp_projs = nn.ModuleDict({
            "def": LLinear(
                dim_start=dim_end_player_fusion,
                dim_end=dim_start_player_post_fusion_mlp,
                pow_iters=pow_iters
            ),
            "off": LLinear(
                dim_start=dim_end_player_fusion,
                dim_end=dim_start_player_post_fusion_mlp,
                pow_iters=pow_iters
            )
        })

        player_post_fusion_mlp_args = dict(
            residual=False, num_mlps=num_mlps_player_post_fusion_mlp,
            dim_start=dim_start_player_post_fusion_mlp,
            dim_end=dim_end_player_post_fusion_mlp,
            norms=[""]*num_mlps_player_post_fusion_mlp,
            dropouts=[0.0]*num_mlps_player_post_fusion_mlp,
            biases=[(False, False)]*num_mlps_player_post_fusion_mlp,
            pow_iters=pow_iters
        )
        self.player_post_fusion_mlp_encs = nn.ModuleDict({
            k: MLPEncoder(**player_post_fusion_mlp_args)
            for k in ["def", "off"]
        })
        self.player_post_fusion_mlp_drops = nn.ModuleDict({
            k: nn.Dropout(dropout_player_post_fusion_mlp)
            for k in ["def", "off"]
        })

        self.player_post_fusion_blocks = nn.ModuleDict({
            k: nn.Sequential(
                self.player_post_fusion_mlp_projs[k],
                self.player_post_fusion_mlp_encs[k],
                self.player_post_fusion_mlp_drops[k]
            )
            for k in ["def", "off"]
        })

        player_conditioning_proj_args = dict(
            type_norm="",
            dim_start=dim_end_player_post_fusion_mlp,
            dim_middle=dim_start_conditioning,
            dim_end=dim_end_conditioning,
            biases=(False, True), norm_bias=False,
            pow_iters=pow_iters
        )
        self.player_conditioning_projs = nn.ModuleDict({
            "def": Normalizer(**player_conditioning_proj_args),
            "off": Normalizer(**player_conditioning_proj_args)
        })

        if self.include_idx:
            dim_embedding_global = \
                dim_end_play + dim_stp + dim_idx + dim_snap
        else:
            dim_embedding_global = dim_end_play + dim_stp + dim_snap

        global_conditioning_proj_args = dict(
            type_norm="",
            dim_start=dim_embedding_global,
            dim_middle=dim_start_conditioning,
            dim_end=dim_end_conditioning,
            biases=(False, True), norm_bias=False,
            pow_iters=pow_iters
        )
        self.global_conditioning_proj = Normalizer(
            **global_conditioning_proj_args)

        self.conditioning_w1 = LLinear(
            dim_start=dim_end_conditioning,
            dim_end=dim_end_conditioning,
            pow_iters=pow_iters
        )
        self.conditioning_w2 = nn.ModuleDict({
            "def": LLinear(
                dim_start=dim_end_conditioning,
                dim_end=dim_end_conditioning,
                pow_iters=pow_iters
            ),
            "off": LLinear(
                dim_start=dim_end_conditioning,
                dim_end=dim_end_conditioning,
                pow_iters=pow_iters
            ),
        })
        self.conditioning_v = nn.ModuleDict({
            "def": LLinear(
                dim_start=dim_end_conditioning,
                dim_end=dim_end_conditioning,
                pow_iters=pow_iters
            ),
            "off": LLinear(
                dim_start=dim_end_conditioning,
                dim_end=dim_end_conditioning,
                pow_iters=pow_iters
            ),
        })
        self.conditioning_u = nn.ModuleDict({
            "def": MLPEncoder(
                residual=False, num_mlps=1,
                dim_start=dim_end_conditioning,
                dim_end=dim_end_conditioning,
                norms=[""], dropouts=[0.0],
                biases=[(False, False)],
                pow_iters=pow_iters
            ),
            "off": MLPEncoder(
                residual=False, num_mlps=1,
                dim_start=dim_end_conditioning,
                dim_end=dim_end_conditioning,
                norms=[""], dropouts=[0.0],
                biases=[(False, False)],
                pow_iters=pow_iters
            ),
        })
        self.tau_conditioning = nn.Parameter(torch.ones(1))
        with torch.no_grad():
            self.tau_conditioning.data.normal_(1.0, 0.01)

        post_conditioning_proj_args = dict(
            type_norm="layer",
            dim_start=dim_end_conditioning,
            dim_middle=dim_output//2,
            dim_end=dim_output,
            biases=(False, True), norm_bias=False,
            pow_iters=pow_iters
        )
        self.post_conditioning_projs = nn.ModuleDict({
            "def": Normalizer(**post_conditioning_proj_args),
            "off": Normalizer(**post_conditioning_proj_args)
        })

    def apply_attention(self, v, k, s, spl):
        if spl:
            norm1 = self.player_pre_attn_projs[k][s]
            attn = self.player_attn_encs[k][s]
            norm2 = self.player_post_attn_projs[k][s]
        else:
            norm1 = self.player_pre_gate_attn_projs[k]
            attn = self.player_gate_attn_encs[k]
            norm2 = self.player_post_gate_attn_projs[k]
        xv = norm1(v)
        xv_shape = xv.shape
        TBN, d = xv_shape
        T = self.trajectory_length - 2
        N = self.player_count * 2
        if spl:
            B = (TBN // T) // N
            xx = xv.reshape(2, T, B, N//2, -1)
            xq, xk = xx.unbind(dim=0)
            xq, xk = \
                xq.permute(2, 0, 1, 3), xk.permute(2, 0, 1, 3)
        else:
            B = (TBN // T) // (N// 2)
            xq, xk = xv.clone(), xv.clone()
        xq, xk = \
            xq.reshape(N//2, T*B, -1), xk.reshape(N//2, T*B, -1)
        a = attn(xq, xk, xk)
        ar = a.reshape(T*B*(N//2), -1)
        aa = norm2(ar)
        return aa
    def apply_role_gate(self, d, k):
        avp = d["acceleration"] + d["velocity"] + d["position"]
        xy = d["x"] + d["y"]
        v1 = torch.cat([avp, xy], dim=-1)
        v2 = self.player_kin_mlp_encs[k](self.player_kin_mlp_projs[k](v1))
        k_enc = self.player_kin_gate_projs[k](v2)
        r_enc = self.player_role_gate_projs[k](d["role"])
        gate = k_enc + torch.sigmoid(self.player_gate_scales[k](r_enc))
        return gate
    def apply_recurrent_aggregation(self, d, k):
        xc = torch.stack(
            [d[kl] for kl in d.keys() if k in kl], dim=0
        ).contiguous()
        xc_shape = xc.shape
        L, NTB, d = xc_shape
        zx, _ = self.player_rnns[k](xc)
        zr1 = zx.reshape(L * NTB, -1)
        zl = self.player_post_rnn_projs[k](zr1)
        zr2 = zl.reshape(L, NTB, -1)
        zt = (1.0 + (
            torch.sigmoid(torch.tensor(self.tau_softmax)).item() *
            torch.sigmoid(self.player_softmax_scales[k])
        ))
        zv = zr2 / zt
        zd = self.player_softmax_softmaxes[k](zv)
        zp = (
            ((1 - torch.sigmoid(self.player_softmax_mixes[k])) * zd) +
            (torch.sigmoid(self.player_softmax_mixes[k]) / zd.size(0))
        )
        zf = (zv * zp).sum(dim=0)
        return zf
    def apply_product(self, d, k):
        ktm, kop = f"{k}_tm", f"{k}_op"
        z0 = (
            (self.player_fusion_projs[ktm](d[ktm]) *
            self.player_fusion_projs[kop](d[kop])) +
            1e-6
        )
        z1 = self.player_post_fusion_blocks[k](z0)
        return z1
    def apply_global_conditioning(self, v, g, h, k):
        z_u = torch.sigmoid(self.conditioning_u[k](g))
        z_v = self.conditioning_v[k](v * z_u)
        z_flr = torch.sigmoid((v.size(-1) / torch.linalg.norm(v))).item()
        z_flr *= 2.0
        z_tau = z_flr + (
            (2.0 - z_flr) * torch.sigmoid(self.tau_conditioning).item()
        )
        z_w2 = self.conditioning_w2[k](torch.tanh(z_tau * z_v))
        z = v + h + z_w2
        return z

    def block_1(self, x, s, k):
        return (
            self.player_post_attn_blocks[k][s](
                self.apply_attention(
                    self.player_pre_attn_blocks[k][s](x),
                    k, s, True
                )
            )
        )
    def block_2(self, x, k):
        return (
            self.player_post_gate_attn_blocks[k](
                self.apply_attention(
                    self.player_pre_gate_attn_blocks[k](
                        self.apply_role_gate(x, k)
                    ),
                    k, "", False
                )
            )
        )
    def block_3(self, x, k):
        return (
            self.player_pre_fusion_blocks[k](
                self.apply_recurrent_aggregation(x, k)
            )
        )
    def block_4(self, x, g, h, k):
        return (
            self.apply_global_conditioning(
                self.player_conditioning_projs[k](
                    self.apply_product(x, k)
                ),
                g, h, k
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x

        bs = x.shape[0] // \
            (1 * (self.trajectory_length - 1) * (self.player_count * 2))
        x = x.reshape(
            (self.trajectory_length - 1), bs, (self.player_count * 2), -1)
        x_shape = x.shape
        T, B, N, d = x_shape

        assert (T - 1) > 1 and self.trajectory_length > 3
        assert N == (self.player_count * 2)

        T -= 1
        x_play = x[1, :, 0, :global_dim-1].clone().detach()
        x_len = x[1, :, 0, global_dim-1].clone().detach()
        x_rest = x[1:, ..., global_dim:].clone().detach()
        x_player = x_rest[..., :-2]
        x_snap = \
            x_rest[:, :, 0, player_tensor_indices["snap"][1]].clone().detach()
        x_snap = x_snap.unsqueeze(-1).reshape(-1, 1)

        e_play = self.play_enc(x_play)
        z1_play = e_play.unsqueeze(0).expand(T, B, -1)
        z_play = z1_play.unsqueeze(2).expand(T, B, N//2, -1)

        e_timestep = self.timestep_enc(x_len)
        z_timestep = e_timestep.unsqueeze(2).expand(-1, -1, N//2, -1)

        e_snap = self.snap_enc(x_snap)
        z1_snap = e_snap.reshape(T, B, -1)
        z_snap = z1_snap.unsqueeze(2).expand(T, B, N//2, -1)

        e_index = self.index_enc(x_rest)
        z1_index = e_index.unsqueeze(0).expand(B, -1, -1)
        z_index = z1_index.unsqueeze(0).expand(T, -1, -1, -1)

        if self.include_idx:
            z_g_cat = \
                torch.cat([z_play, z_timestep, z_index, z_snap], dim=-1)
        else:
            z_g_cat = torch.cat([z_play, z_timestep, z_snap], dim=-1)
        z_g_cat_r = z_g_cat.reshape(T*B*(N//2), -1)
        z_g = self.global_conditioning_proj(z_g_cat_r)
        z_w1 = self.conditioning_w1(z_g)

        if self.backwards:
            x_rest[T-1, :, :N//2, -1] = 0
            x_rest[T-1, :, N//2:, -2] = 0

        x_def = x_player[
            (x_rest[..., -2] == 1)
        ].reshape(T, B, N//2, -1).clone().detach()
        x_off = x_player[
            (x_rest[..., -1] == 1)
        ].reshape(T, B, N//2, -1).clone().detach()

        x_d = {}
        for kx, lags in self.player_keys_d.items():
            xd = {}
            if "def" in kx:
                xd["0"] = x_def.clone()
                xd["00"] = x_off.clone()
            if "off" in kx:
                xd["0"] = x_off.clone()
                xd["00"] = x_def.clone()
            if "tm" in kx:  xd["1"] = xd["0"].clone()
            if "op" in kx:  xd["1"] = xd["00"].clone()
            for lag_str in lags:
                lag = int(lag_str[-4:])
                x_lag = torch.zeros((lag, B, N//2, agent_dim-2)).to(x.device)
                x_f_lag = torch.cat(
                    [x_lag.clone(), xd["1"][:T-lag, ...].clone()], dim=0)
                x_d[lag_str] = torch.cat(
                    [xd["0"].clone(), x_f_lag.clone()], dim=2).detach()

        reentrant = False
        z03_d, z04_d, z05_d = {}, {}, {}
        for k0 in x_d.keys():
            z02_d = {}
            for fs in self.player_feature_slices:
                z00 = x_d[k0].reshape(T*B*N, -1)
                z01 = self.player_initial_projs[k0][fs](z00)
                if self.checkpoints:
                    z02_d[fs] = torch.utils.checkpoint.checkpoint(
                        lambda x, f_s=fs, kk0=k0: self.block_1(x, f_s, kk0),
                        z01,
                        use_reentrant=reentrant
                    )
                else:
                    z02_d[fs] = self.block_1(z01, fs, k0)
            if self.checkpoints:
                z03_d[k0] = torch.utils.checkpoint.checkpoint(
                    lambda x, kk1=k0: self.block_2(x, kk1), z02_d,
                    use_reentrant=reentrant
                )
            else:
                z03_d[k0] = self.block_2(z02_d, k0)
        for k1 in self.player_keys_d.keys():
            if self.checkpoints:
                z04_d[k1] = torch.utils.checkpoint.checkpoint(
                    lambda x, kk2=k1: self.block_3(x, kk2), z03_d,
                    use_reentrant=reentrant
                )
            else:
                z04_d[k1] = self.block_3(z03_d, k1)
        for k2 in ["def", "off"]:
            if self.checkpoints:
                z05_d[k2] = torch.utils.checkpoint.checkpoint(
                    lambda x, kk3=k2: self.block_4(x, z_g, z_w1, kk3), z04_d,
                    use_reentrant=reentrant
                )
            else:
                z05_d[k2] = self.block_4(z04_d, z_g, z_w1, k2)
        z06_def = self.post_conditioning_projs["def"](z05_d["def"])
        z06_off = self.post_conditioning_projs["off"](z05_d["off"])

        z0_f = torch.stack([z06_def, z06_off], dim=0)
        z_f = z0_f.reshape(2, T, B, N//2, -1)

        return z_f
