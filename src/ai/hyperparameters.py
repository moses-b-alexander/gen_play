
from ai.utils import inv_sqrt


def get_encoder_hyperparameters(
    lags: tuple[list[int]],
    dim_play: int, dim_player: int,
    num_heads: int, dropout: float,
    expansion: int
) -> dict:
    num_mlps = {
        "play": 3,
        "pre_attn": 1,
        "post_attn": 1,
        "kin": 1,
        "pre_gate": 1,
        "post_gate": 1,
        "pre_fusion": 2,
        "post_fusion": 2,
        "conditioning": 1,
    }

    taus = {
        "stp": 1.00,
        "pre_gate": 0.20,
        "post_gate":0.20,
        "temporal": 0.50,
    }

    num_rnns = {
        "tm": 1,
        "op": 1,
    }

    def get_size(k: str) -> int:
        return expansion ** num_mlps[k] if num_mlps[k] > 1 else 1

    d = {
        "lags": lags,

        "num_mlps_play": num_mlps["play"],
        "dim_projection_play": dim_play,
        "dim_start_play": dim_play,
        "dim_end_play": get_size("play")*dim_play,
        "dropout_play": dropout,

        "dim_stp": dim_play,
        "tau_stp": taus["stp"],

        "dim_idx": dim_play,

        "dim_snap": dim_play,

        "dim_start_player_initial_proj": dim_player,
        "dim_end_player_initial_proj": dim_player,

        "num_mlps_player_pre_attn_mlp": num_mlps["pre_attn"],
        "dim_start_player_pre_attn_mlp": dim_player,
        "dim_end_player_pre_attn_mlp": get_size("pre_attn")*dim_player,
        "dropout_player_pre_attn_mlp": dropout,

        "num_heads_player_attn": num_heads,
        "dim_middle_player_attn": dim_player,
        "tau_player_attn": taus["pre_gate"],
        "tau_stdev_player_attn": taus["pre_gate"]*inv_sqrt(dim_player),
        "dropout_player_attn": dropout,

        "num_mlps_player_post_attn_mlp": num_mlps["post_attn"],
        "dim_start_player_post_attn_mlp": dim_player,
        "dim_end_player_post_attn_mlp": get_size("post_attn")*dim_player,
        "dropout_player_post_attn_mlp": dropout,

        "num_mlps_player_kin_mlp": num_mlps["kin"],
        "dim_start_player_kin_mlp": dim_player,
        "dim_end_player_kin_mlp": get_size("kin")*dim_player,

        "dim_start_player_gate": dim_player,
        "num_mlps_player_pre_gate_attn_mlp": num_mlps["pre_gate"],
        "dim_start_player_pre_gate_attn_mlp": dim_player,
        "dim_end_player_pre_gate_attn_mlp": get_size("pre_gate")*dim_player,
        "dropout_player_pre_gate_attn_mlp": dropout,

        "num_heads_player_gate_attn": num_heads,
        "dim_middle_player_gate_attn": dim_player,
        "tau_player_gate_attn": taus["post_gate"],
        "tau_stdev_player_gate_attn": taus["post_gate"]*inv_sqrt(dim_player),
        "dropout_player_gate_attn": dropout,

        "num_mlps_player_post_gate_attn_mlp": num_mlps["post_gate"],
        "dim_start_player_post_gate_attn_mlp": dim_player,
        "dim_end_player_post_gate_attn_mlp": get_size("post_gate")*dim_player,
        "dropout_player_post_gate_attn_mlp": dropout,

        "rnn_type": "gru",
        "dim_start_rnn_tm": dim_player,
        "dim_start_rnn_op": dim_player,
        "dim_end_rnn_tm": dim_player,
        "dim_end_rnn_op": dim_player,
        "num_layers_rnn_tm": num_rnns["tm"],
        "num_layers_rnn_op": num_rnns["op"],
        "dropout_rnn_tm": dropout if num_rnns["tm"] > 1 else 0.00,
        "dropout_rnn_op": dropout if num_rnns["op"] > 1 else 0.00,
        "tau_softmax": taus["temporal"],
        "dim_embedding_softmax": dim_player,

        "num_mlps_player_pre_fusion_mlp": num_mlps["pre_fusion"],
        "dim_start_player_pre_fusion_mlp": dim_player,
        "dim_end_player_pre_fusion_mlp": get_size("pre_fusion")*dim_player,
        "dropout_player_pre_fusion_mlp": dropout,

        "dim_start_player_fusion": dim_player,
        "dim_end_player_fusion": dim_player,

        "num_mlps_player_post_fusion_mlp": num_mlps["post_fusion"],
        "dim_start_player_post_fusion_mlp": dim_player,
        "dim_end_player_post_fusion_mlp": get_size("post_fusion")*dim_player,
        "dropout_player_post_fusion_mlp": dropout,

        "dim_start_conditioning": dim_player,
        "dim_end_conditioning": dim_player*expansion,
    }

    return d

def get_diffeq_hyperparameters() -> dict:

    return {}

def get_decoder_hyperparameters(
    min_s: float, max_s: float
) -> dict:
    d = {"min_stdv": min_s, "max_stdv": max_s}

    return d
