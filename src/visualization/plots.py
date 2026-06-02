
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import seaborn as sns
import torch

from common.constants import all_players, fps, screen_mode, team_players
from common.dirs import assets_dir, output_dir
from data.constants import (
    x_field_max, x_field_min, x_endzone_max, x_endzone_min,
    y_bnd, y_hash, y_mid
)
from visualization.constants import clr, fig_x, fig_y


def plot_field(ax: Axes) -> Axes:
    x_extent = [
        x_endzone_min, x_endzone_min, x_endzone_min, x_endzone_max,
        x_endzone_max, x_endzone_max, x_endzone_max, x_endzone_min,
    ]
    y_extent = [
        -np.abs(y_bnd), +np.abs(y_bnd), -np.abs(y_bnd), -np.abs(y_bnd),
        +np.abs(y_bnd), +np.abs(y_bnd), +np.abs(y_bnd), +np.abs(y_bnd),
    ]

    ax.plot(x_extent, y_extent, color=clr, lw=1)
    ax.invert_yaxis()

    for b in [float(x_field_min), float(x_field_max)]:
        ax.plot(
            [b, b], [-np.abs(y_bnd), +np.abs(y_bnd)],
            color=clr, lw=2, zorder=1,
        )

    canvas = patches.Rectangle(
        (0, -np.abs(y_bnd)), x_field_max, 2 * np.abs(y_bnd),
        linewidth=0.1, edgecolor=clr, facecolor="#4a7c41", zorder=0,
    )
    ax.add_patch(canvas)

    for line in range(x_field_min, x_field_max, 10):
        ax.plot(
            [line, line], [-np.abs(y_bnd), +np.abs(y_bnd)],
            ls="-", color=clr, zorder=1, alpha=0.5,
        )

    ax.plot(
        [x_field_min, x_field_max],
        [-np.abs(y_bnd * 0.75), -np.abs(y_bnd * 0.75)],
        ls="-.", color=clr, zorder=1, alpha=0.5,
    )
    ax.plot(
        [x_field_min, x_field_max],
        [+np.abs(y_bnd * 0.75), +np.abs(y_bnd * 0.75)],
        ls="-.", color=clr, zorder=1, alpha=0.5,
    )

    for dash_line in range(5, x_field_max, 5):
        ax.plot(
            [dash_line, dash_line], [-np.abs(y_bnd), +np.abs(y_bnd)],
            ls="--", color=clr, lw=1.0, alpha=0.5, zorder=1,
        )

    for x in np.linspace(x_field_min + 1, x_field_max - 1, 98):
        ax.plot(
            [x, x], [-np.abs(y_bnd), -np.abs(y_bnd) + 0.5],
            color=clr, lw=1.0, alpha=0.5, zorder=1,
        )
        ax.plot(
            [x, x], [+np.abs(y_bnd), +np.abs(y_bnd) - 0.5],
            color=clr, lw=1.0, alpha=0.5, zorder=1,
        )
        ax.plot(
            [x, x], [+np.abs(y_hash), +np.abs(y_hash) - 0.5],
            color=clr, lw=1.0, alpha=0.5, zorder=1,
        )
        ax.plot(
            [x, x], [-np.abs(y_hash), -np.abs(y_hash) + 0.5],
            color=clr, lw=1.0, alpha=0.5, zorder=1,
        )

    coords = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    values = ["10", "20", "30", "40", "50", "60", "70", "80", "90"]
    for x, v in zip(coords, values):
        ax.annotate(
            v, (x, +np.abs(y_bnd) + 1.5),
            ha="center", va="center", color=clr,
        )
        ax.annotate(
            v, (x, -np.abs(y_bnd) - 1.5),
            ha="center", va="center", color=clr,
        )

    x_lim = x_endzone_min - 5, x_endzone_max + 5
    y_lim = +y_bnd + 10, -y_bnd - 10

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)

    ax.axis("off")

    return ax

def plot_init(split: bool=False) -> tuple[Axes, Axes] | Axes:
    if split:
        fig, (ax_t, ax_b) = plt.subplots(
            nrows=2, ncols=1,
            figsize=(fig_x, fig_y * 2),
            sharex=True, sharey=True,
            gridspec_kw={"hspace": (0.06 + 1e-12)}
        )
        if screen_mode.lower() == "dark":
            ax_t.set_facecolor("#222222")
            ax_b.set_facecolor("#222222")
        plot_field(ax_t)
        plot_field(ax_b)
        return (ax_t, ax_b)
    else:
        fig, ax = plt.subplots(figsize=(fig_x, fig_y))
        if screen_mode.lower() == "dark":  ax.set_facecolor("#222222")
        plot_field(ax)
        return ax

def plot_play_1(
    ax: Axes | None,
    tups: tuple[tuple],
    max_t: int, inc: int=1, real: bool=True
) -> tuple[Axes, Slider]:
    if ax is None:  ax = plot_init(split=False)

    fig = ax.figure
    if real:  fig.suptitle("real")
    else:  fig.suptitle("generated")
    plt.subplots_adjust(bottom=0.1)

    (xs0, xs1), (ys0, ys1), (sx, sy) = tups
    x1, x2 = xs0[1], xs1[1]
    y1, y2 = ys0[1], ys1[1]

    vline = ax.axvline(x=sx, color="#008000", linestyle="--", zorder=8)
    sc1 = ax.scatter(x1, y1, s=30, c="#0000FF", zorder=9)
    sc2 = ax.scatter(x2, y2, s=30, c="#FFA500", zorder=9)

    def update(i):
        ii = int(i)
        iii = 1 if ii <= 0 else (max_t - 2 if ii >= max_t - 1 else ii)
        x1, x2 = xs0[iii], xs1[iii]
        y1, y2 = ys0[iii], ys1[iii]
        sc1.set_offsets(np.c_[x1, y1])
        sc2.set_offsets(np.c_[x2, y2])
        fig.canvas.draw_idle()

    s_ax = fig.add_axes([0.155, 0.050, 0.720, 0.050]) # good enough
    inc_r = inc if inc >= 1 and inc <= max_t - 3 else 1
    slider = Slider(
        ax=s_ax, label="FRAME",
        valmin=1, valmax=max_t-2,
        valinit=1, valstep=inc_r
    )
    slider.on_changed(update)

    return (ax, slider)

def plot_play_2(
    axs: tuple[Axes | None, Axes | None] | None,
    tups_r: tuple[tuple], tups_g: tuple[tuple],
    max_t: int, inc: int=1,
    show_def_to_off: bool=True,
    show_off_to_def: bool=False,
) -> tuple[tuple[Axes], Slider]:
    if axs is None or axs[0] is None or axs[1] is None or axs == ():
        axs = plot_init(split=True)

    fig = axs[0].figure
    axs[0].set_title("REAL", fontstyle="italic", loc="left")
    axs[1].set_title("GENERATED", fontstyle="italic", loc="right")
    plt.subplots_adjust(bottom=0.14)

    (xs0r, xs1r), (ys0r, ys1r), (sxr, syr) = tups_r
    x1r, x2r = xs0r[1], xs1r[1]
    y1r, y2r = ys0r[1], ys1r[1]

    (xs0g, xs1g), (ys0g, ys1g), (sxg, syg) = tups_g
    x1g, x2g = xs0g[1], xs1g[1]
    y1g, y2g = ys0g[1], ys1g[1]

    vliner = axs[0].axvline(x=sxr, color="#020402", linestyle="--", zorder=8)
    sc1r = axs[0].scatter(x1r, y1r, s=30, c="#0000FF", zorder=9)
    sc2r = axs[0].scatter(x2r, y2r, s=30, c="#FFA500", zorder=9)

    vlineg = axs[1].axvline(x=sxg, color="#020402", linestyle="--", zorder=8)
    sc1g = axs[1].scatter(x1g, y1g, s=30, c="#0000FF", zorder=9)
    sc2g = axs[1].scatter(x2g, y2g, s=30, c="#FFA500", zorder=9)

    sep_lines_r: list = []
    sep_lines_g: list = []

    def _draw_sep_lines(ax, x_off, x_def, y_off, y_def, store: list) -> None:
        for a in store:
            a.remove()
        store.clear()
        px = np.concatenate([np.asarray(x_off), np.asarray(x_def)])
        py = np.concatenate([np.asarray(y_off), np.asarray(y_def)])
        is_off = np.array([True] * len(x_off) + [False] * len(x_def))
        for n in range(len(px)):
            if np.isnan(px[n]) or np.isnan(py[n]):
                continue
            if is_off[n] and not show_def_to_off:
                continue
            if not is_off[n] and not show_off_to_def:
                continue
            opp_idx = np.where(is_off != is_off[n])[0]
            dists = np.sqrt((px[opp_idx] - px[n])**2 + (py[opp_idx] - py[n])**2)
            valid = ~np.isnan(dists)
            if valid.any():
                nn = opp_idx[np.nanargmin(dists)]
                if not (np.isnan(px[nn]) or np.isnan(py[nn])):
                    arr = ax.annotate(
                        "",
                        xy=(px[nn], py[nn]),
                        xytext=(px[n], py[n]),
                        arrowprops=dict(
                            arrowstyle="-|>",
                            color="#FFFF00",
                            lw=1.0,
                            alpha=0.35,
                            mutation_scale=6,
                        ),
                        zorder=1,
                    )
                    store.append(arr)

    _draw_sep_lines(axs[0], x1r, x2r, y1r, y2r, sep_lines_r)
    _draw_sep_lines(axs[1], x1g, x2g, y1g, y2g, sep_lines_g)

    def update(i):
        ii = int(i)
        iii = 1 if ii <= 0 else (max_t - 2 if ii >= max_t - 1 else ii)

        x1r, x2r = xs0r[iii], xs1r[iii]
        y1r, y2r = ys0r[iii], ys1r[iii]
        sc1r.set_offsets(np.c_[x1r, y1r])
        sc2r.set_offsets(np.c_[x2r, y2r])
        _draw_sep_lines(axs[0], x1r, x2r, y1r, y2r, sep_lines_r)

        x1g, x2g = xs0g[iii], xs1g[iii]
        y1g, y2g = ys0g[iii], ys1g[iii]
        sc1g.set_offsets(np.c_[x1g, y1g])
        sc2g.set_offsets(np.c_[x2g, y2g])
        _draw_sep_lines(axs[1], x1g, x2g, y1g, y2g, sep_lines_g)

        fig.canvas.draw_idle()

    s_ax = fig.add_axes([0.155, 0.050, 0.720, 0.050]) # good enough
    inc_r = inc if inc >= 1 and inc <= max_t - 3 else 1
    slider = Slider(
        ax=s_ax, label="FRAME",
        valmin=1, valmax=max_t-2,
        valinit=1, valstep=inc_r
    )
    slider.on_changed(update)

    return (axs, slider)

def plot_play_features(
    play_df,
    inc: int = 1,
) -> tuple[plt.Figure, Slider]:
    """
    Interactive slider plot with two linked panels:
      - top:    per-player separation bar chart
      - bottom: play-level feature time series with a frame cursor
    """
    N = all_players
    frames = play_df.reset_index(drop=True)
    max_t = len(frames)

    # ── figure layout ─────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(9, 8))
    gs = fig.add_gridspec(2, 1, hspace=0.40,
                          left=0.12, right=0.95, top=0.92, bottom=0.15)
    ax_sep    = fig.add_subplot(gs[0])
    ax_series = fig.add_subplot(gs[1])

    # ── static time-series curves ──────────────────────────────────────────────
    t = frames["play_time_since_snap"].values
    for feat_col, feat_clr, feat_lbl in [
        ("play_qb_pressure", "#FFD700", "qb_pressure"),
        ("play_tgt_sep",     "#00FFFF", "tgt_sep"),
        ("play_tgt_depth",   "#00FF00", "tgt_depth"),
    ]:
        vals = (frames[feat_col].values if feat_col in frames.columns
                else np.full(len(t), np.nan))
        ax_series.plot(t, vals, color=feat_clr, label=feat_lbl, lw=1.5)
    ax_series.axvline(0, color="#FFFFFF", lw=1, ls="--", alpha=0.6, label="snap")
    init_idx = min(1, max_t - 1)
    cursor_vline = ax_series.axvline(t[init_idx], color="#FFA500", lw=1.2,
                                     ls=":", label="frame")
    ax_series.set_facecolor("#1e1e1e")
    ax_series.set_xlabel("time_since_snap")
    ax_series.set_title("Play-level features over time", fontsize=9)
    ax_series.legend(fontsize=7)
    ax_series.set_xlim(t.min(), t.max())

    # ── per-frame helper ───────────────────────────────────────────────────────
    def _frame_arrays(idx):
        row = frames.iloc[idx]
        def _col(prefix, default=np.nan):
            return np.array([row.get(f"{prefix}-{n:02d}", default)
                             for n in range(1, N + 1)])
        px    = _col("player_x")
        sep   = _col("player_sep")
        off   = _col("player_offense", 0.0) == 1
        is_qb = _col("player_position_qb", 0.0) == 1
        is_wr = _col("player_position_wr", 0.0) == 1
        is_te = _col("player_position_te", 0.0) == 1
        is_rb = _col("player_position_rb", 0.0) == 1
        return px, sep, off, is_qb, is_wr, is_te, is_rb

    def _refresh_sep(idx):
        ax_sep.cla()
        px, sep, off, is_qb, is_wr, is_te, is_rb = _frame_arrays(idx)
        valid = ~np.isnan(sep) & ~np.isnan(px)
        labels = []
        for n in range(N):
            if not valid[n]:
                continue
            role = ("QB" if is_qb[n] else ("WR" if is_wr[n] else
                    ("TE" if is_te[n] else ("RB" if is_rb[n] else
                    ("OFF" if off[n] else "DEF")))))
            labels.append((n, role, bool(off[n])))
        if not labels:
            return
        bar_ns  = [l[0] for l in labels]
        bar_col = ["#FFA500" if l[2] else "#0000FF" for l in labels]
        bar_lbl = [f"{l[1]}{'★' if is_qb[l[0]] else ''} {l[0]+1}" for l in labels]
        ax_sep.barh(range(len(bar_ns)), [sep[n] for n in bar_ns],
                    color=bar_col, alpha=0.8)
        ax_sep.set_yticks(range(len(bar_ns)))
        ax_sep.set_yticklabels(bar_lbl, fontsize=7)
        ax_sep.set_xlabel("player_sep (normalised)")
        ax_sep.set_title("Per-player separation", fontsize=9)
        ax_sep.axvline(0, color="#000000", lw=0.5)

    # ── initial draw ───────────────────────────────────────────────────────────
    _refresh_sep(init_idx)

    # ── slider ─────────────────────────────────────────────────────────────────
    s_ax  = fig.add_axes([0.155, 0.040, 0.720, 0.030])
    inc_r = inc if 1 <= inc <= max(max_t - 3, 1) else 1
    slider = Slider(ax=s_ax, label="FRAME", valmin=0,
                    valmax=max(max_t - 1, 1),
                    valinit=init_idx, valstep=inc_r)

    def _update(val):
        idx = max(0, min(int(val), max_t - 1))
        _refresh_sep(idx)
        cursor_vline.set_xdata([t[idx], t[idx]])
        fig.canvas.draw_idle()

    slider.on_changed(_update)
    return fig, slider
