
from matplotlib.axes import Axes
from matplotlib.collections import PathCollection
from matplotlib.lines import Line2D
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as np
import seaborn as sns
import torch

from common.constants import fps, screen_mode, team_players
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
        linewidth=0.1, edgecolor=clr, facecolor="#F2F2F2", zorder=0,
    )
    ax.add_patch(canvas)

    for line in range(x_field_min, x_field_max, 10):
        ax.plot(
            [line, line], [-np.abs(y_bnd), +np.abs(y_bnd)],
            ls="-", color=clr, zorder=1, alpha=0.5,
        )

    ax.plot(
        [x_field_min, x_field_max], [-np.abs(y_bnd) / 2, -np.abs(y_bnd) / 2],
        ls="-.", color=clr, zorder=1, alpha=0.5,
    )
    ax.plot(
        [x_field_min, x_field_max], [+np.abs(y_bnd) / 2, +np.abs(y_bnd) / 2],
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

    vline = ax.axvline(x=sx, color="green", linestyle="--", zorder=8)
    sc1 = ax.scatter(x1, y1, s=30, c="blue", zorder=9)
    sc2 = ax.scatter(x2, y2, s=30, c="orange", zorder=9)

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
    max_t: int, inc: int=1
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

    vliner = axs[0].axvline(x=sxr, color="green", linestyle="--", zorder=8)
    sc1r = axs[0].scatter(x1r, y1r, s=30, c="blue", zorder=9)
    sc2r = axs[0].scatter(x2r, y2r, s=30, c="orange", zorder=9)

    vlineg = axs[1].axvline(x=sxg, color="green", linestyle="--", zorder=8)
    sc1g = axs[1].scatter(x1g, y1g, s=30, c="blue", zorder=9)
    sc2g = axs[1].scatter(x2g, y2g, s=30, c="orange", zorder=9)

    def update(i):
        ii = int(i)
        iii = 1 if ii <= 0 else (max_t - 2 if ii >= max_t - 1 else ii)

        x1r, x2r = xs0r[iii], xs1r[iii]
        y1r, y2r = ys0r[iii], ys1r[iii]
        sc1r.set_offsets(np.c_[x1r, y1r])
        sc2r.set_offsets(np.c_[x2r, y2r])

        x1g, x2g = xs0g[iii], xs1g[iii]
        y1g, y2g = ys0g[iii], ys1g[iii]
        sc1g.set_offsets(np.c_[x1g, y1g])
        sc2g.set_offsets(np.c_[x2g, y2g])

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
