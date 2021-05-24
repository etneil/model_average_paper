#!/usr/bin/env python3

import matplotlib.pyplot as plt
import gvar as gv
import numpy as np
import seaborn as sns

from IPython.display import set_matplotlib_formats

set_matplotlib_formats("png", "pdf")

# Settings for "publication-ready" figures
color_palette = sns.color_palette("deep")
sns.set_palette(color_palette)
sns.palplot(color_palette)

sns.set(style="white")
sns.set_style("ticks")
sns.set_style({"xtick.direction": "in", "ytick.direction": "in"})

sns.set_context(
    "paper", font_scale=2.0, rc={"lines.linewidth": 2.5, "figure.figsize": (7, 5)}
)


def plot_gvcorr(
    gc,
    color="blue",
    log_scale=False,
    offset=0.0,
    x=None,
    xr_offset=True,
    label=None,
    marker="o",
    markersize=6,
    capthick=2,
    capsize=4,
    open_symbol=False,
    fill=False,
    linestyle=" ",
    eb_linestyle=" ",
):
    if x is None:
        x = np.arange(0, len(gc))

    if fill:
        y = np.asarray([gv.mean(g) for g in gc])
        yerr = np.asarray([gv.sdev(g) for g in gc])
        eplot = plt.plot(x + offset, y, color=color, label=label)
        eplot = plt.fill_between(
            x + offset, y - yerr, y + yerr, alpha=0.3, edgecolor="k", facecolor=color
        )
    else:
        if open_symbol:
            eplot = plt.errorbar(
                x=x + offset,
                y=[gv.mean(g) for g in gc],
                yerr=[gv.sdev(g) for g in gc],
                marker=marker,
                markersize=markersize,
                capthick=capthick,
                capsize=capsize,
                linestyle=linestyle,
                color=color,
                label=label,
                mfc="None",
                mec=color,
                mew=1,
            )
        else:
            eplot = plt.errorbar(
                x=x + offset,
                y=[gv.mean(g) for g in gc],
                yerr=[gv.sdev(g) for g in gc],
                marker=marker,
                markersize=markersize,
                capthick=capthick,
                capsize=capsize,
                linestyle=linestyle,
                color=color,
                label=label,
            )

        if eb_linestyle != " ":
            eplot[-1][0].set_linestyle(eb_linestyle)

    if log_scale:
        plt.yscale("symlog")

    if xr_offset:
        plt.xlim(x[0] - 0.2, x[-1] + 0.2)

    return eplot
