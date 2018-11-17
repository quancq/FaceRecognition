import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from settings import DEFAULT_PLOT_SAVE_PATH

sns.set()


def plot_simple_fig(y, x=None, save_path=DEFAULT_PLOT_SAVE_PATH, **kwargs):
    fig, ax = plt.subplots()

    title = kwargs.pop("title", None)
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)
    xticklabels = kwargs.pop("xticklabels", None)
    yticklabels = kwargs.pop("yticklabels", None)

    if x is None:
        x = list(range(len(y)))
    ax.plot(x, y, **kwargs)

    # ax.set_xlim(x[0]-1, x[-1]+1)

    if title is not None:
        ax.set(title=title)
    if xlabel is not None:
        ax.set(xlabel=xlabel)
    if ylabel is not None:
        ax.set(ylabel=ylabel)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    # ax.legend()

    plt.savefig(save_path)
    plt.close(fig)

    print("Save figure to {} done".format(save_path))


def plot_bar(x, y, save_path=DEFAULT_PLOT_SAVE_PATH, **kwargs):
    figsize = kwargs.pop("figsize", None)
    fig, ax = plt.subplots(figsize=figsize)

    title = kwargs.pop("title", None)
    xlabel = kwargs.pop("xlabel", None)
    ylabel = kwargs.pop("ylabel", None)
    xticklabels = kwargs.pop("xticklabels", None)
    yticklabels = kwargs.pop("yticklabels", None)
    xlim = kwargs.pop("xlim", None)
    ylim = kwargs.pop("ylim", None)

    if kwargs.get("color") is None:
        kwargs.update({"color": "C2"})
    if kwargs.get("width") is None:
        kwargs.update({"width": 0.3})

    if x is None:
        x = list(range(len(y)))
    ax.bar(x, y, **kwargs)

    # ax.set_xlim(x[0]-1, x[-1]+1)

    if title is not None:
        ax.set(title=title)
    if xlabel is not None:
        ax.set(xlabel=xlabel)
    if ylabel is not None:
        ax.set(ylabel=ylabel)
    if xticklabels is not None:
        ax.set_xticklabels(xticklabels)
    if yticklabels is not None:
        ax.set_yticklabels(yticklabels)
    if xlim is not None:
        xlim_min, xlim_max = xlim[0], xlim[1]
        xlim_min -= 1
        xlim_max += 1
        ax.set_xlim([xlim_min, xlim_max])
    if ylim is not None:
        ylim_min, ylim_max = ylim[0], ylim[1]
        ylim_min = (ylim_min - 0.2) if ylim_min > 0.2 else 0
        ylim_max = (ylim_max + 1) if ylim_max > 1 else 1
        ax.set_ylim([ylim_min, ylim_max])

    # Show value of each column to see clearly
    x_offset = 0
    y_offset = 0.02
    for p in ax.patches:
        b = p.get_bbox()
        text_value = "{:.4f}".format(b.y1)
        ax.annotate(text_value, xy=(b.x0 + x_offset, b.y1 + y_offset))
    ax.tick_params(axis="x", rotation=15)
    # ax.legend()

    plt.savefig(save_path)
    plt.close(fig)

    print("Save figure to {} done".format(save_path))


def plot_histogram(values, num_bins=10, save_path=DEFAULT_PLOT_SAVE_PATH, title="", xlabel="", ylabel="", figsize=(8, 6), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(values, bins=num_bins, **kwargs)
    # ax.axhline(np.array(values).mean(), color="grey")
    # title = kwargs.get("title")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    # ax.legend()

    plt.savefig(save_path)
    plt.close(fig)

    print("Save figure to {} done".format(save_path))
