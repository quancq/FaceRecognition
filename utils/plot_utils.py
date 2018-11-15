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

    ax.set_xlim(x[0]-1, x[-1]+1)

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


def plot_histogram(values, num_bins=10, save_path=DEFAULT_PLOT_SAVE_PATH, title="", xlabel="", ylabel="", figsize=(8,6), **kwargs):
    fig, ax = plt.subplots(figsize=figsize)

    ax.hist(values, bins=num_bins, **kwargs)
    ax.axhline(np.array(values).mean(), color="grey")
    # title = kwargs.get("title")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)

    # ax.legend()

    plt.savefig(save_path)
    plt.close(fig)

    print("Save figure to {} done".format(save_path))
