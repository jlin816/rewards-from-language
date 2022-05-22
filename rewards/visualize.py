from constants import REWARD_KEYS
import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate
from utils import get_carrier_from_list
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sns.set_style("white")

from rewards.constants import REWARD_KEYS

def reshape_posterior_samples(samples):
    marginals = []
    for s in samples:
        marginals.extend(
            [{"feature": REWARD_KEYS[i], "value": v} for i, v in enumerate(s)]
        )
    return marginals


def plot_posterior_for_paper(samples, true_reward, name, smoothing=0.2):
    means = np.mean(samples, axis=0)
    samples = pd.DataFrame(reshape_posterior_samples(samples))
    g = sns.FacetGrid(
        samples,
        row="feature",
        height=1.0,
        aspect=3.5,
        xlim=[-1.05, 1.05],
        ylim=[0, 1.2],
    )
    g.map(
        sns.kdeplot,
        "value",
        bw_adjust=0.75,
        clip=(-1.01, 1.01),
        fill=True,
        alpha=0.25,
    )
    for mean, true_val, ax in zip(means, true_reward, g.axes.ravel()):
        ax.vlines(
            true_val, *ax.get_ylim(), color="#DB5E57", linestyle="--", linewidth=2
        )

        ax.vlines(
            mean,
            *ax.get_ylim(),
            color="#1F78B4",
            linestyle="-",
            linewidth=2,
            alpha=0.55,
        )
        ax.spines["left"].set_visible(False)
    g.tight_layout()
    g.set_titles(row_template="")
    g.set_axis_labels("Value", "")
    g.set(yticks=[], xticks=[-1, -0.5, 0, 0.5, 1])
    plt.subplots_adjust(wspace=0, hspace=0.1)
    plt.savefig(name, dpi=300, pad_inches=0.0)


def plot_posterior(wt_samples, true_reward_wts=None, title="", save_to=None, vmax=200):
    """Plots posterior distribution feat-by-feat.

    Args:
        wt_samples (np.array): shape (num_samples, num_feats=8)
    """
    fig, axs = plt.subplots(nrows=4, ncols=2, sharey="row", figsize=(10, 10))
    fig.suptitle(title, fontsize=14)

    for i, ax in enumerate(axs.flatten()):
        feat_name = REWARD_KEYS[i]
        ax.hist(wt_samples[:, i], bins=25)
        mean_wt = wt_samples[:, i].mean()
        var_wt = wt_samples[:, i].var()
        ax.vlines(mean_wt, 0, int(vmax * 1.5), color="blue")
        ax.set_title(f"{feat_name}: mean {mean_wt:.2f} var {var_wt:.2f}")
        ax.set_ylim([0, vmax])
        ax.set_xlim([-1.1, 1.1])
        ax.set_xticks(np.arange(-1, 1))

        if true_reward_wts is not None:
            ax.vlines(true_reward_wts[i], 0, int(vmax * 1.5), color="red")

    fig.tight_layout()
    if save_to is not None:
        plt.savefig(save_to)
        plt.close()


def bootstrap_standard_error_of_mean(xs, n_samples=10000):
    samples = np.random.choice(xs, size=(len(xs), n_samples), replace=True)
    means = samples.mean(0)
    return means.std()


def plot_with_error_bars(
    results_by_model,
    xlabel,
    ylabel,
    title,
    save_to,
    error_bar_type="standard_deviation",
    show_plots=False,
):
    """
    Args:
        results_by_model: dict with keys `model_names` (different series on the plot), each a dict with keys {x1: [list of ys (plot mean and show std)], x2: [...], ...}
    """
    for model_name, model_results in results_by_model.items():
        xs = list(sorted(model_results.keys()))
        ys = [model_results[x] for x in xs]
        means_per_x = np.array([np.mean(ys_for_x) for ys_for_x in ys])
        stds_per_x = np.array([np.std(ys_for_x) for ys_for_x in ys])
        if error_bar_type == "standard_deviation":
            err_per_x = stds_per_x
        elif error_bar_type == "standard_error":
            ns_per_x = np.array([len(ys_for_x) for ys_for_x in ys])
            err_per_x = stds_per_x / np.sqrt(ns_per_x)
            print("ns_per_x: ", ns_per_x)
        elif error_bar_type == "standard_error_bootstrap":
            err_per_x = np.array(
                [bootstrap_standard_error_of_mean(ys_for_x) for ys_for_x in ys]
            )
        elif error_bar_type == "none":
            err_per_x = np.zeros_like(means_per_x)
        else:
            raise ValueError(f"invalid error_bar_type {error_bar_type}")

        plt.plot(xs, means_per_x, label=model_name)
        plt.fill_between(
            xs, means_per_x - err_per_x, means_per_x + err_per_x, alpha=0.5
        )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xlim([0, 7])
    plt.title(title)
    plt.legend()
    plt.savefig(save_to)
    if show_plots:
        plt.show()
    plt.close()


def print_flights(options, reward_weights):
    per_feat_rewards = np.multiply(options, reward_weights)
    rewards = list(np.sum(per_feat_rewards, axis=-1))

    reward_weights = list(reward_weights)
    options = [list(opt) for opt in options]
    carriers = [get_carrier_from_list(opt)[0] for opt in options]

    rows = [
        [key, reward_weights[i], options[0][i], options[1][i], options[2][i]]
        for i, key in enumerate(REWARD_KEYS)
    ]
    rows.append(["Total:", 0, rewards[0], rewards[1], rewards[2]])
    print(
        tabulate(
            rows,
            headers=[
                "",
                "r",
                f"o1\n{carriers[0]}",
                f"o2\n{carriers[1]}",
                f"o3\n{carriers[2]}",
            ],
            tablefmt="psql",
            floatfmt=".2f",
        )
    )
