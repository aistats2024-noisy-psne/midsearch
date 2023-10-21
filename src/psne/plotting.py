import re
import warnings
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from scipy.stats import binomtest

# Suppresses seaborn catplot errors
warnings.simplefilter(action="ignore", category=FutureWarning)


def _prop_ci(x) -> tuple:
    """95% Wilson confidence interval for proportion of successes in x."""
    if (len(x) == 1) and (x[0] == 0):
        return 0, 0
    else:
        res = binomtest(x.sum(), len(x)).proportion_ci(
            confidence_level=0.95, method="wilson"
        )
        return res.low, res.high


def _algorithm_latex_name(alg_name: str) -> str:
    """Return LaTeX string for algorithm name."""
    if alg_name == "tsallis_inf":
        return "\\textsc{Tsallis-INF}"
    elif alg_name == "inf_algo" or alg_name == "midsearch":
        return "\\textsc{MidSearch}"
    elif alg_name == "lucbg":
        return "\\textsc{LUCB-G}"
    elif alg_name == "uniform_psne":
        return "\\textsc{Uniform}"
    elif alg_name == "exp3ix":
        return "\\textsc{Exp3-IX}"
    else:
        raise ValueError(f"Unknown algorithm name: {alg_name}")


def _latex_problem_name(problem_name: str, delta_only: bool = False):
    """Return LaTeX string for problem name."""
    pattern = "\[(.+),(.+)\]"
    if delta_only:
        return "$\\Delta_{{\\min}}=%s$" % re.search(pattern, problem_name).group(1)
    else:
        return (
            f"$\\Delta_{{\\min}}=%s, \\beta=%s$"
            % re.search(pattern, problem_name).groups()
        )


def plot_h1_sweep(df: pd.DataFrame, filename: str, exp_type: int = 1):
    """
    Plot results of H1 sweep experiment.
    :param df: DataFrame of results.
    :param filename: Filename to save plot to.
    :param exp_type:
      1: All algorithms, small dims, delta_min=0.05, beta=0.1.
      2: MidSearch vs. Tsallis-INF, all dims, delta_min=0.1/(2^[0,4)), beta=0.1.
      3: MidSearch vs. Tsallis-INF, d=1024, delta_min=0.1/(2^[0,4)), beta=0.1.
      4: MidSearch vs. Tsallis-INF, d=1024, delta_min=0.1/(2^[0,7)), beta=0.1.
      5: MidSearch vs. Tsallis-INF, d=[128,256,512], delta_min=0.01, vary beta.
      6: MidSearch vs. Tsallis-INF, d=[128,256,512], vary delta_min and beta at the same time.
    """
    assert 1 <= exp_type <= 6, f"Unknown experiment type: {exp_type}"
    base_theme_kwargs = {
        "style": "whitegrid",
        "context": "paper",
        "font": "serif",
    }
    base_rcparams = {
        "text.usetex": True,
        "axes.labelsize": 20.0,
        "xtick.labelsize": 18.0,
        "ytick.labelsize": 18.0,
        "axes.titlesize": 20.0,
        "legend.fontsize": 18.0,
        "xtick.bottom": True,
        "xtick.color": ".8",
    }
    # Plotting parameters based on exp_type
    assert 1 <= exp_type <= 6
    one_col_plot = exp_type in [1, 3, 4]
    plot_small_h1 = exp_type != 1
    if exp_type in [3, 4]:
        col_param = "dim"
        ls_param = "problem"
    else:
        col_param = "problem"
        ls_param = "dim"

    if one_col_plot:
        sns.set_theme(
            **base_theme_kwargs,
            font_scale=3,
            rc={
                **base_rcparams,
                "lines.linewidth": 3,
                "legend.fontsize": 16.5,
            },
        )
    else:
        sns.set_theme(
            **base_theme_kwargs,
            font_scale=4,
            rc={
                **base_rcparams,
                "lines.linewidth": 2.25,
            },
        )

    # Round x values to make plots nice (annoying Jax stuff)
    xvals = df["checkpoint"] / df["h1"]
    if plot_small_h1:
        df["samples/H1"] = np.round(xvals)
    else:
        df["samples/H1"] = np.round(xvals / 5) * 5
    df_ = df.rename(columns={"name": "algorithm"})
    df_["algorithm"] = df_["algorithm"].map(_algorithm_latex_name)
    n_algorithms = df_.algorithm.nunique()
    df_["problem"] = df_["problem"].map(
        lambda x: _latex_problem_name(x, delta_only=col_param == "dim")
    )

    # Add point at (0,0) for each line
    zero_df = (
        df_.groupby(["dim", "problem", "algorithm"])[["correct", "samples/H1"]]
        .agg("first")
        .reset_index()
    )
    zero_df["correct"] = 0
    zero_df["samples/H1"] = 0
    df_ = pd.concat((df_, zero_df)).reset_index(drop=True)

    g = sns.FacetGrid(
        df_,
        col=col_param,
        col_wrap=1 if one_col_plot else 2,
        height=4.75,
        sharex=False,
        aspect=1.8 if one_col_plot else 1.6,
        despine=True,
        legend_out=one_col_plot,
        ylim=[-0.1, 1.1],
        xlim=[-1, 51],
    )
    g.map_dataframe(
        sns.lineplot,
        x="samples/H1",
        y="correct",
        hue="algorithm",
        style=ls_param,
        palette=sns.color_palette("colorblind")[:n_algorithms],
        errorbar=_prop_ci,
        ms=10,
        marker="o",
    )
    g.set(xticks=range(0, 55, 5))
    g.set_xticklabels(color=".15")
    if plot_small_h1:
        for ax in g.axes[:]:
            ax.set_xticks(range(1, 5), minor=True)

    if col_param == "problem":
        title_template = "{col_name}"
    else:
        title_template = "$d$={col_name}, $\\beta$=0.1"
    g.set_titles(template=title_template, pad=2)

    if one_col_plot:
        g.add_legend()
        if ls_param == "dim":
            g.legend.get_texts()[n_algorithms + 1].set_text("$d$")
    else:
        ax = g.axes[0]
        handles, labels = ax.get_legend_handles_labels()
        bbox_to_anchor = (0.775, 0.55) if exp_type in [5, 6] else (0.75, 0.8)
        legend_dim = ax.legend(
            handles[n_algorithms + 1 :],
            labels[n_algorithms + 1 :],
            framealpha=1,
            loc="upper left",
            bbox_to_anchor=bbox_to_anchor,
        )
        legend_dim.get_texts()[0].set_text("$d$")

        legend_alg = ax.legend(
            handles[: n_algorithms + 1],
            labels[: n_algorithms + 1],
            framealpha=1,
            loc="upper right",
            bbox_to_anchor=bbox_to_anchor,
        )
        ax.add_artist(legend_dim)
        g._legend = legend_alg

    g.set_xlabels("samples/$H_1$")
    g.set_ylabels(
        "$P$(correct)",
        loc="top",
        rotation="horizontal",
    )
    for ax in g.axes.flat:
        ax.yaxis.set_major_formatter(tkr.PercentFormatter(xmax=1))
        ax.get_yaxis().set_label_coords(0.05, 1.01)

    g.tight_layout()
    plt.savefig(filename, bbox_inches="tight")
