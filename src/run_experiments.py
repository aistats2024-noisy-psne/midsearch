#!/usr/bin/env python

import os
import time

import numpy as np
import pandas as pd

from psne import n_workers, DATA_DIR
from psne.compare import compare
from psne.problems import hard_triangular_fn
from psne.utils import rng_seed
from psne.lucbg.lucbg import lucbg
from psne.exp3ix import exp3ix
from psne.tsallis_inf import tsallis_inf
from psne.midsearch import midsearch
from psne.uniform_psne import uniform_psne


def run(
    experiment_type: int, n_trials: int = 300
) -> tuple[tuple[pd.DataFrame, str], tuple[pd.DataFrame, str]]:
    """
    Run a single experiment and save the results to a file. Not all used in paper.

    :param experiment_type:
      1: All algorithms, small dims, delta_min=0.05, beta=0.1.
      2: MidSearch vs. Tsallis-INF, all dims, delta_min=0.1/(2^[0,4)), beta=0.1.
      3: MidSearch vs. Tsallis-INF, d=1024, delta_min=0.1/(2^[0,4)), beta=0.1.
      4: MidSearch vs. Tsallis-INF, d=1024, delta_min=0.1/(2^[0,7)), beta=0.1.
      5: MidSearch vs. Tsallis-INF, d=[128,256,512], delta_min=0.01, vary beta.
      6: MidSearch vs. Tsallis-INF, d=[128,256,512], vary delta_min and beta at the same time.
    :param n_trials: Number of trials to run. Defaults to 300.
    :return: Tuple of each (dataframe, filename) for the full results and summary.
    """
    assert (1 <= experiment_type <= 6) and isinstance(experiment_type, int)
    if n_trials is not None:
        print(f"Overriding with {n_trials=}.")
    else:
        n_trials = 300
    algorithms = [
        midsearch,
        tsallis_inf,
    ]
    dim, problems = None, None
    if experiment_type == 1:
        # All algorithms, small dims, one problem type
        algorithms += [
            lucbg,
            exp3ix,
            uniform_psne,
        ]
        dim = [64, 32, 16]
        problems = [hard_triangular_fn(delta_min=0.05, beta=0.1)]
    elif experiment_type == 2:
        # MidSearch vs. Tsallis-INF, all dims, multiple problems
        dim = [1024, 512, 256, 128, 64]
        problems = [
            hard_triangular_fn(delta_min=delta_min, beta=0.1)
            for delta_min in 0.1 / (2 ** np.arange(4))
        ]
    elif experiment_type == 3:
        # MidSearch vs. Tsallis-INF, d=1024, multiple problems
        dim = [1024]
        problems = [
            hard_triangular_fn(delta_min=delta_min, beta=0.1)
            for delta_min in 0.1 / (2 ** np.arange(4))
        ]
    elif experiment_type == 4:
        # MidSearch vs. Tsallis-INF, d=1024, delta_min=0.1/(2^[0,7)), beta=0.1.
        dim = [1024]
        problems = [
            hard_triangular_fn(delta_min=delta_min, beta=0.1)
            for delta_min in 0.1 / (2 ** np.arange(7))
        ]
    elif experiment_type == 5:
        # MidSearch vs. Tsallis-INF, d=[128,256,512], delta_min=0.01, vary beta.
        dim = [512, 256, 128]
        problems = [
            hard_triangular_fn(delta_min=0.01, beta=beta)
            for beta in (0.03 + 0.02 * np.arange(4))
        ]
    elif experiment_type == 6:
        # MidSearch vs. Tsallis-INF, d=[128,256,512], vary both delta_min and beta.
        dim = [512, 256, 128]
        problems = [
            hard_triangular_fn(delta_min=delta_min, beta=beta)
            for delta_min, beta in (
                (0.075, 0.1),
                (0.05, 0.075),
                (0.025, 0.05),
                (0.01, 0.025),
            )
        ]
    compare_kwargs = {
        "n_checkpoints": 10,
        "n_matrices": 1,
        "n_trials": n_trials,
    }
    print(
        f"Running {experiment_type=} on {n_workers} cores: "
        f"{len(dim)} dims, {len(algorithms)} algorithms, "
        f"{len(problems)} problems, {n_trials} trials."
    )
    # Checkpoints 5-50H1 in ticks of 5
    df, summary_df = compare(
        algorithms,
        dim=dim,
        problems=problems,
        base_random_seed=rng_seed,
        **compare_kwargs,
    )
    if experiment_type in [2, 3, 4, 5, 6]:
        # Adding 1-5 H1 points for better resolution on some plots
        df_small_h1, _ = compare(
            algorithms,
            dim=dim,
            problems=problems,
            base_random_seed=rng_seed,
            t_max_fn=lambda h1: int(5 * h1),
            n_checkpoints=5,
            n_matrices=1,
            n_trials=n_trials,
        )
        df_small_h1["checkpoint_ix"] = -1
        df_full = pd.concat((df_small_h1, df)).reset_index(drop=True)
    else:
        df_full = df
    # Assumes we are running this file from its location
    timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
    filename = os.path.join(DATA_DIR, f"experiment_{experiment_type}_{timestr}.csv")
    df_full.to_csv(filename, index=False)
    summary_filename = os.path.join(
        DATA_DIR, f"summary_experiment_{experiment_type}_{timestr}.csv"
    )
    summary_df.to_csv(summary_filename, index=False)
    print(summary_df)
    return (df_full, filename), (summary_df, summary_filename)


if __name__ == "__main__":
    for experiment_type_ in [
        1,
        # 2,
        # 3,
        # 4,
        # 5,
        # 6,
    ]:
        run(experiment_type_, n_trials=10)
