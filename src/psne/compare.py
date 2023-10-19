import os
import time
from datetime import timedelta
from typing import Callable, Optional
from tqdm.autonotebook import tqdm, trange

from jax import numpy as jnp
from jax import Array, pmap, device_get, block_until_ready
from jax.typing import ArrayLike
import numpy as np
import pandas as pd

from psne import NashAlgorithm, ProblemGenerator, n_workers, TMP_DATA_DIR
from psne.problems import gap_h1
from psne.utils import shuffle_rows_cols


def _default_t_max_fn(gap: float) -> int:
    """Default function to compute t_max from the H1 gap."""
    return int(50 * gap)


def _uses_jax(algorithm: NashAlgorithm) -> bool:
    """Returns True if the algorithm is written in Jax."""
    return not (
        algorithm.__name__.startswith("inf_algo")
        or algorithm.__name__.startswith("midsearch")
    )


def _make_pmap(algorithm: NashAlgorithm) -> Callable:
    """
    Returns a jax.pmap function for the given algorithm, to be used to map across random seeds.
    Fakes it if the algorithm isn't written in Jax
    :param algorithm: Callable algorithm function.
    :return: Callable parallelized function.
    """
    # Map random seeds across cores, if possible
    # Outputs are mapped across leading dimension
    if not _uses_jax(algorithm):

        def _fake_pmap_alg(
            a: ArrayLike, t_max: int, random_seeds: ArrayLike, n_checkpoints_: int
        ) -> tuple[Array, Array, Array]:
            ij_hats, all_checkpoint_ixs, all_n_samples = [], [], []
            for seed in random_seeds:
                ij_hat, checkpoint_ixs, n_samples = algorithm(
                    a, t_max, seed, n_checkpoints_
                )
                ij_hats.append(ij_hat)
                all_checkpoint_ixs.append(checkpoint_ixs)
                all_n_samples.append(n_samples)
            return (
                jnp.array(ij_hats),
                jnp.array(all_checkpoint_ixs),
                jnp.array(all_n_samples),
            )

        return _fake_pmap_alg
    else:
        # Need to hold t_max and n_checkpoints as static arguments,
        # as lax.scan loop length + other functions depend on them
        # args: a, t_max (static), random_seeds, n_checkpoints (static)
        return pmap(
            algorithm,
            in_axes=(None, None, 0, None),
            static_broadcasted_argnums=(1, 3),
        )


def compare(
    algorithms: list[NashAlgorithm],
    dim: int | list[int],
    problems: list[
        ProblemGenerator
        | tuple[
            ProblemGenerator,
            Optional[int],
            Optional[int],
        ]
    ],
    base_random_seed: int,
    n_checkpoints: int = 10,
    n_matrices: Optional[int] = 1,
    n_trials: Optional[int] = 10,
    t_max_fn: Optional[Callable[[float], int]] = None,
    shuffle: bool = True,
    matrix_seed_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compares multiple algorithms across dimensions, problems, and random instances.

    :param algorithms: List of algorithm callables.
    :param dim: Dimension (or list of dimensions) of matrices to test.
    :param problems: List of problem generators functions to test.
      Each element of the list can also be a tuple of (fn, n_matrices, n_trials); otherwise (or if some are None), the
      passed values of n_matrices and n_trials are used.
    :param n_checkpoints: Number of checkpoints to return for each algorithm.
      Note: slows down algorithms that are not written in Jax.
    :param n_matrices: Number of different matrices to test for each problem.
    :param n_trials: Number of trials to run for each matrix.
    :param t_max_fn: Function to compute the maximum number of iterations to run for each matrix.
      This function takes in the 'H1' gap parameter and returns an int.
    :param base_random_seed: Base random seed for both problem generation and algorithm randomness.
      Different random seeds to be passed will be generated using this as an offset.
    :param shuffle: If True, shuffle the rows and columns of generated matrices so the Nash isn't always (0, 0).
      Highly recommended, as bugs/bad algorithms can lead to false correct guesses of (0, 0).
    :param matrix_seed_offset: Offset to use for the matrix random seed.
     Useful to finish stopped experiments in a consistent way.
    :return: Tuple of results DataFrame and results summary DatatFrame.
    """
    if isinstance(dim, int):
        dim = [dim]
    if t_max_fn is None:
        t_max_fn = _default_t_max_fn
    # Unpack problem data into a list of tuples
    problems_list = []
    for problem in problems:
        if isinstance(problem, tuple):
            assert len(problem) == 3, "Problem tuple must have 3 elements."
            prob_fn, prob_n_matrices, prob_n_trials = problem
            problem_tuple = (
                prob_fn,
                prob_n_matrices or n_matrices,
                prob_n_trials or n_trials,
            )
        else:
            problem_tuple = (problem, n_matrices, n_trials)
        assert (
            problem_tuple[1] and problem_tuple[2]
        ), "Must specify n_matrices and/or n_trials if None used in problem tuple."
        problems_list.append(problem_tuple)
    # See README.md for how to change number of cores
    print(f"Using {n_workers} cores for Jax pmap parallelization.")

    # Experiment hierarchy: dimension -> problem -> matrix -> algorithm -> trials
    results = []
    tqdm_cols = 80
    for d in tqdm(
        dim,
        position=0,
        desc="dims",
        leave=False,
        ncols=tqdm_cols,
    ):
        for problem_fn, prob_n_matrices, prob_n_trials in tqdm(
            problems_list, leave=False, position=1, desc="problems", ncols=tqdm_cols
        ):
            for m_ix in trange(
                prob_n_matrices,
                leave=False,
                position=1,
                desc="matrices",
                ncols=tqdm_cols,
            ):
                # Make sure we use the same matrix for all algorithms
                # (but should change across problem type and dim so matrices aren't
                # submatrices of each other due to how they are generated)
                matrix_seed_offset += 1
                matrix_seed = base_random_seed + matrix_seed_offset
                a_prob = problem_fn(d, matrix_seed)
                ij_star = (0, 0)
                h1 = gap_h1(a_prob, ij_star)
                if shuffle:
                    a_prob, ij_star = shuffle_rows_cols(a_prob, 10 * matrix_seed)
                a_prob = jnp.array(a_prob)

                for algorithm in tqdm(
                    algorithms,
                    position=2,
                    desc="algorithms",
                    leave=False,
                    ncols=tqdm_cols,
                ):
                    pmap_alg = _make_pmap(algorithm)
                    trial_bar = tqdm(
                        total=n_trials,
                        position=3,
                        desc="trials",
                        leave=False,
                        ncols=tqdm_cols,
                    )
                    for t_ix in range(0, n_trials, n_workers):
                        # Stepping by n_workers, save for possibly the last iteration
                        trial_ixs = np.arange(t_ix, min(t_ix + n_workers, n_trials))
                        # Each algorithm should get passed the same random seed for the same trial
                        algorithm_seeds = jnp.array(
                            matrix_seed + int(1e7) + trial_ixs, dtype=int
                        )
                        n_seeds = len(algorithm_seeds)
                        t_max = t_max_fn(h1)
                        # For timing, we need to use block_until_ready to make sure
                        t0 = time.time()
                        # all_ij_hats, all_checkpoint_ixs, all_n_samples
                        alg_results = pmap_alg(
                            a_prob, t_max, algorithm_seeds, n_checkpoints
                        )
                        # Make sure we are doing fair timing
                        if _uses_jax(algorithm):
                            block_until_ready(alg_results)
                        total_time = time.time() - t0
                        average_time = total_time / n_seeds
                        results.extend(
                            [
                                {
                                    "name": algorithm.__name__,
                                    "dim": d,
                                    "problem": problem_fn.__name__,
                                    "matrix": m_ix,
                                    "trial": trial_ix,
                                    "checkpoint": int(checkpoint_ix),
                                    "checkpoint_ix": ckpt_ix,
                                    "correct": ij_star == tuple(ij_hat),
                                    "time": average_time,
                                    "h1": h1,
                                    "t_max": t_max,
                                    "total_samples": n_samples,
                                    "matrix_seed": matrix_seed,
                                    "random_seed": int(alg_seed),
                                }
                                for trial_ix, ij_hats, checkpoint_ixs, n_samples, alg_seed in zip(
                                    trial_ixs,
                                    *alg_results,
                                    device_get(algorithm_seeds),
                                )
                                for ckpt_ix, (ij_hat, checkpoint_ix) in enumerate(
                                    zip(ij_hats, checkpoint_ixs)
                                )
                            ]
                        )
                        trial_bar.update(n_seeds)
                    trial_bar.close()
            # Save some temporary results to disk
            df = pd.DataFrame(results)
            timestr = time.strftime(f"%Y-%m-%d-%H-%M-%S")
            filename = f"dim{d}_{problem_fn.__name__}_{timestr}.csv"
            df.to_csv(os.path.join(TMP_DATA_DIR, filename), index=False)
    # Also return a summary DataFrame for convenience
    df = pd.DataFrame(results)
    # checkpoint divided by h1 should be close to the equality
    ideal_checkpoints = (df["checkpoint_ix"] + 1) * df["t_max"] / n_checkpoints
    df["checkpoint error"] = (
        np.abs(df["checkpoint"] - ideal_checkpoints) / ideal_checkpoints
    )
    gb_columns = ["dim", "problem", "name"]
    summary_df = df.groupby(gb_columns).agg(
        time=(
            "time",
            "median",
        ),  # checkpoints get all the same time, doesn't matter
        matrices=("matrix", "nunique"),
        trials=("trial", "nunique"),
        max_checkpoint_error=("checkpoint error", "max"),
    )
    # Only take the last checkpoint for total time
    summary_df["total time"] = (
        df.loc[df.checkpoint_ix == n_checkpoints - 1, :]
        .groupby(gb_columns)["time"]
        .sum()
    )
    summary_df["correct"] = (
        df.loc[df.checkpoint_ix == n_checkpoints - 1, :]
        .groupby(gb_columns)["correct"]
        .mean()
    )
    summary_df["median time(ms)"] = summary_df["time"].map(lambda t: f"{t * 1000:.1f}")
    summary_df["total time(s)"] = summary_df["total time"].map(lambda t: f"{t * 1:.1f}")
    print(
        "\n\n".join(
            [
                summary_df[
                    [
                        "matrices",
                        "trials",
                        "correct",
                        "median time(ms)",
                        "total time(s)",
                    ]
                ].to_string(),
                "Runtime by algorithm:\n"
                + pd.to_timedelta(
                    summary_df.groupby("name")["total time"].sum(), "s"
                ).to_string(name=False, dtype=False),
                "Runtime by dimension:\n"
                + pd.to_timedelta(
                    summary_df.groupby("dim")["total time"].sum(), "s"
                ).to_string(name=False, dtype=False),
                f"Total runtime: {timedelta(seconds=summary_df['total time'].sum())}.",
            ]
        )
    )
    return df, summary_df
