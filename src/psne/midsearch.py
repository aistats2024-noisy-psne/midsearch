import jax.numpy as jnp
import jax.random
from jax import device_get
from jax.typing import ArrayLike
import numpy as np
from numpy.typing import NDArray

from psne import NashReturn


def find_nash(a: NDArray, rng: np.random.Generator) -> NDArray:
    """
    Finds the Nash equilibrium in a 2x2 matrix. Defaults to random if no Nash exists.
    :param a: Input array.
    :param rng: Numpy random number generator.
    :return: Indices of the Nash equilibrium.
    """
    if a[1, 0] < a[0, 0] < a[0, 1]:
        return np.array([0, 0])
    elif a[0, 0] < a[1, 0] < a[1, 1]:
        return np.array([1, 0])
    elif a[1, 1] < a[0, 1] < a[0, 0]:
        return np.array([0, 1])
    elif a[0, 1] < a[1, 1] < a[1, 0]:
        return np.array([1, 1])
    else:
        return rng.integers(0, 2, size=2)


def find_p(t: int) -> int:
    """Finds largest integer p such that p*int(log4(p)) < t."""
    p = 1
    while p * int(np.log(p) / np.log(4)) < t:
        p *= 4
    p /= 4
    while p * int(np.log(p) / np.log(4)) < t:
        p += 1
    p -= 1
    return int(p)


def infinite_successive_halving(
    rng: np.random.Generator, mu: NDArray, t_max: int
) -> float:
    """
    Implements infinite successive halving algorithm for multi-armed bandits.
    :param rng: Random number generator.
    :param mu: Arm means.
    :param t_max: Total budget.
    :return: Estimated maximum arm mean.
    """
    n = len(mu)
    n_rounds = int(np.log(n) / np.log(4))
    active_idx = np.arange(n)
    # If we don't have the budget, remove some arms at random
    if t_max < (n * n_rounds):
        p = find_p(t_max)
        n_rounds = int(np.log(p) / np.log(4))
        active_idx = rng.choice(active_idx, p, replace=False)
    n_active = len(active_idx)
    if n_rounds == 0:
        # we don't have enough for a full round:
        # choose the first active_index and return its empirical mean
        return np.mean(rng.uniform(size=t_max) < mu[active_idx[0]])
    arm_round_budget = int(t_max / (n_rounds * n_active))
    mu_hats = np.zeros(n_active)
    t = 0
    for _ in range(n_rounds):
        n_active = len(active_idx)
        if n_active <= 3:  # exhaust budget if last round
            arm_round_budget = int((t_max - t) / len(active_idx))
        mu_hats = np.mean(
            rng.uniform(size=(arm_round_budget, n_active)) < mu[active_idx], axis=0
        )
        t += n_active * arm_round_budget
        # Keep the second quartile (by sample sum) of arms
        low = n_active // 4
        high = max((2 * n_active // 4) - 1, 1)
        arm_ixs = np.argsort(-mu_hats)[low:high]
        active_idx = active_idx[arm_ixs]
        arm_round_budget *= 4
        if len(active_idx) == 1:
            break
    assert t <= t_max
    return np.max(mu_hats)


def _sample_small(
    rng: np.random.Generator,
    a: NDArray,
    active_rows: NDArray,
    active_cols: NDArray,
    n_samples: int,
) -> NDArray:
    """
    Sample a 2x2 matrix a set number of times and return the Nash equilibrium.
    :param rng: Random number generator.
    :param a: Full input array of means.
    :param active_rows: Active row indices (2).
    :param active_cols: Active column indices (2)
    :param n_samples: Number of times to sample each element.
    :return: Nash equilibrium.
    """
    assert len(active_rows) == 2 and len(active_cols) == 2
    a_small = a[active_rows][:, active_cols]
    mu = np.mean(rng.uniform(size=(n_samples, 2, 2)) < a_small, axis=0)
    ix, jx = find_nash(mu, rng)
    return np.array([active_rows[ix], active_cols[jx]], dtype=int)


def _midsearch(a: NDArray, t_max: int, random_seed: int) -> tuple[NDArray, int]:
    """
    Algorithm to find a PSNE from noisy samples.
    :param a: Input array of arm means.
    :param t_max: Total sampling budget.
    :param random_seed: Random seed.
    :return: Tuple of (Nash equilibrium, samples used).
    """
    n, m = a.shape
    assert n == m, "a must be square for now"
    assert np.log2(n) % 1 == 0, "n must be a power of 2 for now"
    active_rows = np.arange(n)
    active_cols = np.arange(n)
    n_active_rows, n_active_cols = n, n
    n_base = int(t_max / (2 * np.log2(n)))  # base budget used to derive other budgets
    rng = np.random.default_rng(random_seed)
    t = 0
    while n_active_rows > 1 or n_active_cols > 1:
        n_col_samples = int(n_base / (2 * n_active_cols))
        n_row_samples = int(n_base / (2 * n_active_rows))
        if n_active_rows == 2 and n_active_cols == 2:
            # there are only 2x2 arms left, so directly sample those a few times
            # with the rest of the budget and take the Nash
            n_small_samples = int((t_max - t) // 4)
            t += 4 * n_small_samples
            return _sample_small(rng, a, active_rows, active_cols, n_small_samples), t
        elif n_active_rows >= n_active_cols:
            # For each active column, sample entries in active rows
            col_mu_max = np.zeros(n_active_cols)
            for jx, col in enumerate(active_cols):
                col_mu_max[jx] = infinite_successive_halving(
                    rng, a[active_rows, col], n_col_samples
                )
            t += n_col_samples * n_active_cols
            # j_hat: column with the lowest maximum entry (approx.)
            j_hat = active_cols[np.argmin(col_mu_max)]
            # Sample some more in j_hat column
            j_hat_mu = np.mean(
                rng.uniform(size=(n_row_samples, n_active_rows))
                < a[active_rows, j_hat],
                axis=0,
            )
            t += n_row_samples * n_active_rows
            # Remove the bottom half (by sample mean) of rows in j_hat
            n_active_rows //= 2
            top_row_ixs = np.argsort(j_hat_mu)[n_active_rows:]
            active_rows = active_rows[top_row_ixs]
        elif n_active_rows < n_active_cols:
            # For each active row, sample entries in active cols
            # row_mu_min = 1 - max(1 - row) = 1 - (1 - min(row)) = min(row)
            row_mu_min = np.zeros(n_active_rows)
            for ix, row in enumerate(active_rows):
                row_mu_min[ix] = 1 - infinite_successive_halving(
                    rng, 1 - a[row, active_cols], n_row_samples
                )
            t += n_row_samples * n_active_rows
            # i_hat: row with the highest minimum entry (approx.)
            i_hat = active_rows[np.argmax(row_mu_min)]
            # Sample some more in i_hat row
            i_hat_mu = np.mean(
                rng.uniform(size=(n_col_samples, n_active_cols))
                < a[i_hat, active_cols],
                axis=0,
            )
            t += n_col_samples * n_active_cols
            # Remove the upper half (by sample mean) of cols in i_hat
            n_active_cols //= 2
            bottom_col_ixs = np.argsort(i_hat_mu)[:n_active_cols]
            active_cols = active_cols[bottom_col_ixs]
    return np.array([active_rows[0], active_cols[0]], dtype=int), t


def midsearch(
    a: ArrayLike, t_max: int, random_seed: int, n_checkpoints: int = 10
) -> NashReturn:
    """
    Helper method to align midsearch with other algorithms.
    :param a: Reward matrix.
    :param t_max: Total number of samples.
    :param random_seed: Random seed. Will be used to generate a Jax PRNGKey.
    :param n_checkpoints: Number of checkpoints to return.
    :return: Tuple of (Array of Nash indices, checkpoint indices, total number of samples).
    """
    ij_hats = jnp.zeros((n_checkpoints, 2), dtype=int)
    checkpoint_ixs = jnp.linspace(t_max // n_checkpoints, t_max, n_checkpoints)
    key = jax.random.PRNGKey(random_seed)
    np_seed = int(device_get(jax.random.randint(key, (), 0, 2**16)))
    a_np = device_get(a)
    # Just rerun the algorithm for each checkpoint
    for k, dt_max in enumerate(checkpoint_ixs):
        # Use the same NumPy seed every time
        ij_hat, _ = _midsearch(a_np, dt_max, np_seed)
        ij_hats = ij_hats.at[k].set(ij_hat)
    return ij_hats, checkpoint_ixs, t_max
