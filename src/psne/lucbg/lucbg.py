import jax.numpy as jnp
from jax import random, lax, Array
from jax.typing import ArrayLike

from psne import NashReturn
from psne.lucbg.stage1 import pure_nash, select_tasks
from psne.lucbg.stage2 import pull_arms
from psne.lucbg.stage3 import validate_row_mins


def _jax_loop(vals):
    (
        key,
        t,
        (a, one_m_at),
        mu,
        active_rc,
        selected_rc,
        pulls,
        ucb,
        finished,
    ) = vals

    # Stage 1: select tasks (row/column pair) to use in Stage 2
    ij_hat = select_tasks(mu, active_rc, selected_rc)
    i_hat, j_hat = ij_hat
    key, subkey_r, subkey_c = random.split(key, 3)

    # Stage 2: pull three arms in the selected row or column
    mu, pulls["row"], ucb["row"] = pull_arms(
        i_hat,
        subkey_r,
        a,
        mu,
        pulls["row"],
        pulls["col_t"],
        ucb["row"],
        selected_rc[:, 0],  # tau vector
    )

    # 2b. Update columns
    one_m_mu_t, pulls["col_t"], ucb["col_t"] = pull_arms(
        j_hat,
        subkey_c,
        one_m_at,
        (1 - mu.T),
        pulls["col_t"],
        pulls["row"],
        ucb["col_t"],
        selected_rc[:, 1],
    )
    mu = 1 - one_m_mu_t.T
    t += jnp.sum(3 * (ij_hat > -1))

    # Stage 3. UCB step - check to see if we can eliminate any rows/columns from being active
    row_match, row_match_ix = validate_row_mins(
        i_hat,
        mu,
        ucb["row"],
    )
    col_match, col_match_ix = validate_row_mins(
        j_hat,
        one_m_mu_t,
        ucb["col_t"],
    )
    active_rc = active_rc.at[i_hat, 0].set(active_rc[i_hat, 0] & ~row_match)
    active_rc = active_rc.at[j_hat, 1].set(active_rc[j_hat, 1] & ~col_match)

    # Only now increment selected_rc if ij_hat is not -1
    selected_rc = selected_rc.at[ij_hat, (0, 1)].add(ij_hat != -1)
    finished = ~jnp.any(active_rc)

    vals = (
        key,
        t,
        (a, one_m_at),
        mu,
        active_rc,
        selected_rc,
        pulls,
        ucb,
        finished,
    )
    return vals


def _lucbg(a: ArrayLike, t_max: int, random_seed: int) -> tuple[Array, int]:
    """
    Helper function for lucbg.
    :param a: Input reward matrix.
    :param t_max: Maximum number of samples to take.
    :param random_seed: Random seed.
    :return: Tuple of (Nash equilibrium, number of samples taken).
    """
    n, m = a.shape
    assert n == m, "Only square matrices supported for now."
    active_rc = jnp.ones((n, 2), dtype=bool)  # active rows/cols

    # Initialize arm means with n^2 uniform draws
    key = random.PRNGKey(random_seed)
    key, subkey = random.split(key)
    mu = 1.0 * (random.uniform(subkey, shape=(n, n)) < a)
    t = n**2
    # selected = number of times each task has been selected (tau/3)
    # initialize to n (sharing init pulls for row/col tasks)
    selected_rc = n * jnp.ones((n, 2))
    # initialize to one each (sharing init pulls for row/col tasks)
    pulls = {
        "row": jnp.ones((n, n)),
        "col_t": jnp.ones((n, n)),  # col transpose
    }
    # Start with large ucb bound as init
    ucb = {
        "row": jnp.ones((n, n)),
        "col_t": jnp.ones((n, n)),  # col transpose
    }
    init_vals = (
        key,
        t,
        (a, 1 - a.T),  # preallocate
        mu,
        active_rc,
        selected_rc,
        pulls,
        ucb,
        False,
    )
    vals = lax.while_loop(
        lambda val: (val[1] < t_max) & (~val[-1]),  # val[1] = t, val[-1] = finished
        _jax_loop,
        init_vals,
    )
    t, mu = vals[1], vals[3]
    return pure_nash(mu), t


def lucbg(
    a: ArrayLike, t_max: int, random_seed: int, n_checkpoints: int = 10
) -> NashReturn:
    """
    LUCB-G algorithm. See https://proceedings.mlr.press/v70/zhou17b/zhou17b.pdf for more details.

    :param a: Input reward matrix.
    :param t_max: Maximum number of samples to take.
    :param random_seed: Random seed.
    :param n_checkpoints: Number of checkpoints to use. Will try to evenly space.
    :return: Tuple of (Nash equilibria, corresponding sample indices, total samples taken).
    """
    ij_hats = jnp.zeros((n_checkpoints, 2), dtype=int)
    dt_maxes = jnp.linspace(t_max // n_checkpoints, t_max, n_checkpoints)
    # These can vary a bit with LUCB-G, so return actual number of samples
    checkpoint_ixs = jnp.zeros(n_checkpoints, dtype=int)
    for k, dt_max in enumerate(dt_maxes):
        ij_hat, n_samples = _lucbg(a, dt_max, random_seed)
        ij_hats = ij_hats.at[k].set(ij_hat)
        checkpoint_ixs = checkpoint_ixs.at[k].set(n_samples)
    return ij_hats, checkpoint_ixs, t_max
