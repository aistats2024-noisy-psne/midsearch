import jax.numpy as jnp
from jax import random, lax, Array
from jax.typing import ArrayLike

from psne import NashReturn
from psne.sampling import choice
from psne.utils import most_frequent_arms


def _newton_lax_loop(
    val: tuple[int, float, float, ArrayLike, ArrayLike, float]
) -> tuple[int, float, float, Array, Array, float]:
    """
    jax.lax.scan loop for Newton iteration of probability vector update.
    :param val: tuple of (iter_: int, x: float, eta: float, loss_sum: Array, w: Array, w_sum: float) to be updated and
      passed to next loop iteration,
    :return: Updated tuple of values.
    """
    iter_, x, eta, loss_sum, w, w_sum = val
    w = 4 / (eta * (loss_sum - x)) ** 2
    w_sum = jnp.sum(w)
    x = x - (w_sum - 1) / (eta * jnp.sum(jnp.sqrt(w) ** 3))
    return iter_ + 1, x, eta, loss_sum, w, w_sum


def newton(x: float, eta: float, loss_sum: ArrayLike) -> tuple[float, Array]:
    """
    Newton's method for probability vector update. Max 100 iterations.
    :param x: Initial value for x normalization constant.
    :param eta: Learning rate.
    :param loss_sum: Sum of losses for each index.
    :return: Tuple of updated normalization constant and probability vector.
    """
    # vals = (iter_, x, eta, loss_sum, w, w_sum)
    initial_val = (0, x, eta, loss_sum, jnp.zeros_like(loss_sum), 0)
    iter_, x, _, _, w, w_sum = lax.while_loop(
        lambda val: ~jnp.isclose(val[5], 1, atol=1e-08) & (val[0] < 8),
        _newton_lax_loop,
        initial_val,
    )
    return x, w


def _tsallis_inf_lax_loop(
    val: tuple[ArrayLike, ArrayLike, ArrayLike],
    draws_t: tuple[ArrayLike, int],
) -> tuple[tuple[Array, Array, Array], Array]:
    """
    Main jax.lax.scan loop for Tsallis-Inf algorithm.
    :param draws_t: Tuple of (3-vector of random draws, iteration).
    :param val: Tuple of
      (a: Array, x_row_col: Array, x_col: float, rc_loss_sum: Array)
    :return: Tuple of (update values, pulled arm index).
    """
    a, x_rc, rc_loss_sum = val
    draws, t = draws_t
    eta: float = jnp.sqrt(4 / (t + 1))
    x_row, p_row = newton(x_rc[0], eta, rc_loss_sum[:, 0])
    x_col, p_col = newton(x_rc[1], eta, rc_loss_sum[:, 1])
    x_rc = jnp.array([x_row, x_col])
    probs = jnp.array([p_row, p_col]).T
    rc_ix = choice(probs, draws[:2])
    row_ix, col_ix = rc_ix
    loss = draws[2] < a[row_ix, col_ix]

    rc_loss_sum = lax.cond(
        loss,
        lambda _: rc_loss_sum.at[col_ix, 1].add(1 / p_col[col_ix]),
        lambda _: rc_loss_sum.at[row_ix, 0].add(1 / p_row[row_ix]),
        None,
    )
    return (a, x_rc, rc_loss_sum), rc_ix


def tsallis_inf(
    a: ArrayLike, t_max: int, random_seed: int, n_checkpoints: int = 10
) -> NashReturn:
    """
    Tsallis-1/2-INF. See https://www.jmlr.org/papers/volume22/19-753/19-753.pdf for more details.
    :param a: Jax array of arm means.
    :param t_max: Number of iterations.
    :param random_seed: Random seed. Will be used to generate a Jax PRNGKey.
    :param n_checkpoints: Number of checkpoints to return.
    :return: Tuple of (Array of Nash guesses, checkpoint iterations, total iterations).
    """
    n, m = a.shape
    assert n == m, "Only square matrices for now."
    init_val = (
        a,
        -jnp.sqrt(jnp.array([n, n])),  # x_row, x_col
        jnp.zeros((n, 2)),  # row and column loss sums
    )
    key = random.PRNGKey(random_seed)
    # Only one row of these is being used for samples
    all_draws = random.uniform(key, shape=(t_max, 3))
    _, rc_ixs = lax.scan(
        _tsallis_inf_lax_loop,
        init_val,
        (all_draws, jnp.arange(t_max)),
    )
    checkpoint_ixs = jnp.linspace(t_max // n_checkpoints, t_max, n_checkpoints)
    return most_frequent_arms(rc_ixs, n, checkpoint_ixs), checkpoint_ixs, t_max
