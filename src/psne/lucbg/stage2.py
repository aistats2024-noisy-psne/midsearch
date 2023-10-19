import jax.numpy as jnp
from jax import Array, jit, random, lax
from jax.typing import ArrayLike


def _sqrt_log(a: ArrayLike, b: ArrayLike) -> Array:
    """Helper function for computing UCB bounds. Returns sqrt(log(a) / b)."""
    return jnp.sqrt(jnp.log(a) / b)


def _ucb_bound(n: int, u: ArrayLike, t: ArrayLike) -> ArrayLike:
    """
    Helper function for computing UCB bounds. Delta is hardcoded for ease of use.

    Computes sqrt(log(n^2 * t^4 / (4 * delta)) / (2 * u)).
    :param n: Number of arms.
    :param u: Array of denominator values.
    :param t: Array of numerator values.
    :return: Array of bound values.
    """
    delta = 0.01
    return jnp.sqrt(jnp.log(n**2 * t**4 / (4 * delta)) / (2 * u))


def _sample_min_index(
    values: ArrayLike,
    row_ix: int,
    a: ArrayLike,
    draw: ArrayLike,
) -> tuple[bool, Array]:
    """
    For a given matrix of values, for each row, samples the arm with the lowest value and returns the updated
    mu and pulls vectors.

    :param values: Array of values to take minimum of.
    :param row_ix: Index of the row to sample from.
    :param a: n x n reward matrix.
    :param draw: Random draw from uniform distribution.
    :return: Tuple of (loss, sample indices).
    """
    sample_ixs = jnp.argmin(values, axis=1)
    # Indexes into multiple arms
    s = (row_ix, sample_ixs)
    loss = draw < a[s]
    return loss, sample_ixs


def _pull_row_arms(
    ix: int,
    key: random.PRNGKey,
    a: ArrayLike,
    mu: ArrayLike,
    row_pulls: ArrayLike,
    col_pulls_t: ArrayLike,
    lower_ucb: ArrayLike,
    selected: ArrayLike,
) -> tuple[Array, Array, Array]:
    """
    Pulls arms for a given row.

    Assumes row layout. To use with columns, pass in (1-a.T), (1-mu.T), etc. (see lucbg.py).
    Slightly differs from paper in timing of updates for brevity.

    :param ix: Row index for pulling arms.
    :param a: n x n reward matrix.
    :param mu: n x n empirical mean matrix.
    :param row_pulls: n x n row pulls matrix.
    :param col_pulls_t: n x n column pulls matrix transposed.
    :param lower_ucb: n x n lower UCB matrix.
    :param selected: n-vector of number of times each row has been selected for stage 2.
    :return: Tuple of (updated mu, updated row_pulls, updated lower_ucb).
    """
    n = a.shape[0]
    # tau = 3 * (number of times row ix has been selected for stage 2)
    tau = 3 * selected[ix]
    row_mu = mu[ix]  # vector of means
    row_pulls_ix = row_pulls[ix]  # vector of row_pulls only
    bounds = row_mu - jnp.array(
        [
            jnp.zeros(n),
            _sqrt_log(tau / 3, row_pulls_ix / 2),
        ]
    )
    # Sample according to the first 2 bounds
    draws = random.uniform(key, shape=(3,))
    loss12, s12 = _sample_min_index(bounds, ix, a, draws[:2])
    bound_3 = row_mu - jnp.array([lower_ucb[ix]])
    # For the third sample, exclude s1
    bound_3 = bound_3.at[0, s12[0]].set(jnp.max(bound_3) + 1)
    loss_3, s3 = _sample_min_index(bound_3, ix, a, draws[2])
    loss_s = jnp.concatenate([loss12, loss_3])
    s = jnp.concatenate([s12, s3])
    ix_s = (ix, s)
    # Update row pulls
    row_pulls = row_pulls.at[ix_s].add(1)
    row_pulls_s = row_pulls[ix_s]
    # Need to compute new *total* pulls at that index to update total mean
    # - note flipped indices for col_pulls_t
    total_pulls_s = row_pulls_s + col_pulls_t[s, ix]
    mu = mu.at[ix_s].set((mu[ix_s] * (total_pulls_s - 1) + loss_s) / total_pulls_s)
    # lower_ucb = beta(T, tau)
    lower_ucb = lower_ucb.at[ix_s].set(_ucb_bound(n, row_pulls_s, tau))
    return mu, row_pulls, lower_ucb


@jit
def pull_arms(ix, *args):
    """
    Wrapper for _pull_row_arms that handles the -1 case.
    :param ix: Index of row to pull arms from.
    :param args: Arguments to pass to _pull_row_arms.
    :return: Tuple of (updated mu, updated row_pulls, updated lower_ucb).
    """
    return lax.cond(
        ix != -1,
        _pull_row_arms,
        lambda _ix, *args_: (args_[2], args[3], args_[5]),  # noop
        ix,
        *args,
    )
