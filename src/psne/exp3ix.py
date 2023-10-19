import jax.random
import jax.numpy as jnp
from jax import vmap, Array, lax
from jax.typing import ArrayLike

from psne import NashReturn
from psne.sampling import choice
from psne.utils import most_frequent_arms


def _sample(
    a: ArrayLike, weights: ArrayLike, draws: ArrayLike
) -> tuple[Array, Array, Array]:
    """
    Samples arms according to distribution and updates pulls.
    Also returns sampling probabilities and the chosen arms.

    :param a: Reward matrix (n x n).
    :param weights: Weights for sampling distributions over arms (n x 2).
    :param draws: Uniform random draws for use in current iteration (3-vector).
    :return: Updated (losses, p_arms, arm index). Losses will be either [0, 1] or [1, 0].
    """
    # choose arms using the first 2 random numbers
    prob = weights / jnp.sum(weights, axis=0)
    arm_ix = choice(prob, draws[:2])
    # draw from a using the 3rd random number
    loss = draws[2] < a[*arm_ix]
    losses = jnp.array([1 - loss, loss])
    return losses, prob[arm_ix, (0, 1)], arm_ix


def _update_arm_weights(
    loss_est_sum: ArrayLike,
    weights: ArrayLike,
    loss: ArrayLike,
    p_arm: ArrayLike,
    arm: ArrayLike,
    eta: float,
) -> tuple[Array, Array]:
    """
    Update sampling weight vector for a single arm. Will be vmapped over the 2 players.
    Functional update for use with Jax.

    :param loss_est_sum: Running sum of loss estimates (n-vector).
    :param weights: Weights for sampling distributions over arms (n-vector).
    :param loss: Loss for the pulled arm (0 or 1).
    :param p_arm: Sampling probability for the pulled arm.
    :param arm: Index of the pulled arm.
    :param eta: Learning rate.
    :return: Updated (loss_est_sum, weights).
    """
    loss_est_sum = loss_est_sum.at[arm].add(-loss * eta / (p_arm + eta / 2))
    weights = weights.at[arm].set(jnp.exp(loss_est_sum[arm]))
    return loss_est_sum, weights


# vmap over players
# input shapes are (n, 2), (n, 2), (2,), (2,), (2,), ()
# output shapes are (n, 2), (n, 2)
_update_weights = vmap(
    _update_arm_weights, in_axes=(1, 1, 0, 0, 0, None), out_axes=(1, 1)
)


def _lax_update(carry, x):
    """
    Wrapper for main loop for EXP3-IX, to be consumed by jax.lax.scan.
    Signature required by jax.lax.scan is (carry, y) = f(carry, x).

    :param carry: Tuple of (a, loss_est_sum, weights).
      a: Reward matrix (n x n).
      loss_est_sum: Running sum of loss estimates (n x 2).
      weights: Weights for sampling distributions over arms (n x 2).
    :param x: Tuple of (draws, t_range) (both Jax arrays).
      draws: (n x 3) uniform random draws for use in each iteration.
      t_range: Jax array of iteration numbers.
    :return: Tuple of (updated (a, loss_est_sum, weights),  arm index).
    """
    a, loss_est_sum, weights = carry
    draws, t = x
    losses, p_arms, arm_ix = _sample(a, weights, draws)
    n = a.shape[0]
    eta: float = jnp.sqrt(jnp.log(n) / (n * (t + 1)))
    loss_est_sum, weights = _update_weights(
        loss_est_sum, weights, losses, p_arms, arm_ix, eta
    )
    return (a, loss_est_sum, weights), arm_ix


def exp3ix(
    a: ArrayLike, t_max: int, random_seed: int, n_checkpoints: int = 10
) -> NashReturn:
    """
    EXP3-IX algorithm. Details can be found at https://arxiv.org/pdf/1506.03271.pdf.
    Uses eta = sqrt(log_n/(n * t)) and gamma = eta / 2 learning rates.

    :param a: Reward matrix.
    :param t_max: Total number of iterations.
    :param random_seed: Random seed. Will be used to generate a Jax PRNGKey.
    :param n_checkpoints: Number of checkpoints to return.
    :return: Tuple of (Array of Nash indices, checkpoint indices, total number of samples).
    """
    n, m = a.shape
    assert n == m
    init_val = (
        a,
        jnp.zeros((n, 2)),  # loss_est_sum
        jnp.ones((n, 2)),  # weights
    )
    key = jax.random.PRNGKey(random_seed)
    all_draws = jax.random.uniform(key, shape=(t_max, 3))
    _, rc_ixs = lax.scan(
        _lax_update,
        init_val,
        (all_draws, jnp.arange(t_max)),
    )
    checkpoint_ixs = jnp.linspace(t_max // n_checkpoints, t_max, n_checkpoints)
    return most_frequent_arms(rc_ixs, n, checkpoint_ixs), checkpoint_ixs, t_max
