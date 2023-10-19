import jax.numpy as jnp
from jax import lax
from jax.typing import ArrayLike


def _validate_row_mins(mu: ArrayLike, ucb: ArrayLike) -> tuple[bool, int]:
    """
    Determines if the largest upper bound is less than all lower bounds in that row (except itself).
    Used to determine a probable minimum.
    :param mu: Row of empirical means.
    :param ucb: Row of confidence bounds.
    :return: Tuple of (is_valid, index of probable minimum).
    """
    upper_bound = mu + ucb
    lower_bound = mu - ucb
    min_ix = jnp.argmin(upper_bound)
    min_upper_bound = upper_bound[min_ix]
    c = jnp.count_nonzero(min_upper_bound > lower_bound)
    return (c == 1), min_ix


def validate_row_mins(ix, mu, ucb) -> tuple[bool, int]:
    """
    Wrapper of _validate_row_mins that handles -1 indices.
    :param ix: Index of row to validate.
    :param mu: Row of empirical means.
    :param ucb: Row of confidence bounds.
    :return: Tuple of (is_valid, index of probable minimum).
    """
    return lax.cond(
        ix != -1,
        _validate_row_mins,
        lambda *_: (False, -1),  # noop
        mu[ix],
        ucb[ix],
    )
