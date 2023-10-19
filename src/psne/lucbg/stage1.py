import jax
import jax.numpy as jnp
from jax import Array, jit
from jax.typing import ArrayLike


@jit
def pure_nash(a: ArrayLike) -> Array:
    """
    Returns a (nonstrict) pure Nash equilibrium of a. Returns [-1, -1] if one doesn't exist.
    :param a: n x n matrix.
    :return: 2-element array of indices of pure Nash equilibrium.
    """
    row_mins: Array = jnp.min(a, axis=1)[:, None]
    col_maxs: Array = jnp.max(a, axis=0)
    mask = row_mins == col_maxs
    return jnp.argwhere(mask, size=1, fill_value=-1)[0]


@jit
def select_tasks(mu: ArrayLike, active_rc: ArrayLike, selected_rc: ArrayLike) -> Array:
    """
    Selects the task (row/column pair) to sample from in the next step.

    Returns the PSNE if it exists; else, returns the least-sampled active row/column, if possible.

    :param mu: n x n matrix of empirical means.
    :param active_rc: n x 2 bool matrix of active rows and columns.
    :param selected_rc: n x 2 matrix of number of times each row and column has been selected.
    :return: 2-element array of task indices.
    """
    null = jnp.array([-1, -1])
    ij_nash = pure_nash(mu)
    # filter ij_nash by it being active
    ok_nash = (ij_nash != -1) & active_rc[ij_nash, (0, 1)]
    ij_nash = jax.lax.select(ok_nash, ij_nash, null)
    # make sure inactive rows and columns are not selected
    activated = jnp.where(active_rc, x=selected_rc, y=jnp.max(selected_rc) + 1)
    least_sampled = jax.lax.select(
        jnp.any(active_rc, axis=0), jnp.argmin(activated, axis=0), null
    )
    ij_hat = jax.lax.select(ij_nash != -1, ij_nash, least_sampled)
    return ij_hat
