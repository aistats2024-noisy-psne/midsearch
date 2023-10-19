import jax.numpy as jnp
from jax import vmap, Array
from jax.typing import ArrayLike
from numpy.random import default_rng

rng_seed = 76771
base_rng = default_rng(seed=rng_seed)
# Can use RandomState as a backup, if needed
# base_rng = np.random.RandomState(rng_seed)


def shuffle_rows_cols(a: ArrayLike, random_seed) -> tuple[Array, tuple]:
    """
    Shuffle rows and columns of a matrix.
    :param a: Input matrix.
    :param random_seed: Random seed.
    :return: Tuple of (shuffled matrix, new location of (0, 0) entry).
    """
    d = a.shape[0]
    shuffle_rng = default_rng(random_seed)
    row_p, col_p = list(shuffle_rng.permutation(d)), list(shuffle_rng.permutation(d))
    a = a[row_p, :][:, col_p]
    ij_star = (row_p.index(0), col_p.index(0))
    return a, ij_star


def _partial_bincount(items: ArrayLike, n: int, bound: int) -> Array:
    """Helper function for most_frequent_arms."""
    m = items.shape[0]
    return jnp.bincount(items, weights=1 * (jnp.arange(m) < bound), length=n**2)


_v_partial_bincount = vmap(_partial_bincount, in_axes=(None, None, 0), out_axes=0)


def most_frequent_arms(ind_array: ArrayLike, n: int, checkpoints: ArrayLike) -> Array:
    """
    Given a T x 2 array, return the most frequently occurring row at various submatrices,
    i.e. when taking the top k rows.
    :param ind_array: Input array.
    :param n: Maximum value of each entry in ind_array.
    :param checkpoints: Array of checkpoints (i.e. top k_i rows to consider).
    :return: Array of most frequent rows.
    """
    flat_ixs = ind_array[:, 0] * n + ind_array[:, 1]
    pull_counts = _v_partial_bincount(flat_ixs, n, checkpoints)
    max_ix = jnp.argmax(pull_counts, axis=1)
    return jnp.array(jnp.divmod(max_ix, n)).T
