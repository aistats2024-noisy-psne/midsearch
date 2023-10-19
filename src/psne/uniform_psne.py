import jax.numpy as jnp
from jax import random, lax, Array
from jax.typing import ArrayLike

from psne import NashReturn


def _find_psne(key: random.PRNGKey, a: ArrayLike) -> Array:
    """
    Find a (non-strict) PSNE of a matrix. If there is no PSNE, return a random Nash equilibrium.
    :param key: Random key.
    :param a: Input matrix.
    :return: Nash equilibrium.
    """
    n = len(a)
    row_mins: Array = jnp.min(a, axis=1)[:, None]
    col_maxs: Array = jnp.max(a, axis=0)
    mask = row_mins == col_maxs
    ij_star = jnp.argwhere(mask, size=1, fill_value=-1)[0]
    # If there's no Nash, just guess at random
    return lax.cond(
        jnp.all(ij_star == -1),
        lambda _: random.randint(key, shape=(2,), minval=0, maxval=n),
        lambda _: ij_star,
        None,
    )


def _lax_scan_loop(vals: tuple, _) -> tuple[tuple, None]:
    """
    Main lax.scan loop for uniform sampling algorithm.
    :param vals: Tuple of (key, a, successes array, pulls array).
    :param _: Unused
    :return: Tuple of (updated values, None).
    """
    key, a, successes, pulls = vals
    key, subkey1, subkey2 = random.split(key, 3)
    n = len(successes)
    idx = random.randint(subkey1, shape=(2,), minval=0, maxval=n)
    pull = random.uniform(subkey2) < a[*idx]
    successes = successes.at[*idx].add(pull)
    pulls = pulls.at[*idx].add(1)
    return (key, a, successes, pulls), None


def uniform_psne(
    a: ArrayLike, t_max: int, random_seed: int, n_checkpoints: int = 10
) -> NashReturn:
    """
    Uniform sampling algorithm for finding a PSNE.

    :param a: Input matrix.
    :param t_max: Number of samples to take.
    :param random_seed: Random seed.
    :param n_checkpoints: Number of checkpoints to return.
    :return: Tuple of (Array of Nash guesses, checkpoint iterations, total iterations).
    """
    n, _ = a.shape
    key = random.PRNGKey(random_seed)
    key, subkeys = random.split(key)
    # Sample each arm at least once
    successes = jnp.int32(random.uniform(subkeys, shape=(n, n)) < a)
    pulls = jnp.ones((n, n))
    ij_hats = jnp.zeros((n_checkpoints, 2), dtype=int)
    # TODO making this not constant (hashable/static) is difficult,
    #  so just take approximate checkpoints
    dt_max = (t_max - n**2) // n_checkpoints
    for k in range(n_checkpoints):
        key, subkey1, subkey2 = random.split(key, 3)
        init_val = (subkey1, a, successes, pulls)
        (_, _, successes, pulls), _ = lax.scan(
            _lax_scan_loop,
            init_val,
            xs=None,
            length=dt_max,
        )
        ij_hats = ij_hats.at[k].set(_find_psne(subkey2, successes / pulls))
    checkpoint_ixs = jnp.arange(1, n_checkpoints + 1) * dt_max
    return ij_hats, checkpoint_ixs, dt_max * n_checkpoints
