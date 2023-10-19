import jax.numpy as jnp
from jax import Array, vmap
from jax.typing import ArrayLike


def _choice(p: ArrayLike, draws: ArrayLike) -> Array:
    """
    Fast version of np.random.choice for a single draw.

    To be used with vmap.
    :param p: Sampling distribution over arms. NOTE: not checked for sum to 1.
    :param draws: Random draws from [0, 1).
    :return: Index of the chosen arm.
    """
    return jnp.searchsorted(jnp.cumsum(p), draws)


# vmapped version of choice
choice = vmap(_choice, in_axes=(1, 0), out_axes=0)
# useful for multiple draws x multiple arms
choice_2 = vmap(_choice, in_axes=(1, 1), out_axes=1)
