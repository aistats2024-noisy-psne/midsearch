from typing import Callable

import numpy as np
from numpy.typing import NDArray
from numpy.random import default_rng


def planted_random(n: int, random_seed: int, alpha: float = 0.3) -> NDArray:
    """
    Generate a square reward matrix with a planted equilibrium.

    :param n: Matrix size.
    :param random_seed: Seed for rng.
    :param alpha: Controls the shape of the equilibrium row and column.
    :return: Reward matrix.
    """
    rng = default_rng(random_seed)
    a = rng.random((n, n))
    # increase from 0 at a[-1,0] to 0.5 at a[0,0] to 1 at a[0,-1]
    row_vals = ((np.arange(n) / n) ** alpha) / 2
    a[:, 0] = 0.5 - row_vals
    a[0, :] = 0.5 + row_vals
    # choose constant 1:n column values to make all column sums equal
    col_sum = np.sum(a[:, 0])
    a[1:, 1:] = (col_sum - a[:1, 1:]) / (n - 1)
    return a


def hard_triangular_fn(
    delta_min: float = 0.01, beta: float = 0.1
) -> Callable[[int], NDArray]:
    """
    Returns function to generate a square reward matrix with a unique PSNE.
    :param delta_min: Gap parameter.
    :param beta: Controls the shape of the equilibrium row and column.
    :return: Callable that returns a reward matrix.
    """
    assert 0 < delta_min <= beta, "Need 0 < delta_min <= beta."

    def _f(n: int, *_args) -> NDArray:
        """
        Square reward matrix with a unique PSNE. Hard instance for EXP3-IX.
        :param n: Matrix size.
        :param _args: Unused.
        :return: Reward matrix.
        """
        a = np.zeros((n, n))
        a[np.triu_indices(n)] = 1
        a[np.diag_indices(n)] = 0.5
        a[:, 0] = 0.5 - beta
        a[0, :] = 0.5 + beta
        a[0, 0] = 0.5
        a[1, 0] = 0.5 - delta_min
        a[0, 1] = 0.5 + delta_min
        return a

    _f.__name__ = f"hard_triangular[{delta_min},{beta}]"
    return _f


def gap_h1(a: NDArray, ij_star: tuple[int, int], row: bool = True) -> float:
    """
    Compute the sum of inverse gaps squared in a row or column.
    :param a: Input array
    :param ij_star: Indices of the Nash entry.
    :param row: If True, computes row gaps. Otherwise, computes column gaps.
    :return: Sum of inverse gaps squared.
    """
    n, m = a.shape
    if row:
        b = a[np.arange(n) != ij_star[0], ij_star[1]]
    else:
        b = a[ij_star[0], np.arange(m) != ij_star[1]]
    return (1 / (a[ij_star] - b) ** 2).sum()


def square_lucb(n: int, random_seed: int) -> NDArray:
    """
    Generate a square reward matrix with a unique PSNE.
    :param n: Matrix size.
    :param random_seed: Random seed.
    :return: Matrix.
    """
    rng = default_rng(random_seed)
    a = rng.random((n, n))
    a[:, 0] = 0.4
    a[0, :] = 0.6
    a[0, 0] = 0.5
    return a
