import os

# See README.md for why this is necessary
n_workers = 8
os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={n_workers}"

from typing import Callable
from numpy.typing import NDArray
from jax import Array, config
from jax.typing import ArrayLike

# Using float64 is helpful for stability of Tsallis-inf Newton step.
config.update("jax_enable_x64", True)

# All algorithms should have type signature:
# alg(a: ArrayLike, t_max: int, random_seed: int, n_checkpoints: int)
#   -> tuple[Array(n_checkpoints, 2), Array(n_checkpoints), int]
NashReturn = tuple[Array, Array, int]
NashAlgorithm = Callable[[ArrayLike, int, int, int], NashReturn]

# All problem generators should have the type signature, and should all
# have (0, 0) as the Nash equilibrium:
# prob_fn(dim: int, random_seed: int, **kwargs) -> NDArray
ProblemGenerator = Callable[[int, int], NDArray]

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data"))
TMP_DATA_DIR = os.path.join(DATA_DIR, "tmp")
