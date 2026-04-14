"""
Isolated random number generator for reproducible chain runs.

All stochastic operations in the library use ``falcomchain.random.rng``
instead of the global ``random`` module. This ensures that external
library calls (numpy, networkx, etc.) cannot perturb the chain's
random state.

Usage::

    from falcomchain.random import rng, set_seed

    set_seed(2025)        # deterministic from here
    rng.choice([1, 2, 3]) # use rng instead of random

To reproduce a chain: call ``set_seed(s)`` before constructing the
initial partition. The same seed + same library versions = identical chain.
"""

import random as _random


rng: _random.Random = _random.Random()
"""Dedicated RNG instance used by all FalcomChain stochastic operations."""


def set_seed(seed: int) -> None:
    """
    Seed the library's RNG for reproducible runs.

    :param seed: Integer seed.
    """
    rng.seed(seed)
