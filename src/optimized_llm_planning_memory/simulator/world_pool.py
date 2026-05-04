"""
simulator/world_pool.py
=======================
WorldPool — pre-generates a fixed set of SimulatorProtocol instances and
serves them to parallel RL training environments.

Why a pool instead of creating a new world on every reset()?
------------------------------------------------------------
``CompressionEnv.reset()`` calls ``simulator_factory(seed)`` on every episode
reset.  If world generation (running WorldGenerator) takes ~0.5–2 s, a 50k-step
training run with 10 steps/episode and 2 parallel envs = ~10,000 resets, which
would spend 83–333 minutes just creating worlds.

A pool of N pre-generated worlds amortises this cost to N × generation_time at
startup.  With N=20 and 0.5 s/world that is 10 s startup, saving hours overall.

Memory footprint: each synthetic world has ~200 entities (hotels, attractions,
restaurants, flights) at ~1 KB/entity → ~200 KB/world.  A pool of 50 worlds = 10 MB.
This is negligible on any Colab T4 instance (15 GB RAM).

Diversity trade-off
-------------------
Training on a fixed pool means the policy sees at most N distinct worlds per
epoch.  With 40 training requests and N=20 worlds, the agent encounters ≤800
unique (request, world) pairs — still far more variety than a single world.
Set ``unique_per_episode=True`` in ``configs/simulator/default.yaml`` to fall
back to a fresh world every reset if world generation is cheap (small worlds or
fast hardware).

Thread safety
-------------
The pool is built once before ``make_vec_env()`` creates any subprocesses.
Worlds are returned by reference; SubprocVecEnv pickles them on spawn, so each
worker process gets its own copy.  No locks are needed.

Usage (in RLTrainer._make_vec_env())
--------------------------------------
    pool = WorldPool(
        simulator_factory=self._simulator_factory,
        pool_size=config.simulator.pool_size,
        seed_range=config.simulator.seed_range,
        rng_seed=self._config.seed,
    )
    pool.build()
    env_factory = lambda: CompressionEnv(
        simulator_factory=pool.sample,
        ...
    )
"""

from __future__ import annotations

import random
from typing import Callable

from optimized_llm_planning_memory.simulator.protocol import SimulatorProtocol
from optimized_llm_planning_memory.utils.logging import get_logger

log = get_logger(__name__)


class WorldPool:
    """
    Pre-generated pool of SimulatorProtocol instances.

    Parameters
    ----------
    simulator_factory : Callable(seed: int) → SimulatorProtocol.
                        Exactly the same factory used directly in CompressionEnv.
    pool_size         : Number of worlds to generate at startup.
    seed_range        : Inclusive [min, max] from which world seeds are drawn.
    rng_seed          : Seed for the RNG that selects which world to return on
                        each sample() call (reproducible across runs).
    """

    def __init__(
        self,
        simulator_factory: Callable[[int], SimulatorProtocol],
        pool_size: int = 20,
        seed_range: tuple[int, int] = (0, 9999),
        rng_seed: int = 42,
    ) -> None:
        self._factory = simulator_factory
        self._pool_size = pool_size
        self._seed_range = seed_range
        self._rng = random.Random(rng_seed)
        self._pool: list[SimulatorProtocol] = []

    def build(self) -> None:
        """
        Generate all worlds in the pool.

        Must be called once before ``sample()`` is used.  Called by
        ``RLTrainer._make_vec_env()`` during trainer setup.
        """
        if self._pool:
            return  # idempotent — skip if already built

        lo, hi = self._seed_range
        seeds = self._rng.sample(range(lo, hi + 1), k=min(self._pool_size, hi - lo + 1))

        log.info("world_pool.build.start", pool_size=len(seeds))
        for i, seed in enumerate(seeds):
            try:
                world = self._factory(seed)
                self._pool.append(world)
            except Exception as exc:
                log.warning("world_pool.build.error", seed=seed, error=str(exc))
        log.info("world_pool.build.done", built=len(self._pool))

    def sample(self, seed: int | None = None) -> SimulatorProtocol:
        """
        Return a random world from the pool.

        The ``seed`` parameter is accepted for API compatibility with the
        ``simulator_factory(seed)`` signature used by ``CompressionEnv``, but
        is ignored — the pool's internal RNG controls selection.  Ignoring the
        per-episode seed here is intentional: seeding every sample identically
        would cause all parallel envs to always get the same world.

        Raises
        ------
        RuntimeError
            If the pool has not been built yet.
        """
        if not self._pool:
            raise RuntimeError(
                "WorldPool is empty. Call build() before sample(). "
                "Alternatively, set simulator.unique_per_episode=true in config "
                "to skip the pool and generate a fresh world per episode."
            )
        return self._rng.choice(self._pool)

    def __len__(self) -> int:
        return len(self._pool)

    @property
    def is_built(self) -> bool:
        return len(self._pool) > 0
