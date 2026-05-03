"""
Unit tests — WorldPool pre-generation and sampling.

Verifies:
1. build() creates pool_size worlds.
2. sample() returns a world from the pool.
3. Calling build() twice is idempotent.
4. sample() before build() raises RuntimeError.
5. pool.sample ignores its seed argument (intentional design).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from optimized_llm_planning_memory.simulator.world_pool import WorldPool


def _make_factory(call_log: list):
    """Return a factory that records each call and returns a new MagicMock world."""
    def factory(seed: int) -> MagicMock:
        world = MagicMock(name=f"world_{seed}")
        call_log.append(seed)
        return world
    return factory


@pytest.mark.unit_test
class TestWorldPool:
    def test_build_creates_correct_number_of_worlds(self):
        call_log = []
        pool = WorldPool(
            simulator_factory=_make_factory(call_log),
            pool_size=5,
            seed_range=(0, 100),
            rng_seed=42,
        )
        pool.build()
        assert len(pool) == 5
        assert len(call_log) == 5

    def test_build_is_idempotent(self):
        call_log = []
        pool = WorldPool(
            simulator_factory=_make_factory(call_log),
            pool_size=3,
            seed_range=(0, 100),
            rng_seed=42,
        )
        pool.build()
        pool.build()  # second call should be no-op
        assert len(pool) == 3
        assert len(call_log) == 3  # factory only called once

    def test_sample_before_build_raises_runtime_error(self):
        pool = WorldPool(
            simulator_factory=_make_factory([]),
            pool_size=5,
            seed_range=(0, 100),
        )
        with pytest.raises(RuntimeError, match="WorldPool is empty"):
            pool.sample()

    def test_sample_returns_object_from_pool(self):
        call_log = []
        factory = _make_factory(call_log)
        pool = WorldPool(
            simulator_factory=factory,
            pool_size=4,
            seed_range=(0, 100),
            rng_seed=0,
        )
        pool.build()
        world = pool.sample(seed=999)  # seed argument is intentionally ignored
        assert world is not None

    def test_sample_distributes_across_pool(self):
        """With enough samples, all worlds in the pool should be returned."""
        worlds = [MagicMock(name=f"world_{i}") for i in range(5)]
        idx = [0]

        def rotating_factory(seed):
            w = worlds[idx[0] % len(worlds)]
            idx[0] += 1
            return w

        pool = WorldPool(
            simulator_factory=rotating_factory,
            pool_size=5,
            seed_range=(0, 100),
            rng_seed=7,
        )
        pool.build()

        sampled = {id(pool.sample()) for _ in range(50)}
        # Over 50 samples from a pool of 5, we expect multiple distinct worlds
        assert len(sampled) > 1

    def test_is_built_property(self):
        pool = WorldPool(
            simulator_factory=_make_factory([]),
            pool_size=2,
            seed_range=(0, 10),
        )
        assert not pool.is_built
        pool.build()
        assert pool.is_built

    def test_pool_size_capped_by_seed_range(self):
        """When pool_size > available seeds, pool is capped to range size."""
        call_log = []
        pool = WorldPool(
            simulator_factory=_make_factory(call_log),
            pool_size=100,
            seed_range=(0, 3),  # only 4 distinct seeds
            rng_seed=0,
        )
        pool.build()
        assert len(pool) == 4  # capped at range size
