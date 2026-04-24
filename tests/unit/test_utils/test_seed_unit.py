"""Unit tests for utils/seed.py — set_seed reproducibility."""

from __future__ import annotations

import random

import pytest

from optimized_llm_planning_memory.utils.seed import set_seed


@pytest.mark.unit
class TestSetSeedPython:
    def test_same_seed_produces_same_sequence(self):
        set_seed(42)
        seq1 = [random.random() for _ in range(5)]
        set_seed(42)
        seq2 = [random.random() for _ in range(5)]
        assert seq1 == seq2

    def test_different_seeds_produce_different_sequences(self):
        set_seed(1)
        seq1 = [random.random() for _ in range(5)]
        set_seed(2)
        seq2 = [random.random() for _ in range(5)]
        assert seq1 != seq2

    def test_set_seed_does_not_raise(self):
        set_seed(0)
        set_seed(1)
        set_seed(2**31 - 1)

    def test_set_seed_called_twice_same_result(self):
        set_seed(99)
        v1 = random.randint(0, 10000)
        set_seed(99)
        v2 = random.randint(0, 10000)
        assert v1 == v2


@pytest.mark.unit
class TestSetSeedNumpy:
    def test_numpy_seeded_deterministically(self):
        np = pytest.importorskip("numpy")
        set_seed(42)
        arr1 = np.random.rand(5).tolist()
        set_seed(42)
        arr2 = np.random.rand(5).tolist()
        assert arr1 == arr2


@pytest.mark.unit
class TestSetSeedTorch:
    def test_torch_seeded_deterministically(self):
        torch = pytest.importorskip("torch")
        set_seed(7)
        t1 = torch.rand(5).tolist()
        set_seed(7)
        t2 = torch.rand(5).tolist()
        assert t1 == t2
