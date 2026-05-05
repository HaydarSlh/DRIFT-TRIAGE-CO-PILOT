"""Smoke tests for drift compute functions."""
import numpy as np
import pytest
from platform.drift.compute import psi


def test_psi_zero_for_identical_distributions():
    rng = np.random.default_rng(42)
    data = rng.normal(size=1000)
    score = psi(data, data)
    assert score == pytest.approx(0.0, abs=1e-2)


def test_psi_positive_for_shifted_distribution():
    rng = np.random.default_rng(42)
    expected = rng.normal(loc=0, size=1000)
    actual = rng.normal(loc=3, size=1000)
    assert psi(expected, actual) > 0.1
