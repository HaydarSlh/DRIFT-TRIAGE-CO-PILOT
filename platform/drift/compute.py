"""Drift scoring — PSI and chi-squared statistics."""
from __future__ import annotations

import numpy as np


def psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """Population Stability Index."""
    breakpoints = np.linspace(0, 100, buckets + 1)
    expected_pcts = np.histogram(expected, np.percentile(expected, breakpoints))[0] / len(expected)
    actual_pcts = np.histogram(actual, np.percentile(expected, breakpoints))[0] / len(actual)
    expected_pcts = np.where(expected_pcts == 0, 1e-4, expected_pcts)
    actual_pcts = np.where(actual_pcts == 0, 1e-4, actual_pcts)
    return float(np.sum((actual_pcts - expected_pcts) * np.log(actual_pcts / expected_pcts)))


def chi_squared(expected: np.ndarray, actual: np.ndarray) -> float:
    """Chi-squared drift statistic."""
    from scipy.stats import chi2_contingency
    contingency = np.array([expected, actual])
    stat, *_ = chi2_contingency(contingency)
    return float(stat)
