import numpy as np
import json
from scipy.stats import chi2_contingency
from ml_platform.config import REFERENCE_STATS_PATH
import pandas as pd

class RollingWindow:
    def __init__(self, max_size=500):
        self.max_size = max_size
        self.records = []

    def add(self, record):
        self.records.append(record)
        if len(self.records) > self.max_size:
            self.records = self.records[-self.max_size:]

    def to_df(self):
        return pd.DataFrame(self.records)

def psi(actual_values, reference_bins, reference_bin_props):
    bins = np.array(reference_bins)
    bins_expanded = np.concatenate([[-np.inf], bins[1:-1], [np.inf]])
    actual_counts, _ = np.histogram(actual_values, bins=bins_expanded)
    total_actual = actual_counts.sum()
    if total_actual == 0:
        return 0.0
    actual_dist = actual_counts / total_actual
    expected_dist = np.array(reference_bin_props)
    epsilon = 1e-10
    actual_dist = np.clip(actual_dist, epsilon, 1)
    expected_dist = np.clip(expected_dist, epsilon, 1)
    return float(np.sum((actual_dist - expected_dist) * np.log(actual_dist / expected_dist)))

def chi2(observed, expected_freqs):
    obs_counts = observed.value_counts()
    categories = list(expected_freqs.keys())
    obs_vec = np.array([obs_counts.get(cat, 0) for cat in categories], dtype=np.float64)
    total = obs_vec.sum()
    if total == 0:
        return 0.0
    exp_vec = np.array([expected_freqs[cat] * total for cat in categories], dtype=np.float64)
    chi2_stat, _, _, _ = chi2_contingency([obs_vec, exp_vec])
    return float(chi2_stat)

def output_drift(current_positive_rate, reference_positive_rate):
    return current_positive_rate - reference_positive_rate

def compute_severity(psi_values, chi2_values, output_drift_val,
                     psi_thresholds=(0.1, 0.25), chi2_critical=10.0, output_drift_thresh=0.1):
    max_psi = max(psi_values.values()) if psi_values else 0.0
    max_chi2 = max(chi2_values.values()) if chi2_values else 0.0
    if max_psi >= psi_thresholds[1] or max_chi2 >= chi2_critical or abs(output_drift_val) > output_drift_thresh:
        return "red"
    elif max_psi >= psi_thresholds[0] or max_chi2 >= chi2_critical/2:
        return "yellow"
    return "green"