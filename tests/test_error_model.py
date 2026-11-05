"""Count-error symptoms on simulated sets whose trouble we planted: a clean set reads calm, a rest shows
in the gaps, a vetoed set shows in the rule-combined gap, a borderline hump moves with the cut, lost
samples and a clipped gyro show in signal quality."""

import numpy as np
import pytest

from src.error_model import risk_coverage, symptoms
from src.splice import degrade
from test_counter import FS, FULL, elevation
from test_detector import sensors


class Sure:
    """A learned model that gives every hump the same probability."""
    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        return np.column_stack([np.full(len(X), 1 - self.p), np.full(len(X), self.p)])


def test_a_clean_set_reads_calm():
    seg = sensors(elevation([FULL] * 8), mount_seed=7, seed=3)
    s = symptoms(seg, Sure(0.9), None)
    assert (s["rule"], s["combined"], s["shipped"]) == (8, 8, 8)
    assert s["rule_minus_combined"] == 0 and s["threshold_stability"] == 0
    assert s["learned_close_calls"] == 0 and s["counter_disagreement"] == 0
    assert s["longest_gap_ratio"] < 1.2 and s["rhythm_strength"] > 0.8 and s["signal_quality"] < 0.01
    assert s["seconds_per_rep"] == pytest.approx((seg["t"].iloc[-1] - seg["t"].iloc[0]) / 8)


def test_a_rest_shows_in_the_gaps_and_a_veto_in_the_rule_combined_gap():
    rest = sensors(np.r_[elevation([FULL] * 4), np.zeros(int(8 * FS)), elevation([FULL] * 4)], mount_seed=7, seed=3)
    s = symptoms(rest, Sure(0.9), None)
    assert s["shipped"] == 8 and s["longest_gap_ratio"] > 3 and s["rhythm_irregularity"] > 0.5
    vetoed = symptoms(sensors(elevation([FULL] * 6), mount_seed=7, seed=3), Sure(0.01), None)
    assert vetoed["combined"] == 0 and vetoed["rule_minus_combined"] == vetoed["rule"] == 6
    assert vetoed["counter_disagreement"] == 3  # mean of |shipped 0 - rule 6| and |shipped 0 - learned 0|


def test_a_borderline_hump_moves_with_the_cut():
    reps = [FULL] * 3 + [{"top": 96}] + [FULL] * 3  # 0.8 of a full hump: just under the 0.825 cut
    s = symptoms(sensors(elevation(reps), mount_seed=7, seed=3), Sure(0.9), None)
    assert s["rule"] == 6 and s["threshold_stability"] == 1 and s["close_calls"] > 0


def test_lost_samples_and_a_clipped_gyro_show_in_signal_quality():
    seg = sensors(elevation([FULL] * 6), mount_seed=7, seed=3)
    seg.insert(0, "host_t", seg["t"])
    assert symptoms(degrade(seg, 0.05, 60.0, seed=1), Sure(0.9), None)["signal_quality"] > 0.05


def test_risk_coverage_keeps_the_lowest_risk_sets_first():
    rc = risk_coverage(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.9, 0.8]))
    assert rc["error"].tolist() == [0.0, 0.0, 0.5 * 2 / 3, 0.5]
