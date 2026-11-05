"""Validator: a scripted rest in the middle of a set is not idle time."""

import numpy as np
import pandas as pd

from src.validate import check_set
from test_counter import FS, FULL, elevation
from test_detector import sensors


def test_a_scripted_rest_is_left_out_of_the_movement_and_pace_checks():
    angle = np.r_[elevation([FULL]), np.zeros(int(15 * FS)), elevation([FULL])]  # the rest is most of the set
    seg = sensors(angle, mount_seed=2, seed=1)
    row = pd.Series({"set_idx": 7, "exercise": "curl", "n_samples": len(seg), "actual_reps": 2})
    level = lambda checks, name: next(c.level for c in checks if c.name == name)
    assert (level(check_set(seg, row), "movement"), level(check_set(seg, row), "rep pace")) == ("FAIL", "WARN")
    assert (level(check_set(seg, row, 15.0), "movement"), level(check_set(seg, row, 15.0), "rep pace")) == ("PASS", "PASS")
