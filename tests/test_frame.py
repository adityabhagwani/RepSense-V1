"""The Nano's axes: the library's remap mirrors them, to_physical undoes it, check_frame catches it."""

import numpy as np
import pandas as pd

from src.dataio import ACCEL, GYRO, to_physical
from src.validate import check_frame
from test_counter import FULL, wrist_set


def library_remap(df):
    """What the Arduino library does on this board: x = -(chip y), y = -(chip x). A mirror."""
    out = df.copy()
    for a, b in (("ax", "ay"), ("gx", "gy")):
        out[a], out[b] = -df[b], -df[a]
    return out


def undo_mirror(df):
    out = df.copy()
    out["az"], out["gz"] = -df["az"], -df["gz"]
    return out


def test_to_physical_flips_z_on_both_sensors():
    raw = pd.DataFrame({"host_t": [0.0, 0.02], "seq": [0, 1], "t_us": [0, 20000],
                        "ax_mg": [10, 10], "ay_mg": [20, 20], "az_mg": [1000, 1000],
                        "gx_cdps": [100, 100], "gy_cdps": [200, 200], "gz_cdps": [500, 500]})
    df = to_physical(raw)
    assert df[ACCEL].iloc[0].tolist() == [0.01, 0.02, -1.0]
    assert df[GYRO].iloc[0].tolist() == [1.0, 2.0, -5.0]


def test_frame_check_fails_on_mirrored_axes_and_passes_once_undone():
    chip = wrist_set([FULL] * 10)  # consistent axes, as the chip itself measures
    mirrored = library_remap(chip)
    assert check_frame(chip)[0].level == "PASS"
    assert check_frame(mirrored)[0].level == "FAIL"
    assert check_frame(undo_mirror(mirrored))[0].level == "PASS"
