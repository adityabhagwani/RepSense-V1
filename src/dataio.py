"""File formats shared by every stage: what the board sends and what we save.

The receiver writes these files and everything downstream reads them, so the
formats are defined once here instead of being retyped in each script.
"""

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Columns after load_session(): acceleration in g, angular rate in deg/s.
ACCEL = ["ax", "ay", "az"]
GYRO = ["gx", "gy", "gz"]

# One line from the board: seq,t_us,ax_mg,ay_mg,az_mg,gx_cdps,gy_cdps,gz_cdps
BOARD_FIELDS = ["seq", "t_us", "ax_mg", "ay_mg", "az_mg", "gx_cdps", "gy_cdps", "gz_cdps"]

# raw/<session>.csv: the board's fields plus host_t, the Mac arrival time.
# host_t is the clock shared with the annotations file.
RAW_COLUMNS = ["host_t", *BOARD_FIELDS]

# annotations/<session>.csv: one row per set attempt.
ANNOTATION_COLUMNS = [
    "session", "participant", "set_idx", "exercise", "scripted_reps", "actual_reps",
    "condition", "notes", "start_host_t", "end_host_t", "n_samples", "dropped", "status",
]

# protocols/<session>.csv `marked` column: scripted movements as movement:kind, e.g.
# "4:half_lift;9:half_lower". A movement is any lift in the set, half reps included, numbered
# from 1. Unmarked = full. Half reps don't count toward `reps`; drift and pause reps do.
HALF_KINDS = ("half_lift", "half_lower")
MARK_KINDS = (*HALF_KINDS, "drift", "pause")


def format_marked(marks: dict[int, str]) -> str:
    return ";".join(f"{m}:{k}" for m, k in sorted(marks.items()))


def parse_marked(text: str) -> dict[int, str]:
    """`marked` cell -> {movement: kind}. Raises on anything malformed instead of guessing."""
    marks = {}
    for item in filter(None, text.split(";")):
        movement, _, kind = item.partition(":")
        if not movement.isdigit() or kind not in MARK_KINDS or int(movement) in marks:
            raise ValueError(f"bad marked entry {item!r} in {text!r}")
        marks[int(movement)] = kind
    return marks


def movement_count(reps: int, marks: dict[int, str]) -> int:
    """Lifts in a set: the full reps plus the scripted half reps."""
    return reps + sum(kind in HALF_KINDS for kind in marks.values())


def with_true_reps(ann: pd.DataFrame, session: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """Annotations plus `true_reps`: the typed full-rep count unless data/count_fixes.csv corrects it.

    The fix log is append-only; the latest fix for a set wins. Raw annotations are never edited.
    """
    out = ann.copy()
    out["true_reps"] = out["actual_reps"]
    path = data_dir / "count_fixes.csv"
    if path.exists():
        fixes = pd.read_csv(path)
        fixes = fixes[fixes["session"] == session].drop_duplicates("set_idx", keep="last")
        for _, f in fixes.iterrows():
            out.loc[(out["set_idx"] == f["set_idx"]) & (out["status"] == "ok"), "true_reps"] = f["full_reps"]
    return out


def raw_path(session: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "raw" / f"{session}.csv"


def annotation_path(session: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "annotations" / f"{session}.csv"


def to_physical(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw file rows -> board time in seconds and sensor values in g and deg/s.

    `t` comes from the board clock, unwrapped across micros() overflow.
    `segment` counts board reboots: it goes up wherever seq restarts.
    """
    seq = raw["seq"].to_numpy(np.int64)
    host_t = raw["host_t"].to_numpy(float)
    dt_us = np.diff(raw["t_us"].to_numpy(np.int64)) % 2**32  # micros() wraps every ~71.6 min
    reboot = np.diff(seq) <= 0
    # Across a reboot the board clock restarts, so bridge the gap with the Mac clock.
    dt_us = np.where(reboot, np.round(np.diff(host_t) * 1e6), dt_us)

    df = pd.DataFrame({
        "host_t": host_t,
        "seq": seq,
        "t": np.concatenate([[0.0], np.cumsum(dt_us) / 1e6]),
        "segment": np.concatenate([[0], np.cumsum(reboot)]),
    })
    for name, col in zip(ACCEL, ["ax_mg", "ay_mg", "az_mg"]):
        df[name] = raw[col].to_numpy() / 1000
    for name, col in zip(GYRO, ["gx_cdps", "gy_cdps", "gz_cdps"]):
        df[name] = raw[col].to_numpy() / 100
    # Undo a mirror. For this board the Arduino library reports x = -(chip y), y = -(chip x),
    # which is a mirror image of the chip's axes. Directions (accelerometer) survive a mirror;
    # turns (gyro) come out reversed, so gyro and accelerometer disagree about which way the
    # board moved. Flipping z on both turns the mirror back into a proper frame. The USB axis
    # (-x) is unchanged. validate.check_frame catches any frame like this.
    df["az"] = -df["az"]
    df["gz"] = -df["gz"]
    return df


def rotation_speed(df: pd.DataFrame) -> np.ndarray:
    """How fast the sensor is turning (deg/s), whatever its orientation: the gyro vector's length."""
    return np.sqrt((df[GYRO] ** 2).sum(axis=1)).to_numpy()


def load_session(session: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return to_physical(pd.read_csv(raw_path(session, data_dir)))


def load_annotations(session: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(annotation_path(session, data_dir))


def slice_set(df: pd.DataFrame, row) -> pd.DataFrame:
    """A set's samples: the shared-clock join between the raw file and one annotation row."""
    return df[(df["host_t"] >= row["start_host_t"]) & (df["host_t"] <= row["end_host_t"])]


def parse_line(line: str) -> list[int] | None:
    """Turn one board line into 8 ints, or None for info lines and garbage."""
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split(",")
    if len(parts) != len(BOARD_FIELDS):
        return None
    try:
        return [int(p) for p in parts]
    except ValueError:
        return None
