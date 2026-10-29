"""Check that a recorded session can be trusted before it joins the dataset.

Three layers of checks, one per layer of failure:
  timing  (whole file)  did USB or the firmware lose or delay samples?
  sensor  (per set)     did the signal clip, is the sensor calibrated?
  labels  (per set)     does the signal agree with what the annotation says happened?

FAIL means re-record, WARN means usable but worth knowing, PASS means fine.
Exit code is 1 if anything failed, so scripts can refuse bad sessions.

    python -m src.validate validation2
"""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.dataio import ACCEL, GYRO, load_annotations, load_session, rotation_speed, slice_set
from src.record import MIN_SET_SECONDS

REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"

# Timing: what the firmware promises.
HZ = 50
RATE_TOLERANCE_HZ = 1.0
LATE_S = 1.5 / HZ               # a gap over 1.5 periods means the board skipped a slot
MAX_LATE_FRACTION = 0.001

# Sensor: library ranges are +-4 g and +-2000 deg/s; readings this close are clipped.
ACCEL_CLIP_G = 3.99
GYRO_CLIP_DPS = 1995
MAX_CLIP_FRACTION = 0.01        # lift sets: a few clipped samples are a WARN, more is a FAIL
EXPECTED_CLIPPING = {"shake"}   # shaking hard is supposed to hit the limits
GRAVITY_TOLERANCE_G = 0.05      # validation2 still set read 1.002 g
MAX_GYRO_BIAS_DPS = 3.0         # validation2 at rest read 1.1 deg/s

# Labels: measured on validation2 (rest 1.1 deg/s, unweighted curls ~140 deg/s median).
# Revisit after Phase 2 dev sessions add slow and paused sets.
QUIET_WINDOW_S = 10             # judge a still set on its quietest stretch, not the settling
STILL_DPS = 5.0                 # comfortably above the 1.1 deg/s resting reading
MIN_STILL_FRACTION = 0.8
MOVING_DPS = 20.0               # well above rest, well below any deliberate movement
MIN_MOVING_FRACTION = 0.3
MIN_S_PER_REP = 0.8             # faster than this is not a real rep
MAX_S_PER_REP = 10.0            # slower usually means idle time inside the set window

LEVELS = ["PASS", "WARN", "FAIL"]


@dataclass
class Check:
    scope: str    # "file" or a set like "3 curl"
    name: str
    level: str    # PASS, WARN or FAIL
    detail: str


def worst(levels) -> str:
    return max(levels, key=LEVELS.index, default="PASS")


def check_timing(df: pd.DataFrame) -> list[Check]:
    """Layer 1: did every sample arrive, on time, from one uninterrupted boot?"""
    same_boot = np.diff(df["segment"].to_numpy()) == 0
    dt = np.diff(df["t"].to_numpy())[same_boot]
    lost = int(np.clip(np.diff(df["seq"].to_numpy())[same_boot] - 1, 0, None).sum())
    late = int((dt > LATE_S).sum())
    reboots = int(df["segment"].iloc[-1])
    rate = (len(df) - 1) / df["t"].iloc[-1]

    def level(ok):
        return "PASS" if ok else "FAIL"

    return [
        Check("file", "rate", level(abs(rate - HZ) <= RATE_TOLERANCE_HZ), f"{rate:.2f} Hz (want {HZ} ±{RATE_TOLERANCE_HZ:g})"),
        Check("file", "lost lines", level(lost == 0), f"{lost} missing seq numbers (USB layer)"),
        Check("file", "late samples", level(late <= MAX_LATE_FRACTION * len(df)),
              f"{late} gaps > {LATE_S * 1000:.0f} ms (firmware layer); longest {dt.max() * 1000:.1f} ms"),
        Check("file", "reboots", level(reboots == 0), f"{reboots} board reboots"),
    ]


def quietest_window(set_df: pd.DataFrame) -> pd.DataFrame:
    """The QUIET_WINDOW_S stretch with the least rotation (the whole set if it is shorter)."""
    n = min(len(set_df), QUIET_WINDOW_S * HZ)
    mean_rotation = np.convolve(rotation_speed(set_df), np.ones(n) / n, mode="valid")
    start = int(np.argmin(mean_rotation))
    return set_df.iloc[start:start + n]


def check_set(set_df: pd.DataFrame, row) -> list[Check]:
    """Layers 2 and 3 for one set: sensor health, then label versus signal."""
    scope = f"{int(row['set_idx'])} {row['exercise']}"
    checks = []

    def add(name, level, detail):
        checks.append(Check(scope, name, level, detail))

    if len(set_df) < 2:
        add("samples", "FAIL", f"{len(set_df)} samples in the set window")
        return checks

    seconds = set_df["t"].iloc[-1] - set_df["t"].iloc[0]
    rotation = rotation_speed(set_df)
    is_still = row["exercise"] == "still"

    # The join: rows found by the shared clock should match what the receiver counted.
    off = abs(len(set_df) - int(row["n_samples"]))
    add("clock join", "PASS" if off <= 2 else "WARN", f"{len(set_df)} rows found, receiver counted {int(row['n_samples'])}")

    # Layer 2: sensor.
    clipped = ((set_df[ACCEL].abs() >= ACCEL_CLIP_G).any(axis=1) | (set_df[GYRO].abs() >= GYRO_CLIP_DPS).any(axis=1)).mean()
    if clipped == 0:
        level = "PASS"
    elif row["exercise"] in EXPECTED_CLIPPING or clipped <= MAX_CLIP_FRACTION:
        level = "WARN"
    else:
        level = "FAIL"
    add("clipping", level, f"{clipped:.1%} of samples at the range limit")

    # Layer 3: labels, plus calibration for still sets.
    add("duration", "PASS" if seconds >= MIN_SET_SECONDS else "FAIL", f"{seconds:.1f} s (min {MIN_SET_SECONDS:g})")
    if is_still:
        still_fraction = (rotation < STILL_DPS).mean()
        add("stillness", "PASS" if still_fraction >= MIN_STILL_FRACTION else "FAIL",
            f"{still_fraction:.0%} of the set under {STILL_DPS:g} deg/s (want ≥{MIN_STILL_FRACTION:.0%})")
        quiet = quietest_window(set_df)
        if rotation_speed(quiet).mean() >= STILL_DPS:
            add("calibration", "WARN", "skipped: no still stretch to measure gravity on")
        else:
            gravity = np.sqrt((quiet[ACCEL] ** 2).sum(axis=1)).mean()
            bias = quiet[GYRO].mean().abs().max()
            add("gravity", "PASS" if abs(gravity - 1) <= GRAVITY_TOLERANCE_G else "FAIL",
                f"{gravity:.3f} g at rest (want 1 ±{GRAVITY_TOLERANCE_G:g})")
            add("gyro offset", "PASS" if bias <= MAX_GYRO_BIAS_DPS else "WARN",
                f"{bias:.2f} deg/s largest axis at rest (want ≤{MAX_GYRO_BIAS_DPS:g})")
    else:
        moving_fraction = (rotation > MOVING_DPS).mean()
        add("movement", "PASS" if moving_fraction >= MIN_MOVING_FRACTION else "FAIL",
            f"{moving_fraction:.0%} of the set over {MOVING_DPS:g} deg/s (want ≥{MIN_MOVING_FRACTION:.0%})")
        reps = row["actual_reps"]
        if pd.notna(reps) and reps > 0:
            per_rep = seconds / reps
            level = "FAIL" if per_rep < MIN_S_PER_REP else "WARN" if per_rep > MAX_S_PER_REP else "PASS"
            add("rep pace", level, f"{per_rep:.2f} s per rep for {int(reps)} reps")
    return checks


def validate(df: pd.DataFrame, ann: pd.DataFrame) -> list[Check]:
    checks = check_timing(df)
    for _, row in ann[ann["status"] == "ok"].iterrows():
        checks += check_set(slice_set(df, row), row)
    return checks


def plot_session(df: pd.DataFrame, ann: pd.DataFrame, checks: list[Check], path: Path):
    """Accel and gyro over time, each set shaded by its worst check."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"PASS": "tab:green", "WARN": "tab:orange", "FAIL": "tab:red"}
    fig, (ax_a, ax_g) = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    for col in ACCEL:
        ax_a.plot(df["t"], df[col], lw=0.6, label=col)
    for col in GYRO:
        ax_g.plot(df["t"], df[col], lw=0.6, label=col)
    for _, row in ann[ann["status"] == "ok"].iterrows():
        scope = f"{int(row['set_idx'])} {row['exercise']}"
        level = worst(c.level for c in checks if c.scope == scope)
        t0, t1 = np.interp([row["start_host_t"], row["end_host_t"]], df["host_t"], df["t"])
        for ax in (ax_a, ax_g):
            ax.axvspan(t0, t1, color=colors[level], alpha=0.15)
        ax_a.text(t0, 1.02, f"{scope} ({level})", transform=ax_a.get_xaxis_transform(), fontsize=8)
    ax_a.set_ylabel("acceleration (g)")
    ax_g.set_ylabel("rotation (deg/s)")
    ax_g.set_xlabel("time (s, board clock)")
    ax_a.legend(loc="upper right", fontsize=8)
    ax_g.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def report(session: str, plot: bool = True) -> bool:
    """Print every check for one session; return True if nothing failed."""
    df, ann = load_session(session), load_annotations(session)
    checks = validate(df, ann)
    print(f"\nSession {session} · {len(df)} samples · {df['t'].iloc[-1]:.1f} s")
    for c in checks:
        print(f"  [{c.scope:<12}] {c.level:<4}  {c.name:<13} {c.detail}")

    verdict = worst(c.level for c in checks)
    warnings = sum(c.level == "WARN" for c in checks)
    failures = sum(c.level == "FAIL" for c in checks)
    line = f"Verdict: {verdict} ({failures} fail, {warnings} warn)"
    if plot:
        path = REPORTS_DIR / f"{session}_signals.png"
        plot_session(df, ann, checks, path)
        line += f" · plot: {path.relative_to(REPORTS_DIR.parent)}"
    print(line)
    return failures == 0


def main():
    parser = argparse.ArgumentParser(description="Check recorded RepSense sessions.")
    parser.add_argument("sessions", nargs="+", help="session ids, e.g. validation2")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    results = [report(s, plot=not args.no_plot) for s in args.sessions]
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
