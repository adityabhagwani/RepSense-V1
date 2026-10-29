"""Put the phone (upper arm) on the Nano's (wrist) clock and sample it at the Nano's times.

1. Nano board clock -> Unix time: a line fit of host_t against t averages out USB jitter.
2. Coarse offset: the phone and Mac clocks are both internet-synced, so the search starts at 0.
3. Fine offset at each sync gesture: cross-correlate rotation speed, which is the same on both
   sensors during a straight-arm swing whatever each one's orientation, within ±SEARCH_S.
4. Drift: the offsets at the first and last gesture define a line.
5. Resample: zero-delay low-pass of the ~100 Hz phone signals, then interpolate at each Nano time.

Convention: offset is what you add to a phone time to get the matching Nano time.

    python -m src.sync dev1
"""

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

from src.dataio import ACCEL, DATA_DIR, GYRO, load_annotations, load_session, rotation_speed

SEARCH_S = 2.0         # wall clocks should agree far better than this; reps repeat every ~1.5 s
STEP_S = 0.01          # search resolution, refined below one step by a parabola fit
PAD_S = 1.0            # widen each gesture window so the burst's edges are included
CUTOFF_HZ = 20.0       # under the 25 Hz limit of the 50 Hz Nano grid
MIN_STRENGTH = 0.8     # a weaker match means the gesture wasn't clean (bent elbow, too slow)
MAX_DRIFT_PPM = 200.0  # real clock crystals drift far less; more suggests a bug
UPPER_ARM = "ua_"      # prefix for phone columns in the aligned table

ALIGNED_DIR = DATA_DIR / "aligned"
REPORTS_DIR = Path(__file__).resolve().parent.parent / "reports"


@dataclass
class SyncReport:
    gesture_t: list[float]   # Unix time (Nano clock) at the middle of each gesture
    offsets_s: list[float]   # phone -> Nano shift found at each gesture
    strengths: list[float]   # correlation at that shift (1 = identical shape)
    drift_ppm: float         # how fast the offset changes, in parts per million
    problems: list[str]


def board_to_unix(nano: pd.DataFrame) -> np.ndarray:
    """Nano sample times on the Mac's Unix clock, from a line fit host_t = a + b*t."""
    t = nano["t"].to_numpy()
    b, a = np.polyfit(t, nano["host_t"].to_numpy(), 1)
    return a + b * t


def lowpass(t: np.ndarray, values: np.ndarray, cutoff_hz: float = CUTOFF_HZ) -> np.ndarray:
    """Zero-delay low-pass: filter forward, then backward, so the delays cancel."""
    fs = 1 / np.median(np.diff(t))
    b, a = butter(4, cutoff_hz / (fs / 2))
    return filtfilt(b, a, values, axis=0)


def _correlation(x: np.ndarray, y: np.ndarray) -> float:
    if np.isnan(y).any() or x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def find_offset(t_ref, x_ref, t_other, x_other, window, guess=0.0, search_s=SEARCH_S, step_s=STEP_S):
    """Shift that lines `other` up with `ref` inside `window` (ref clock), and the match strength.

    Tries every shift in guess ± search_s and keeps the best correlation, then fits a
    parabola through the best score and its two neighbours for sub-step precision.
    """
    grid = np.arange(window[0], window[1], step_s)
    ref = np.interp(grid, t_ref, x_ref)
    shifts = guess + np.arange(-search_s, search_s + step_s / 2, step_s)
    scores = np.array([
        _correlation(ref, np.interp(grid - s, t_other, x_other, left=np.nan, right=np.nan)) for s in shifts
    ])
    if np.isnan(scores).all():
        return np.nan, np.nan
    best = int(np.nanargmax(scores))
    shift = shifts[best]
    if 0 < best < len(shifts) - 1 and np.isfinite(scores[best - 1:best + 2]).all():
        y0, y1, y2 = scores[best - 1:best + 2]
        curvature = y0 - 2 * y1 + y2
        if curvature < 0:
            shift += 0.5 * (y0 - y2) / curvature * step_s
    return float(shift), float(scores[best])


def align(nano: pd.DataFrame, phone_accel: pd.DataFrame, phone_gyro: pd.DataFrame,
          ann: pd.DataFrame) -> tuple[pd.DataFrame, SyncReport]:
    """The Nano table plus upper-arm columns (ua_ax ... ua_gz) sampled at the same instants."""
    gestures = ann[(ann["exercise"] == "sync") & (ann["status"] == "ok")]
    if gestures.empty:
        raise ValueError("no sync gesture in the annotations; scripts need a 'sync' row at the start and end")

    nano_unix = board_to_unix(nano)
    nano_speed = rotation_speed(nano)
    gyro_t = phone_gyro["unix_t"].to_numpy()
    gyro_filtered = lowpass(gyro_t, phone_gyro[GYRO].to_numpy())
    phone_speed = np.linalg.norm(gyro_filtered, axis=1)

    times, offsets, strengths, problems = [], [], [], []
    for _, g in gestures.iterrows():
        window = (g["start_host_t"] - PAD_S, g["end_host_t"] + PAD_S)
        offset, strength = find_offset(nano_unix, nano_speed, gyro_t, phone_speed, window)
        times.append((window[0] + window[1]) / 2)
        offsets.append(offset)
        strengths.append(strength)
        label = f"gesture at set {int(g['set_idx'])}"
        if np.isnan(offset):
            problems.append(f"{label}: phone recording doesn't cover it")
        elif strength < MIN_STRENGTH:
            problems.append(f"{label}: weak match {strength:.2f} (want ≥{MIN_STRENGTH:g}); was the elbow straight?")

    usable = [i for i, (o, s) in enumerate(zip(offsets, strengths)) if np.isfinite(o) and s >= MIN_STRENGTH]
    if not usable:
        raise ValueError(f"no usable sync gesture: {problems}")
    first, last = usable[0], usable[-1]
    if first == last:
        slope = 0.0
        problems.append("only one usable gesture: drift is not corrected")
    else:
        slope = (offsets[last] - offsets[first]) / (times[last] - times[first])
    drift_ppm = slope * 1e6
    if abs(drift_ppm) > MAX_DRIFT_PPM:
        problems.append(f"drift {drift_ppm:.0f} ppm is implausible (want ≤{MAX_DRIFT_PPM:g}); check the gestures")

    # Phone time for each Nano sample: nano = phone + offset(t), so phone = nano - offset(t).
    query = nano_unix - (offsets[first] + slope * (nano_unix - times[first]))
    out = nano.copy()
    out["unix_t"] = nano_unix
    accel_t = phone_accel["unix_t"].to_numpy()
    accel_filtered = lowpass(accel_t, phone_accel[ACCEL].to_numpy())
    for t_src, filtered, cols in [(accel_t, accel_filtered, ACCEL), (gyro_t, gyro_filtered, GYRO)]:
        for i, col in enumerate(cols):
            out[UPPER_ARM + col] = np.interp(query, t_src, filtered[:, i], left=np.nan, right=np.nan)

    return out, SyncReport(times, offsets, strengths, drift_ppm, problems)


def plot_gestures(nano_unix, nano_speed, aligned, report: SyncReport, path: Path):
    """Rotation speed of both sensors at each gesture: wall clocks only (top) vs synced (bottom)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(report.gesture_t)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 5), squeeze=False)
    ua_speed = np.sqrt((aligned[[UPPER_ARM + c for c in GYRO]] ** 2).sum(axis=1)).to_numpy()
    for j, (center, offset) in enumerate(zip(report.gesture_t, report.offsets_s)):
        near = np.abs(nano_unix - center) < 4
        for row, (shift, title) in enumerate([(offset, "wall clocks only"), (0.0, "after sync")]):
            ax = axes[row, j]
            ax.plot(nano_unix[near] - center, nano_speed[near], lw=1, label="wrist (Nano)")
            # Unsynced, each phone sample sits at its own timestamp: nano time - offset.
            ax.plot(nano_unix[near] - center - shift, ua_speed[near], lw=1, label="upper arm (phone)")
            ax.set_title(f"gesture {j + 1}: {title}", fontsize=9)
            ax.set_xlabel("s from gesture middle")
        axes[0, j].legend(fontsize=8)
    axes[0, 0].set_ylabel("rotation speed (deg/s)")
    axes[1, 0].set_ylabel("rotation speed (deg/s)")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    from src.phone import load_phyphox, phone_dir  # only the command line reads phyphox files

    parser = argparse.ArgumentParser(description="Align the phone (upper arm) to the Nano (wrist).")
    parser.add_argument("session")
    args = parser.parse_args()

    nano, ann = load_session(args.session), load_annotations(args.session)
    phone_accel, phone_gyro = load_phyphox(phone_dir(args.session))
    try:
        aligned, report = align(nano, phone_accel, phone_gyro, ann)
    except ValueError as e:
        sys.exit(f"Sync failed for {args.session}: {e}")

    ALIGNED_DIR.mkdir(parents=True, exist_ok=True)
    aligned.to_csv(ALIGNED_DIR / f"{args.session}.csv", index=False)
    (ALIGNED_DIR / f"{args.session}_sync.json").write_text(json.dumps(asdict(report), indent=2))
    plot_path = REPORTS_DIR / f"{args.session}_sync.png"
    plot_gestures(aligned["unix_t"].to_numpy(), rotation_speed(nano), aligned, report, plot_path)

    for t, o, s in zip(report.gesture_t, report.offsets_s, report.strengths):
        print(f"gesture at {t - report.gesture_t[0]:7.1f} s: offset {o * 1000:+8.1f} ms, match {s:.3f}")
    print(f"drift {report.drift_ppm:+.1f} ppm · plot: {plot_path.relative_to(REPORTS_DIR.parent)}")
    for p in report.problems:
        print(f"  problem: {p}")
    sys.exit(1 if report.problems else 0)


if __name__ == "__main__":
    main()
