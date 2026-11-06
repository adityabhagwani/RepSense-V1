"""Wrist-only rep counter: what a product band would run. Its mistakes are what the error model
(Phase 5) learns to predict, so it is meant to be simple and explainable, not perfect.

Two independent counters per set:
  angle: the forearm's angle from its hanging direction at the set start (the "ax humps", made
         mounting-proof by measuring from gravity instead of one board axis), smoothed. A hump
         that rises at least HUMP_SHARE of the set's typical hump is one rep, so half reps
         (about half as tall) are left out.
  gyro:  rotation about the set's main rotation axis. Each rep has one lowering swing; a pause
         halfway up splits the lifting swing but not the lowering one, so lowering swings are
         counted.
Their disagreement is an error-model feature (FEATURES.md #9).

Tuned only on dev sessions, then frozen (git tag counter-v1) before any held-out session.

counter-v3 (11-05) sizes an angle hump by how far it comes back down before the next real lift
starts, instead of how far it stands above the dips beside it. A lower-halfway rep (half-lower) and
the lift after it make two humps for one rep; height kept whichever of the two tops was taller, so
it sometimes kept the half-lower and the learned check then vetoed it, losing a real rep. The rep
that never came back down is the one dropped now: the next lift starts high.

    python -m src.counter dev1 dev1b dev2b
"""

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.dataio import ACCEL, GYRO, load_annotations, load_session, slice_set, with_true_reps
from src.labels import hang_axis, segment_angle, track_down
from src.sync import REPORTS_DIR, lowpass

SMOOTH_HZ = 3.0          # reps are under ~1 Hz; this removes jitter without blunting humps
START_S = 0.3            # sets start with the arm hanging: the first 0.3 s give "hanging"
MIN_SPACING_S = 0.8      # two reps closer than this are one
ANGLE_FLOOR_DEG = 15.0   # angle humps smaller than this aren't even candidates
GYRO_FLOOR_DPS = 20.0    # same for swings in rotation speed
HUMP_SHARE = 0.825       # a hump counts if at least this share of the set's typical hump. Middle of the
                         # best dev range (0.80-0.85, 95% exact after the axis fix); dev half reps reach
                         # 0.45-0.79, full reps 0.95+


EDGE_SHARE = 0.1        # a lift starts where it has risen this share of the way up from its dip, and ends
                        # where it is back within this share of the next dip. The dip itself is just the
                        # lowest point since the last lift, which after a rest can be seconds earlier


def lift_span(angle: np.ndarray, dip_before: int, top: int, dip_after: int, share: float = EDGE_SHARE) -> tuple[int, int]:
    """(start, end) sample of the lift that peaks at `top`: the last sample before it still within
    `share` of the way up from dip_before, and the first after it back within `share` of dip_after."""
    rise = angle[dip_before:top + 1] <= angle[dip_before] + share * (angle[top] - angle[dip_before])
    fall = angle[top:dip_after + 1] <= angle[dip_after] + share * (angle[top] - angle[dip_after])
    return dip_before + int(np.nonzero(rise)[0][-1]), top + int(np.nonzero(fall)[0][0])


RULES = ("v2", "v3")  # counter-v2: height above the dips beside it; counter-v3: fall to the next real lift


def fall_to_next_lift(x: np.ndarray, peaks: np.ndarray, share: float = HUMP_SHARE) -> np.ndarray:
    """How far each hump comes back down before the next real lift: the next hump whose top reaches at
    least `share` of the set's typical top. A small wobble on the way down, or a half lift after a full
    rep, is not a next lift, so it doesn't cut the fall short."""
    tall = np.nonzero(x[peaks] >= share * np.median(x[peaks]))[0]
    falls = []
    for k, top in enumerate(peaks):
        later = tall[tall > k]
        end = peaks[later[0]] if len(later) else len(x) - 1
        falls.append(x[top] - x[top:end + 1].min())
    return np.array(falls)


@dataclass
class Count:
    n: int                   # reps counted
    times: np.ndarray        # when each counted rep peaked (s from set start)
    heights: np.ndarray      # prominence of each counted hump
    candidates: int          # humps considered before the HUMP_SHARE cut
    typical: float           # the set's typical hump prominence
    near_cut: int            # candidates within ±20% of the cut: counts decided by a hair


def count_humps(x: np.ndarray, t: np.ndarray, floor: float, size: str = "prominence") -> Count:
    """Humps that rise at least HUMP_SHARE of the set's typical hump, at least MIN_SPACING_S apart.

    size="prominence": how far a hump stands above the dips beside it (angle humps).
    size="height": how far it rises above zero (rotation swings: a swing in the middle of a set
    sits between two opposite swings and would look twice as prominent as the last one).
    size="fall": how far it comes back down before the next real lift (angle humps, counter-v3).
    """
    fs = 1 / np.median(np.diff(t))
    spacing = max(1, int(MIN_SPACING_S * fs))
    if size == "height":
        peaks, props = find_peaks(x, height=floor, distance=spacing)
    else:
        peaks, props = find_peaks(x, prominence=floor, distance=spacing)
    if len(peaks) == 0:
        return Count(0, np.array([]), np.array([]), 0, 0.0, 0)
    prom = {"height": props.get("peak_heights"), "prominence": props.get("prominences")}.get(size)
    if size == "fall":
        prom = fall_to_next_lift(x, peaks)
    typical = float(np.median(prom))
    cut = HUMP_SHARE * typical
    keep = prom >= cut
    near = int(np.sum(np.abs(prom - cut) <= 0.2 * cut))
    return Count(int(keep.sum()), t[peaks[keep]] - t[0], prom[keep], len(peaks), typical, near)


def forearm_angle(seg: pd.DataFrame) -> np.ndarray:
    """Forearm angle from its hanging direction, degrees, smoothed. Wrist only."""
    t = seg["t"].to_numpy()
    down = track_down(t, seg[GYRO].to_numpy(), seg[ACCEL].to_numpy())
    start = np.arange(max(1, int(START_S / np.median(np.diff(t)))))
    return lowpass(t, segment_angle(down, hang_axis(down, start)), SMOOTH_HZ)


def lowering_rate(seg: pd.DataFrame) -> np.ndarray:
    """Rotation rate about the set's main axis, signed so lowering is positive, smoothed."""
    t, gyro = seg["t"].to_numpy(), seg[GYRO].to_numpy()
    axis = np.linalg.svd(gyro, full_matrices=False)[2][0]
    rate = lowpass(t, gyro @ axis, SMOOTH_HZ)
    # The set starts hanging, so the first big swing is a lift: lowering has the other sign.
    first = rate[np.argmax(np.abs(rate) > GYRO_FLOOR_DPS)] if np.any(np.abs(rate) > GYRO_FLOOR_DPS) else 1.0
    return -rate if first > 0 else rate


def count_set(seg: pd.DataFrame, rule: str = "v2") -> dict:
    t = seg["t"].to_numpy()
    by_angle = count_humps(forearm_angle(seg), t, ANGLE_FLOOR_DEG, size="prominence" if rule == "v2" else "fall")
    by_gyro = count_humps(lowering_rate(seg), t, GYRO_FLOOR_DPS, size="height")
    return {"angle": by_angle, "gyro": by_gyro}


def score(sessions: list[str]) -> pd.DataFrame:
    """One row per curl set: true reps, both counts, and the hump details later used as features."""
    rows = []
    for session in sessions:
        raw = load_session(session)
        ann = with_true_reps(load_annotations(session), session)
        for _, a in ann[(ann["exercise"] == "curl") & (ann["status"] == "ok")].iterrows():
            c = count_set(slice_set(raw, a))
            rows.append({
                "session": session, "set_idx": int(a["set_idx"]), "condition": a["condition"],
                "true_reps": int(a["true_reps"]), "angle_count": c["angle"].n, "gyro_count": c["gyro"].n,
                "angle_candidates": c["angle"].candidates, "angle_near_cut": c["angle"].near_cut,
                "angle_typical_deg": c["angle"].typical,
                "angle_weakest_share": (c["angle"].heights.min() / c["angle"].typical) if c["angle"].n else np.nan,
            })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, col: str) -> str:
    err = df[col] - df["true_reps"]
    return (f"{col:12s} exact {np.mean(err == 0):5.0%} | within one {np.mean(err.abs() <= 1):5.0%} | "
            f"mean reps off {err.abs().mean():.2f} | over/under {int((err > 0).sum())}/{int((err < 0).sum())}")


def main():
    parser = argparse.ArgumentParser(description="Count reps from the wrist alone and score against true reps.")
    parser.add_argument("sessions", nargs="+")
    args = parser.parse_args()
    df = score(args.sessions)
    out = REPORTS_DIR / ("counter_" + "_".join(args.sessions) + ".csv")
    df.to_csv(out, index=False)
    print(df[["session", "set_idx", "condition", "true_reps", "angle_count", "gyro_count"]].to_string(index=False))
    print(f"\n{len(df)} curl sets")
    print(summarize(df, "angle_count"))
    print(summarize(df, "gyro_count"))
    print(f"saved {out.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
