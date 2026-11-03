"""Teacher labels for curls: elbow bend from the wrist (Nano) and upper-arm (phone) IMUs.

Lab only. The product never runs this; the wrist-only student learns to predict its output
(see FEATURES.md, "Teacher labels and their wrist clues").

Think of the forearm as an airplane body with its nose toward the hand:
  roll  = twist (palm turning), about the forearm's own axis
  pitch = the curl, about the elbow hinge
  yaw   = sideways wobble

1. track_down follows "down" as each sensor sees it: the gyro carries fast motion, the
   accelerometer slowly corrects drift.
2. "Up" while the arm hangs at the set's rep bottoms is the segment's axis as its sensor sees it
   (the forearm's for the Nano, the upper arm's for the phone). Rep bottoms, not the moment Enter
   was pressed: the arm can be bent reaching for the laptop. Anything before the arm first
   settles there is pre-set jitter and is ignored (settle).
3. Segment angle = angle between that axis and "up" now: 0 hanging, 90 horizontal. Rolling
   around the axis can't change it, and it doesn't care which plane you curl in.
4. Elbow bend = forearm angle - upper-arm angle; drift = upper-arm angle. Reps = peaks.
5. Per lift, the Nano's rotation splits into roll (twist) and, around the forearm, the lift's
   main axis (pitch) plus what's left (yaw = wobble, sideways motion out of step with the
   curl). Plane tilt = how far the lift's axis sits from the set's usual curl axis (a rep
   that leans out as it rises). Tilted more than 45 deg, it was a sideways swing, not a curl,
   and isn't counted.
6. Hang offset (per gesture) = angle between the USB direction, which a product band would use
   as its forearm axis, and the measured hanging axis: mounting tip plus natural hang. It is
   the student-vs-teacher difference, measured.

    python -m src.labels dev1b
    python -m src.labels usbcheck --measure-usb   # Nano held still in the hand, USB up
"""

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.dataio import ACCEL, DATA_DIR, GYRO, load_annotations, load_session, slice_set
from src.sync import ALIGNED_DIR, REPORTS_DIR, UPPER_ARM

UA_ACCEL = [UPPER_ARM + c for c in ACCEL]
UA_GYRO = [UPPER_ARM + c for c in GYRO]
LABELS_DIR = DATA_DIR / "labels"
USB_AXIS_FILE = DATA_DIR / "usb_axis.json"

TRACK_TAU_S = 1.0          # the accelerometer pulls "down" back over ~1 s; faster motion is the gyro's job
START_S = 0.3              # sets start with the arm hanging: the first 0.3 s give each segment's axis
STILL_S = 0.5              # stillest stretch around a gesture = arm hanging
MIN_REP_DEG = 25.0         # a lift bends the elbow at least this much (half reps are ~60 deg)
MIN_REP_S = 0.6            # lift tops closer together than this are one lift
MAX_AXIS_OFF_DEG = 45.0    # lift axis this far from the set's usual curl axis: not a curl (palm twist moves it ±30)
MAX_SLIP_DEG = 10.0        # one sensor's hanging "up" moving this much more than the other's: a strap moved
MAX_HOLD_TILT_DEG = 25.0   # USB test: a reading further than this from every board axis is ambiguous
NEAR_END_DEG = 25.0        # samples this close to an end of the swing count as time spent there
SETTLED_DEG = 20.0         # the arm has settled once it first comes within this of its rep-bottom hang
SHOVE_DPS_PER_G = 200.0    # when finding a still moment, 0.05 g of push counts like 10 deg/s of turning


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle between unit vectors in degrees, row by row when given arrays of vectors."""
    return np.degrees(np.arccos(np.clip(np.sum(a * b, axis=-1), -1.0, 1.0)))


def axis_angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Degrees between two axes, ignoring sign: an axis and its reverse are the same axis."""
    return float(np.degrees(np.arccos(np.clip(abs(a @ b), 0.0, 1.0))))


def track_down(t: np.ndarray, gyro_dps: np.ndarray, accel: np.ndarray, tau_s: float = TRACK_TAU_S) -> np.ndarray:
    """"Down" as the sensor sees it, one unit vector per sample.

    Each sample: turn the estimate by the gyro, then nudge it toward the accelerometer by
    dt/(tau_s + dt), about 2% at 50 Hz with tau_s = 1 s. tau_s = inf follows the gyro alone.
    """
    felt = unit(-accel)  # at rest the accelerometer reads "up"
    rate = np.radians(gyro_dps)
    dt = np.diff(t, prepend=t[0])
    pull = dt / (tau_s + dt)  # share of the accelerometer's opinion taken each sample
    down = np.empty_like(felt)
    down[0] = felt[0]
    for i in range(1, len(t)):
        # The sensor turned by rate*dt, so a direction fixed in the room turns the other way.
        d = down[i - 1] - np.cross(rate[i], down[i - 1]) * dt[i]
        d = (1 - pull[i]) * d + pull[i] * felt[i]
        down[i] = d / np.linalg.norm(d)
    return down


def stillest(t: np.ndarray, gyro_dps: np.ndarray, seconds: float, accel: np.ndarray | None = None) -> slice:
    """The stretch of `seconds` with the least rotation, and with accel also the least shove.

    Not turning isn't enough: a sensor being pushed reads gravity plus the push. Still means
    the accelerometer reads exactly 1 g too.
    """
    n = max(1, min(len(t), int(round(seconds / np.median(np.diff(t))))))
    motion = np.linalg.norm(gyro_dps, axis=1)
    if accel is not None:
        motion = motion + SHOVE_DPS_PER_G * np.abs(np.linalg.norm(accel, axis=1) - 1)
    i = int(np.argmin(np.convolve(motion, np.ones(n) / n, mode="valid")))
    return slice(i, i + n)


def hang_axis(down: np.ndarray, idx) -> np.ndarray:
    """A segment's axis as its sensor sees it: "up" while the arm hangs (median, so a twitch can't drag it)."""
    return unit(-np.median(down[idx], axis=0))


def segment_angle(down: np.ndarray, axis: np.ndarray) -> np.ndarray:
    """Angle between the segment's axis and "up", per sample: 0 hanging, 90 horizontal."""
    return angle_deg(axis, -down)


def tops_and_bottoms(bend: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Lift tops (peaks) and the low points around them: one more bottom than tops."""
    tops, _ = find_peaks(bend, prominence=MIN_REP_DEG, distance=max(1, int(MIN_REP_S * fs)))
    edges = np.r_[0, tops, len(bend) - 1]
    bottoms = np.array([edges[k] + int(np.argmin(bend[edges[k]:edges[k + 1] + 1]))
                        for k in range(len(edges) - 1)])
    return tops, bottoms


def lift_motion(t: np.ndarray, gyro: np.ndarray, axis: np.ndarray, start: int, end: int):
    """(curl axis, twist deg, wobble deg) of one lift, from the Nano's gyro.

    Roll about the forearm axis, added up, is the twist. The rest of the rotation lies around
    the forearm: its main direction over the lift is the curl axis (pitch), and what's left
    over, added up, is the wobble (yaw).
    """
    span = slice(start, end + 1)
    dt = np.diff(t[span], prepend=t[start])
    roll = gyro[span] @ axis
    around = gyro[span] - np.outer(roll, axis)
    curl_axis = np.linalg.svd(around, full_matrices=False)[2][0]
    yaw = around @ np.cross(axis, curl_axis)
    return curl_axis, float(np.ptp(np.cumsum(roll * dt))), float(np.ptp(np.cumsum(yaw * dt)))


@dataclass
class Bend:
    t: np.ndarray
    bend: np.ndarray       # elbow bend from the set's starting hang, degrees
    upper: np.ndarray      # upper-arm angle from its hang, degrees (drift)
    fore_axis: np.ndarray  # forearm axis in the Nano's frame
    tops: np.ndarray
    bottoms: np.ndarray


def settle(t: np.ndarray, down: np.ndarray, gyro: np.ndarray, accel: np.ndarray, usb_axis=None):
    """(zero, first, bottoms): where the arm hangs between reps, when it first settles there, and the
    rep-bottom samples used.

    Zero is "up" at the set's rep bottoms (median, so a lower-halfway rep can't drag it), not the
    moment Enter was pressed: reaching for the laptop with the curling hand leaves the arm bent at
    the start of a set. Everything before the arm first comes back near zero is pre-set jitter.
    """
    fs = 1 / np.median(np.diff(t))
    # The forearm swings between two ends, hanging and the top: the two most different "down"
    # directions in the set (each averaged over its most extreme 5% of samples). Which end is the
    # bottom: where the USB (along the forearm, toward the elbow) points closer to straight up,
    # when the USB axis is known; otherwise the end the arm spends more time near (it rests there
    # between reps). On dev the time rule alone picked right in 20 of 21 sets.
    def far_from(ref):
        a = angle_deg(down, ref)
        return unit(down[a >= np.percentile(a, 95)].mean(axis=0))
    end1 = far_from(unit(down.mean(axis=0)))
    end2 = far_from(end1)
    if usb_axis is not None:
        usb = unit(np.asarray(usb_axis, float))
        bottom = end1 if angle_deg(usb, -end1) <= angle_deg(usb, -end2) else end2
    else:
        near1, near2 = ((angle_deg(down, e) < NEAR_END_DEG).sum() for e in (end1, end2))
        bottom = end1 if near1 >= near2 else end2
    rough = angle_deg(down, bottom)
    _, dips = tops_and_bottoms(rough, fs)
    bottoms = dips[1:-1] if len(dips) > 2 else dips  # the set's own edges may be pre- or post-set
    zero = hang_axis(down, bottoms)
    # The first time the forearm comes back near its hang. Moving or not: a lift can start straight
    # from the drop after reaching for the laptop, without the arm ever stopping at the bottom.
    settled = np.flatnonzero(segment_angle(down, zero) < SETTLED_DEG)
    return zero, int(settled[0]) if len(settled) else 0, bottoms


def elbow_bend(seg: pd.DataFrame, usb_axis=None) -> Bend:
    t = seg["t"].to_numpy()
    fs = 1 / np.median(np.diff(t))
    gyro, accel = seg[GYRO].to_numpy(), seg[ACCEL].to_numpy()
    fore_down = track_down(t, gyro, accel)
    upper_down = track_down(t, seg[UA_GYRO].to_numpy(), seg[UA_ACCEL].to_numpy())
    fore_axis, first, bottoms = settle(t, fore_down, gyro, accel, usb_axis)
    upper_axis = hang_axis(upper_down, bottoms)  # the upper arm's hang at the same moments
    upper = segment_angle(upper_down, upper_axis)
    bend = segment_angle(fore_down, fore_axis) - upper
    tops, bottoms = tops_and_bottoms(bend[first:], fs)  # nothing before the arm settles counts
    return Bend(t, bend, upper, fore_axis, tops + first, bottoms + first)


@dataclass
class SessionInfo:
    hang_offset_deg: list[float]   # USB direction vs measured hanging forearm axis, per gesture
    slip_deg: dict                 # hanging "up" change between first and last gesture, per sensor
    problems: list[str] = field(default_factory=list)


def session_checks(aligned: pd.DataFrame, ann: pd.DataFrame, usb_axis=None) -> SessionInfo:
    """Strap-moved check and hang offset, both from the still hanging moment of each gesture."""
    gestures = ann[(ann["exercise"] == "sync") & (ann["status"] == "ok")]
    if gestures.empty:
        return SessionInfo([], {}, ["no sync gesture: strap movement and hang offset can't be checked"])
    found = []
    for _, g in gestures.iterrows():
        seg = slice_set(aligned, g)
        quiet = stillest(seg["t"].to_numpy(), seg[GYRO].to_numpy(), STILL_S, seg[ACCEL].to_numpy())
        found.append((unit(seg[ACCEL].to_numpy()[quiet].mean(axis=0)),
                      unit(seg[UA_ACCEL].to_numpy()[quiet].mean(axis=0))))
    (f_first, u_first), (f_last, u_last) = found[0], found[-1]
    info = SessionInfo([], {"wrist": float(angle_deg(f_first, f_last)), "upper arm": float(angle_deg(u_first, u_last))})
    # The gesture keeps the elbow locked, so if the whole arm hangs at a new angle, both sensors' "up"
    # turns by the same amount. A strap that moved turns only its own sensor. Flag the difference.
    gap = info.slip_deg["upper arm"] - info.slip_deg["wrist"]
    if abs(gap) > MAX_SLIP_DEG:
        moved, other = ("upper arm", "wrist") if gap > 0 else ("wrist", "upper arm")
        info.problems.append(f"{moved}: hanging 'up' moved {info.slip_deg[moved]:.0f}° between gestures, "
                             f"{abs(gap):.0f}° more than the {other} (want ≤{MAX_SLIP_DEG:g}): a strap moved")
    if len(found) == 1:
        info.problems.append("only one gesture: strap movement can't be checked")
    if usb_axis is not None:
        usb = unit(np.asarray(usb_axis, float))
        info.hang_offset_deg = [float(angle_deg(usb, f_up)) for f_up, _ in found]
    return info


def label_set(seg: pd.DataFrame, row, usb_axis=None) -> tuple[pd.DataFrame, list[str]]:
    """One row per counted lift, plus problems found while labeling this set."""
    i = int(row["set_idx"])
    if seg[UA_GYRO + UA_ACCEL].isna().any().any():
        return pd.DataFrame(), [f"set {i}: the phone recording doesn't cover this set"]
    b = elbow_bend(seg, usb_axis)
    if len(b.tops) == 0:
        return pd.DataFrame(), [f"set {i}: no lifts found"]

    gyro = seg[GYRO].to_numpy()
    motion = [lift_motion(b.t, gyro, b.fore_axis, b.bottoms[k], b.bottoms[k + 1]) for k in range(len(b.tops))]
    axes = np.array([m[0] for m in motion])
    typical = np.linalg.svd(axes, full_matrices=False)[2][0]  # the set's usual curl axis, sign-free
    problems, keep = [], []
    tilts = [axis_angle_deg(axis, typical) for axis, _, _ in motion]
    for k, off in enumerate(tilts):
        if off <= MAX_AXIS_OFF_DEG:
            keep.append(k)
        else:
            problems.append(f"set {i}: lift at {b.t[b.tops[k]] - b.t[0]:.1f} s turned {off:.0f}° away from the "
                            f"set's curl axis (want ≤{MAX_AXIS_OFF_DEG:g}): not counted (sideways swing?)")
    if not keep:
        return pd.DataFrame(), problems + [f"set {i}: no lift looked like a curl"]

    set_top = np.median(b.bend[b.tops[keep]])
    set_bottom = np.median(b.bend[b.bottoms[keep]])
    reps = []
    for n, k in enumerate(keep, start=1):
        start, top, end = b.bottoms[k], b.tops[k], b.bottoms[k + 1]
        _, twist, wobble = motion[k]
        reps.append({
            "set_idx": i, "rep": n,
            "start_t": b.t[start], "top_t": b.t[top], "end_t": b.t[end],
            "rom_deg": b.bend[top] - b.bend[start],
            "top_gap_deg": set_top - b.bend[top],          # positive = stopped short of the usual top
            "start_gap_deg": b.bend[start] - set_bottom,   # positive = started above the usual bottom
            "end_gap_deg": b.bend[end] - set_bottom,       # positive = didn't lower all the way
            "bottom_bend_deg": b.bend[start],              # how far from the set's starting hang
            "drift_deg": np.ptp(b.upper[start:end + 1]),
            "twist_deg": twist, "wobble_deg": wobble,
            "plane_tilt_deg": tilts[k],  # this lift's curl axis vs the set's usual one (a leaning rep)
            "up_s": b.t[top] - b.t[start], "down_s": b.t[end] - b.t[top],
        })
    return pd.DataFrame(reps), problems


def label_session(aligned: pd.DataFrame, ann: pd.DataFrame, usb_axis=None) -> tuple[pd.DataFrame, SessionInfo]:
    info = session_checks(aligned, ann, usb_axis)
    tables = []
    for _, row in ann[(ann["exercise"] == "curl") & (ann["status"] == "ok")].iterrows():
        reps, problems = label_set(slice_set(aligned, row), row, usb_axis)
        info.problems += problems
        if not reps.empty:
            tables.append(reps)
    return (pd.concat(tables, ignore_index=True) if tables else pd.DataFrame()), info


def measure_usb_axis(df: pd.DataFrame) -> np.ndarray:
    """The Nano held still, USB up: the accelerometer reads "up" along the USB, give or take the hand."""
    quiet = stillest(df["t"].to_numpy(), df[GYRO].to_numpy(), 1.0, df[ACCEL].to_numpy())
    accel, rotation = df[ACCEL].to_numpy()[quiet], np.linalg.norm(df[GYRO].to_numpy()[quiet], axis=1)
    g = np.linalg.norm(accel.mean(axis=0))
    if not 0.95 <= g <= 1.05 or rotation.mean() > 5:
        raise ValueError(f"no still second found (|a| {g:.3f} g, rotation {rotation.mean():.1f} deg/s)")
    # The chip sits square on the board, so the USB axis is exactly one board axis. A hand-held
    # reading only has to pick which one; snapping removes however crooked the hand was.
    measured = unit(accel.mean(axis=0))
    snapped = np.zeros(3)
    i = int(np.argmax(np.abs(measured)))
    snapped[i] = np.sign(measured[i])
    tilt = float(angle_deg(measured, snapped))
    if tilt > MAX_HOLD_TILT_DEG:
        raise ValueError(f"reading is {tilt:.0f}° from the nearest board axis (want ≤{MAX_HOLD_TILT_DEG:g}): "
                         "hold the USB closer to straight up")
    print(f"measured {measured.round(3).tolist()}: held {tilt:.1f}° off vertical, snapped to {snapped.tolist()}")
    return snapped


def plot_sets(aligned, ann, reps, path, usb_axis=None):
    """Elbow bend and upper-arm angle per curl set, counted lift tops and starts marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sets = ann[(ann["exercise"] == "curl") & (ann["status"] == "ok")]
    fig, axes = plt.subplots(len(sets), 1, figsize=(13, 3 * len(sets)), squeeze=False)
    for ax, (_, row) in zip(axes[:, 0], sets.iterrows()):
        seg = slice_set(aligned, row)
        if seg[UA_GYRO].isna().any().any():
            ax.set_title(f"set {int(row['set_idx'])}: no phone data", fontsize=10)
            continue
        b = elbow_bend(seg, usb_axis)
        t0 = b.t[0]
        r = reps[reps["set_idx"] == row["set_idx"]] if not reps.empty else reps
        ax.plot(b.t - t0, b.bend, label="elbow bend")
        ax.plot(b.t - t0, b.upper, label="upper-arm angle (drift)")
        if not r.empty:
            ax.plot(r["top_t"] - t0, np.interp(r["top_t"], b.t, b.bend), "v", label="counted lift")
            ax.plot(r["start_t"] - t0, np.interp(r["start_t"], b.t, b.bend), "^", label="lift start")
        ax.set_title(f"set {int(row['set_idx'])} {row['condition']}: {len(r)} lifts counted, "
                     f"{row['actual_reps']} full reps typed", fontsize=10)
        ax.set_ylabel("degrees")
        ax.legend(fontsize=8, loc="upper right")
    axes[-1, 0].set_xlabel("s from set start")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Teacher labels (elbow bend) for a synced session.")
    parser.add_argument("session")
    parser.add_argument("--measure-usb", action="store_true",
                        help="measure the USB axis from a recording of the Nano held still, USB up")
    args = parser.parse_args()

    if args.measure_usb:
        try:
            # Only inside the first recorded set: lying flat on the table would be stiller than a hand.
            df, ann = load_session(args.session), load_annotations(args.session)
            ok = ann[ann["status"] == "ok"]
            axis = measure_usb_axis(slice_set(df, ok.iloc[0]) if not ok.empty else df)
        except (OSError, ValueError) as e:
            sys.exit(f"USB axis measurement failed: {e}")
        USB_AXIS_FILE.write_text(json.dumps({"axis": axis.round(4).tolist(), "measured_from": args.session,
                                             "date": date.today().isoformat()}, indent=2) + "\n")
        sys.exit(print(f"USB axis {axis.round(3).tolist()} saved to {USB_AXIS_FILE.relative_to(DATA_DIR.parent)}"))

    aligned_path = ALIGNED_DIR / f"{args.session}.csv"
    if not aligned_path.exists():
        sys.exit(f"{aligned_path} not found: run `python -m src.sync {args.session}` first")
    usb_axis = json.loads(USB_AXIS_FILE.read_text())["axis"] if USB_AXIS_FILE.exists() else None
    aligned, ann = pd.read_csv(aligned_path), load_annotations(args.session)
    reps, info = label_session(aligned, ann, usb_axis)

    LABELS_DIR.mkdir(parents=True, exist_ok=True)
    reps.to_csv(LABELS_DIR / f"{args.session}.csv", index=False)
    plot_path = REPORTS_DIR / f"{args.session}_elbow.png"
    plot_sets(aligned, ann, reps, plot_path, usb_axis)

    print(f"strap check (hanging 'up', first vs last gesture): wrist {info.slip_deg.get('wrist', np.nan):.1f}°, "
          f"upper arm {info.slip_deg.get('upper arm', np.nan):.1f}° (equal = the arm hung differently; "
          f"a gap over {MAX_SLIP_DEG:g}° = a strap moved)")
    if usb_axis is None:
        print("hang offset: USB axis not measured yet (record the Nano held still, USB up, then --measure-usb)")
    else:
        print("hang offset per gesture (USB vs hanging forearm): " + ", ".join(f"{e:.1f}°" for e in info.hang_offset_deg))
    cols = ["set_idx", "rep", "rom_deg", "end_gap_deg", "bottom_bend_deg", "drift_deg", "twist_deg",
            "wobble_deg", "plane_tilt_deg"]
    print(reps[cols].round(0).to_string(index=False) if not reps.empty else "no reps labeled")
    print(f"plot: {plot_path.relative_to(REPORTS_DIR.parent)}")
    for p in info.problems:
        print(f"  problem: {p}")
    sys.exit(1 if info.problems else 0)


if __name__ == "__main__":
    main()
