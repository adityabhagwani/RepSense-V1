"""Learned rep detector: decides, hump by hump, whether a full rep happened. Wrist only.

The rule counter (counter-v2) judges each hump on one number: its height relative to the set's
typical hump. Here every candidate hump gets a short resume from all six signals, and a logistic
regression trained on labeled dev humps decides "full rep" or "not": half reps and non-rep
movement (walking, re-racking, adjusting) both count as "not". Count = humps with p >= 0.5.
Every feature is wrist-only and measured from the hanging direction or as a magnitude, so how
the band sits doesn't change it (FEATURES.md rules).

Labels come from the scripts: in a scripted set whose hump count matches its scripted lifts,
hump n gets lift n's mark (half_lift / half_lower -> not a full rep, everything else -> full);
every hump in a distractor set is "not a rep". Sets whose counts don't match, and exhaustion
sets (no script), are skipped for training rather than guessed.

    python -m src.detector dev1 dev1b dev2b    # leave-one-session-out check, then train and save
"""

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.counter import ANGLE_FLOOR_DEG, MIN_SPACING_S, SMOOTH_HZ, count_set
from src.dataio import (ACCEL, GYRO, HALF_KINDS, load_annotations, load_session, movement_count,
                        parse_marked, slice_set, with_true_reps)
from src.labels import USB_AXIS_FILE, segment_angle, settle, track_down
from src.make_script import PROTOCOLS_DIR
from src.sync import lowpass

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODEL_PATH = MODELS_DIR / "detector.joblib"
GATE = 0.05      # combined: a rule-counted hump must look at least this likely to be a full rep to the
                 # learned model; set before seeing combined results, not tuned
MATCH_S = 0.6    # a rule hump and a learned hump this close in time are the same hump
MIN_REPS_GAP_S = 12.0  # min-reps-v1 (frozen 11-04): a combined-counted rep counts only if another one
                       # peaks within this many seconds. Bends that leak through the veto in
                       # distractors come one at a time; the longest gap between lifts in a set is 9.7 s
SCRIPTED = ("normal", "slow", "fast", "pause", "alternating", "partial", "drift")
FEATURES = [
    "height_deg",      # how far the hump rises above the dips beside it
    "height_share",    # that, relative to the set's typical hump (what the rule counter uses)
    "top_deg",         # forearm angle at the top, from hanging: walking never gets far
    "start_deg",       # forearm angle at the dip before: high after a "lower halfway"
    "end_deg",         # forearm angle at the dip after: high on a "lower halfway"
    "rise_s", "fall_s",
    "jitter_dps",      # shake riding on the hump: rotation speed minus its smoothed version
    "twist_deg",       # rotation around the forearm during the hump
    "one_axis_share",  # share of the hump's rotation about a single axis (a curl is a hinge)
    "rhythm_off",      # how far the gaps to the neighboring humps are from the set's usual gap
    "peak_speed_dps",
]


def hump_features(seg: pd.DataFrame, usb_axis=None) -> pd.DataFrame:
    """One row per candidate hump in a set: the features above plus t_top (s from set start)."""
    t, gyro, accel = seg["t"].to_numpy(), seg[GYRO].to_numpy(), seg[ACCEL].to_numpy()
    fs = 1 / np.median(np.diff(t))
    down = track_down(t, gyro, accel)
    axis, first, _ = settle(t, down, gyro, accel, usb_axis)  # zero = the hang at rep bottoms; drop pre-set jitter
    angle = lowpass(t, segment_angle(down, axis), SMOOTH_HZ)
    t0 = t[0]
    t, gyro, angle = t[first:], gyro[first:], angle[first:]
    peaks, props = find_peaks(angle, prominence=ANGLE_FLOOR_DEG, distance=max(1, int(MIN_SPACING_S * fs)))
    if len(peaks) == 0:
        return pd.DataFrame(columns=FEATURES + ["t_top"])

    edges = np.r_[0, peaks, len(t) - 1]
    dips = [edges[k] + int(np.argmin(angle[edges[k]:edges[k + 1] + 1])) for k in range(len(edges) - 1)]
    speed = np.linalg.norm(gyro, axis=1)
    shake = speed - lowpass(t, speed, SMOOTH_HZ)
    gaps = np.diff(t[peaks])
    usual_gap = np.median(gaps) if len(gaps) else np.nan

    rows = []
    for k, top in enumerate(peaks):
        a, b = dips[k], dips[k + 1]
        span = slice(a, b + 1)
        dt = np.diff(t[span], prepend=t[a])
        sv = np.linalg.svd(gyro[span], full_matrices=False)[1]
        neighbors = [g for g in (gaps[k - 1] if k > 0 else None, gaps[k] if k < len(gaps) else None) if g is not None]
        rows.append({
            "height_deg": props["prominences"][k],
            "top_deg": angle[top], "start_deg": angle[a], "end_deg": angle[b],
            "rise_s": t[top] - t[a], "fall_s": t[b] - t[top],
            "jitter_dps": float(np.sqrt(np.mean(shake[span] ** 2))),
            "twist_deg": float(np.ptp(np.cumsum((gyro[span] @ axis) * dt))),
            "one_axis_share": float(sv[0] ** 2 / (sv ** 2).sum()),
            "rhythm_off": float(np.mean([abs(g - usual_gap) / usual_gap for g in neighbors])) if neighbors else 0.0,
            "peak_speed_dps": float(speed[span].max()),
            "t_top": t[top] - t0,
        })
    df = pd.DataFrame(rows)
    df["height_share"] = df["height_deg"] / df["height_deg"].median()
    return df[FEATURES + ["t_top"]]


def label_humps(n_humps: int, condition: str, reps, marked: str) -> np.ndarray | None:
    """1 = full rep, 0 = not (half rep or not a rep), from the script. None = can't tell, skip."""
    if condition == "distractor":
        return np.zeros(n_humps, int)
    if condition not in SCRIPTED:
        return None  # exhaustion sets have no script to label from
    marks = parse_marked(marked)
    if n_humps != movement_count(int(reps), marks):
        return None  # a missing or extra hump would shift every label after it
    return np.array([0 if marks.get(m) in HALF_KINDS else 1 for m in range(1, n_humps + 1)])


def build_dataset(sessions: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(humps, sets): every candidate hump of every curl and distractor set, and one row per set."""
    usb_axis = json.loads(USB_AXIS_FILE.read_text())["axis"] if USB_AXIS_FILE.exists() else None
    humps, sets = [], []
    for session in sessions:
        raw = load_session(session)
        ann = with_true_reps(load_annotations(session), session)
        script = pd.read_csv(PROTOCOLS_DIR / f"{session}.csv", dtype=str, keep_default_na=False)
        for _, a in ann[ann["status"] == "ok"].iterrows():
            if a["exercise"] != "curl" and a["condition"] != "distractor":
                continue
            seg = slice_set(raw, a)
            f = hump_features(seg, usb_axis)
            row = script.iloc[int(a["set_idx"]) - 1]
            labels = label_humps(len(f), a["condition"], row["reps"], row["marked"])
            f["label"] = labels if labels is not None else np.nan
            f["session"], f["set_idx"] = session, int(a["set_idx"])
            humps.append(f)
            true = 0 if a["condition"] == "distractor" else int(a["true_reps"])
            rule = count_set(seg)["angle"]
            sets.append({"session": session, "set_idx": int(a["set_idx"]), "condition": a["condition"],
                         "exercise": a["exercise"], "true_reps": true, "rule_count": rule.n,
                         "rule_times": rule.times})
    return pd.concat(humps, ignore_index=True), pd.DataFrame(sets)


def train(humps: pd.DataFrame):
    labeled = humps.dropna(subset=["label"])
    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000))
    return model.fit(labeled[FEATURES], labeled["label"].astype(int))


def count_with(model, humps: pd.DataFrame) -> pd.Series:
    """Full reps per (session, set_idx): humps the model gives p >= 0.5."""
    p = model.predict_proba(humps[FEATURES])[:, 1] if len(humps) else np.array([])
    return pd.Series(p >= 0.5, index=humps.index).groupby([humps["session"], humps["set_idx"]]).sum()


def combined_times(rule_times, tops: np.ndarray, p: np.ndarray) -> np.ndarray:
    """The rule decides full vs half (height); the learned model vetoes humps that aren't curls at all.

    Each rule-counted hump is matched to the learned hump nearest in time; it counts only if one is
    within MATCH_S and the learned model gives it p >= GATE. Returns the times of the humps kept.
    """
    rule_times = np.asarray(rule_times, float)
    if len(tops) == 0:
        return rule_times[:0]
    nearest = [int(np.argmin(np.abs(tops - x))) for x in rule_times]
    keep = [abs(tops[k] - x) <= MATCH_S and p[k] >= GATE for k, x in zip(nearest, rule_times)]
    return rule_times[np.array(keep, bool)]


def combined_count(rule_times, tops: np.ndarray, p: np.ndarray) -> int:
    return len(combined_times(rule_times, tops, p))


def min_reps(times) -> np.ndarray:
    """min-reps-v1: the reps (times) that have another rep within MIN_REPS_GAP_S, on either side."""
    times = np.sort(np.asarray(times, float))
    if len(times) < 2:
        return times[:0]
    close = np.diff(times) <= MIN_REPS_GAP_S
    return times[np.r_[False, close] | np.r_[close, False]]


def add_counts(sets: pd.DataFrame, humps: pd.DataFrame, p: pd.Series) -> pd.DataFrame:
    """learned_count, combined_count and min_reps_count per set, given a probability for every hump."""
    out = sets.copy()
    for col in ("learned_count", "combined_count", "min_reps_count"):
        out[col] = 0
    for k, r in out.iterrows():
        mine = (humps["session"] == r["session"]) & (humps["set_idx"] == r["set_idx"])
        tops, probs = humps.loc[mine, "t_top"].to_numpy(), p[mine].to_numpy()
        kept = combined_times(r["rule_times"], tops, probs)
        out.loc[k, "learned_count"] = int((probs >= 0.5).sum())
        out.loc[k, "combined_count"] = len(kept)
        out.loc[k, "min_reps_count"] = len(min_reps(kept))
    return out


def leave_one_session_out(humps: pd.DataFrame, sets: pd.DataFrame) -> pd.DataFrame:
    """Each session's sets counted with a model trained only on the other sessions."""
    p = pd.Series(np.nan, index=humps.index)
    for session in sets["session"].unique():
        model = train(humps[humps["session"] != session])
        mine = humps["session"] == session
        if mine.any():
            p[mine] = model.predict_proba(humps.loc[mine, FEATURES])[:, 1]
    return add_counts(sets, humps, p)


def summarize(df: pd.DataFrame, col: str) -> str:
    curls = df[df["condition"] != "distractor"]
    err = curls[col] - curls["true_reps"]
    fake = int(df.loc[df["condition"] == "distractor", col].sum())
    return (f"{col:14s} curl sets exact {np.mean(err == 0):4.0%} | within one {np.mean(err.abs() <= 1):4.0%} | "
            f"avg off {err.abs().mean():.2f} | fake reps in distractors {fake}")


def main():
    parser = argparse.ArgumentParser(description="Train and check the learned rep detector, or evaluate frozen versions.")
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--evaluate", action="store_true",
                        help="count these sessions with the frozen rule, learned, combined and min-reps versions "
                             "(no training)")
    args = parser.parse_args()
    cols = ["session", "set_idx", "condition", "true_reps", "rule_count", "learned_count", "combined_count",
            "min_reps_count"]

    humps, sets = build_dataset(args.sessions)
    if args.evaluate:
        model = joblib.load(MODEL_PATH)
        p = pd.Series(model.predict_proba(humps[FEATURES])[:, 1] if len(humps) else [], index=humps.index)
        result = add_counts(sets, humps, p)
        print(result[cols].to_string(index=False))
        for col in cols[4:]:
            print(summarize(result, col))
        return

    labeled = humps.dropna(subset=["label"])
    print(f"{len(humps)} candidate humps; {len(labeled)} labeled ({int(labeled['label'].sum())} full rep, "
          f"{int((labeled['label'] == 0).sum())} not); {len(sets)} sets")
    result = leave_one_session_out(humps, sets)
    print(result[cols].to_string(index=False))
    print(summarize(result, "rule_count") + "   (rule cut tuned on all of dev: optimistic)")
    print(summarize(result, "learned_count") + "   (each session held out)")
    print(summarize(result, "combined_count") + "   (learned part held out)")
    print(summarize(result, "min_reps_count") + "   (learned part held out)")

    model = train(humps)
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    weights = pd.Series(model[-1].coef_[0], index=FEATURES).sort_values(key=abs, ascending=False)
    print(f"saved {MODEL_PATH.relative_to(MODELS_DIR.parent)}; weights (+ pushes toward 'full rep'):")
    print(weights.round(2).to_string())


if __name__ == "__main__":
    main()
