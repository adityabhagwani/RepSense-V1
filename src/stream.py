"""Continuous replay: find the sets in a whole recording without the Enter presses.

The counters so far were handed each set's start and end. A band on someone's wrist gets one long
stream. This finds the sets itself, the way you'd spot one across a gym: one big arm bend could be
anything (grabbing a bottle), two within BOX_GAP_S of each other start a set, and BOX_GAP_S
without one ends it. Each box is then counted by the frozen counters exactly as an annotated set
would be, so the only new thing being tested is where the boxes go.

Lifts are humps in the forearm angle measured from the board's USB direction (USB toward the
elbow, so it points up while the arm hangs; data/usb_axis.json). That zero needs no set start.
The board's mounting offset (~30 deg on dev) shifts the angle but barely changes hump heights,
which is all a lift needs.

Tuned on dev1b and dev2b only, then frozen (git tag stream-v1) before scoring p2, p4 and p5.

    python -m src.stream dev1b dev2b --plot
"""

import argparse
import json
import re
from dataclasses import dataclass, field

import joblib
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.counter import MIN_SPACING_S, SMOOTH_HZ, count_set, lift_span
from src.dataio import ACCEL, DATA_DIR, GYRO, load_annotations, load_session, slice_set, with_true_reps
from src.detector import (FEATURES, MODEL_PATHS, add_counts, build_dataset, combined_times, hump_features,
                          min_reps, train)
from src.labels import USB_AXIS_FILE, segment_angle, track_down
from src.sync import REPORTS_DIR, lowpass

LIFT_MIN_DEG = 50.0  # a hump this tall is a lift. Dev: curl-set humps typically 120-138 deg tall,
                     # walking and strap fiddling 24-37; at 50 every dev1b curl hump passes
BOX_GAP_S = 12.0     # lifts whose tops are this close belong to one set. Same 12 s as min-reps-v1:
                     # the longest lift-to-lift time in any dev or held-out set is 9.7 s
PAD_S = 2.0          # a box starts this long before its first lift (and ends this long after its
                     # last), so the arm hangs at its start, like after an Enter press
DEV_SESSIONS = ("dev1", "dev1b", "dev2b")  # what the frozen learned model was trained on
IDLE_AFTER = {"p3_s1": 1793403434.12}     # sessions.csv: the Nano kept logging an idle arm after this host_t
UNRESOLVED = {("p3_s1", 2)}                # alternating set whose true count was never settled (typed 9)
SESSIONS_CSV = DATA_DIR / "sessions.csv"


@dataclass
class Lift:
    start: float   # the dip before, on the clock passed to find_lifts
    top: float
    end: float     # the dip after
    height: float  # degrees above the higher of the two dips


@dataclass
class Box:
    start: float
    end: float
    lifts: list[Lift] = field(default_factory=list)


def stream_angle(df: pd.DataFrame, usb_axis) -> np.ndarray:
    """Forearm angle from the USB direction over a whole recording, degrees, smoothed. Wrist only."""
    t = df["t"].to_numpy()
    down = track_down(t, df[GYRO].to_numpy(), df[ACCEL].to_numpy())
    return lowpass(t, segment_angle(down, np.asarray(usb_axis, float)), SMOOTH_HZ)


def find_lifts(t: np.ndarray, angle: np.ndarray, clock: np.ndarray | None = None) -> list[Lift]:
    """Humps at least LIFT_MIN_DEG tall and MIN_SPACING_S apart, with the dips on either side.

    `t` sets the sample rate; times are reported on `clock` (default `t`), e.g. host_t so boxes
    line up with the annotations.
    """
    clock = t if clock is None else clock
    fs = 1 / np.median(np.diff(t))
    tops, props = find_peaks(angle, prominence=LIFT_MIN_DEG, distance=max(1, int(MIN_SPACING_S * fs)))
    edges = np.r_[0, tops, len(t) - 1]
    dips = [edges[k] + int(np.argmin(angle[edges[k]:edges[k + 1] + 1])) for k in range(len(edges) - 1)]
    lifts = []
    for k, top in enumerate(tops):
        start, end = lift_span(angle, dips[k], top, dips[k + 1])
        lifts.append(Lift(clock[start], clock[top], clock[end], float(props["prominences"][k])))
    return lifts


def draw_boxes(lifts: list[Lift]) -> list[Box]:
    """Lifts whose tops are at most BOX_GAP_S apart form a group; groups of 2+ become boxes.

    A lone lift makes no box. Padding never reaches into the next box: where two padded boxes
    would overlap, they meet halfway.
    """
    groups: list[list[Lift]] = []
    for lift in lifts:
        if groups and lift.top - groups[-1][-1].top <= BOX_GAP_S:
            groups[-1].append(lift)
        else:
            groups.append([lift])
    boxes = [Box(g[0].start - PAD_S, g[-1].end + PAD_S, g) for g in groups if len(g) >= 2]
    for a, b in zip(boxes, boxes[1:]):
        if a.end > b.start:
            a.end = b.start = (a.lifts[-1].end + b.lifts[0].start) / 2
    return boxes


def model_for(session: str, span: str = "dip"):
    """The saved learned model (span "dip" = detector-v1, "lift" = detector-v2), except on its own
    training sessions: there, one trained on the other dev sessions, so dev numbers have the learned
    part held out (as in the dev scores)."""
    if session in DEV_SESSIONS:
        return train(build_dataset([s for s in DEV_SESSIONS if s != session], span)[0])
    return joblib.load(MODEL_PATHS[span])


def count_box(seg: pd.DataFrame, model, usb_axis, span: str = "dip") -> dict:
    """The frozen counters on one box, exactly as on an annotated set (detector.build_dataset), then
    min-reps-v1. The kept reps' times come back on the host clock."""
    rule = count_set(seg)["angle"]
    humps = hump_features(seg, usb_axis, span)
    p = model.predict_proba(humps[FEATURES])[:, 1] if len(humps) else np.array([])
    kept = combined_times(rule.times, humps["t_top"].to_numpy(), p)
    reps = min_reps(kept)
    return {"rule": rule.n, "learned": int((p >= 0.5).sum()), "combined": len(kept), "min_reps": len(reps),
            "rep_times": seg["host_t"].iloc[0] + reps}


def set_kind(row) -> str:
    if row["status"] != "ok":
        return "discarded"  # redone sets: real movement, but no settled count
    if row["condition"] == "gesture":
        return "swing"
    return "distractor" if row["condition"] == "distractor" else "curl"


def match_boxes(boxes: list[Box], ann: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per box: which annotated sets it overlaps; per curl set: how many boxes overlap it)."""
    ann = ann.assign(kind=ann.apply(set_kind, axis=1))
    rows = []
    for b in boxes:
        hit = ann[(ann["start_host_t"] <= b.end) & (ann["end_host_t"] >= b.start)]
        kinds = set(hit["kind"])
        what = ("curl" if "curl" in kinds else "distractor" if "distractor" in kinds else
                "swing" if "swing" in kinds else "discarded" if "discarded" in kinds else "nothing")
        rows.append({"start": b.start, "end": b.end, "lifts": len(b.lifts), "on": what,
                     "sets": " ".join(f"{r.set_idx}:{r.kind}" for r in hit.itertuples())})
    per_box = pd.DataFrame(rows, columns=["start", "end", "lifts", "on", "sets"])
    curls = ann[ann["kind"] == "curl"]
    box_ids = [[k for k, b in enumerate(boxes) if b.start <= r.end_host_t and b.end >= r.start_host_t]
               for r in curls.itertuples()]
    shared = [any(((curls["start_host_t"] <= boxes[k].end) & (curls["end_host_t"] >= boxes[k].start)).sum() > 1
                  for k in ids) for ids in box_ids]
    per_set = curls[["set_idx", "condition"]].assign(box_ids=box_ids, merged=shared)
    n = per_set["box_ids"].map(len)
    per_set["result"] = np.select([per_set["merged"], n == 0, n > 1], ["merged", "missed", "split"], "found once")
    return per_box, per_set


def replay(session: str, usb_axis) -> tuple[pd.DataFrame, np.ndarray, list[Lift], list[Box]]:
    df = load_session(session)
    angle = stream_angle(df, usb_axis)
    lifts = find_lifts(df["t"].to_numpy(), angle, df["host_t"].to_numpy())
    return df, angle, lifts, draw_boxes(lifts)


def flagged_sets(session: str, path=SESSIONS_CSV) -> set[int]:
    """Set numbers the session's notes say to score separately ("Flagged sets 5, 11 and 14 scored
    separately"), decided before evaluating."""
    notes = pd.read_csv(path).set_index("session")["notes"]
    m = re.search(r"Flagged sets? ([\d, and]+?) scored separately", str(notes.get(session, "")))
    return {int(x) for x in re.findall(r"\d+", m.group(1))} if m else set()


def place_reps(reps: pd.DataFrame, ann: pd.DataFrame, idle_after: float | None = None) -> pd.Series:
    """Where each counted rep (columns t, box_on) fell: inside an annotated curl, distractor, swing
    or discarded set; in the idle stretch; at a curl set's edge (outside its annotated window but in a
    box drawn around it); or in the rest between sets."""
    ann = ann.assign(kind=ann.apply(set_kind, axis=1))
    out = []
    for r in reps.itertuples():
        inside = ann[(ann["start_host_t"] <= r.t) & (ann["end_host_t"] >= r.t)]
        if len(inside):
            out.append(inside["kind"].iloc[0])
        elif idle_after is not None and r.t > idle_after:
            out.append("idle")
        else:
            out.append("edge" if r.box_on == "curl" else "rest")
    return pd.Series(out, index=reps.index, dtype=object)


def score(session: str, usb_axis, span: str = "dip") -> dict:
    """Find and count the sets in one whole recording, then line them up with the annotations and
    with the same counters run on the Enter-press boxes."""
    df, angle, lifts, boxes = replay(session, usb_axis)
    ann = with_true_reps(load_annotations(session), session)
    model = model_for(session, span)
    per_box, per_set = match_boxes(boxes, ann)
    counts = [count_box(slice_set(df, {"start_host_t": b.start, "end_host_t": b.end}), model, usb_axis, span)
              for b in boxes]
    for col in ("rule", "learned", "combined", "min_reps"):
        per_box[col] = [c[col] for c in counts]
    reps = pd.DataFrame([{"t": x, "box": k, "box_on": per_box["on"][k]} for k, c in enumerate(counts)
                         for x in c["rep_times"]], columns=["t", "box", "box_on"])
    reps["where"] = place_reps(reps, ann, IDLE_AFTER.get(session))

    humps, sets = build_dataset([session], span)
    enter = add_counts(sets, humps, pd.Series(model.predict_proba(humps[FEATURES])[:, 1], index=humps.index))
    per_set = per_set.merge(ann[["set_idx", "true_reps", "start_host_t", "end_host_t"]], on="set_idx").merge(
        enter[["set_idx", "min_reps_count"]].rename(columns={"min_reps_count": "enter"}), on="set_idx")
    per_set["stream"] = per_set["box_ids"].map(lambda ids: int(per_box.loc[ids, "min_reps"].sum()))
    per_set["edge_reps"] = per_set["box_ids"].map(lambda ids: int((reps["box"].isin(ids) & (reps["where"] == "edge")).sum()))
    once = per_set["result"] == "found once"
    per_set["start_off"] = np.where(once, [boxes[i[0]].start if i else np.nan for i in per_set["box_ids"]], np.nan) \
        - per_set["start_host_t"]
    per_set["end_off"] = np.where(once, [boxes[i[-1]].end if i else np.nan for i in per_set["box_ids"]], np.nan) \
        - per_set["end_host_t"]
    flagged = flagged_sets(session)
    per_set["group"] = ["unresolved" if (session, i) in UNRESOLVED else "flagged" if i in flagged else "clean"
                        for i in per_set["set_idx"]]

    host = df["host_t"].to_numpy()
    idle_min = (host[-1] - IDLE_AFTER[session]) / 60 if session in IDLE_AFTER else 0.0
    in_sets_min = float((ann["end_host_t"] - ann["start_host_t"]).sum()) / 60
    return {"session": session, "df": df, "angle": angle, "lifts": lifts, "boxes": boxes, "ann": ann,
            "per_box": per_box, "per_set": per_set, "reps": reps, "idle_min": idle_min,
            "rest_min": (host[-1] - host[0]) / 60 - in_sets_min - idle_min,
            "enter_distractor_reps": int(enter.loc[enter["condition"] == "distractor", "min_reps_count"].sum())}


def summary_row(r: dict) -> dict:
    s, reps = r["per_set"], r["reps"]
    clean, flagged = s[s["group"] == "clean"], s[s["group"] == "flagged"]
    found = s[s["result"] == "found once"]
    n = lambda where: int((reps["where"] == where).sum())
    return {
        "session": r["session"],
        "curl sets found once": f"{(s['result'] == 'found once').sum()}/{len(s)}",
        "clean exact Enter -> found": f"{(clean['enter'] == clean['true_reps']).sum()} -> "
                                      f"{(clean['stream'] == clean['true_reps']).sum()} /{len(clean)}",
        "clean same count as Enter": f"{(clean['stream'] == clean['enter']).sum()}/{len(clean)}",
        "flagged exact Enter -> found": f"{(flagged['enter'] == flagged['true_reps']).sum()} -> "
                                        f"{(flagged['stream'] == flagged['true_reps']).sum()} /{len(flagged)}",
        "rest reps (per 10 min)": f"{n('rest')} ({10 * n('rest') / r['rest_min']:.1f}) in {r['rest_min']:.0f} min",
        "distractor reps Enter -> found": f"{r['enter_distractor_reps']} -> {n('distractor')}",
        "swing reps": n("swing"),
        "edge reps": n("edge"),
        "idle reps": f"{n('idle')} in {r['idle_min']:.0f} min" if r["idle_min"] else "-",
        "start / end off, median s": f"{found['start_off'].median():+.1f} / {found['end_off'].median():+.1f}"
        if len(found) else "-",
    }


def pooled_row(results: list[dict]) -> dict:
    """The stream-v1 pass criteria, pooled over sessions (clean curl sets only; PLAN.md 11-04)."""
    s = pd.concat([r["per_set"] for r in results])
    clean = s[s["group"] == "clean"]
    reps = pd.concat([r["reps"] for r in results])
    rest_min = sum(r["rest_min"] for r in results)
    idle_min = sum(r["idle_min"] for r in results)
    rest = int((reps["where"] == "rest").sum())
    return {
        "1. clean sets found once (>= 22/23)": f"{(clean['result'] == 'found once').sum()}/{len(clean)}",
        "2. clean same count as Enter (>= 21/23)": f"{(clean['stream'] == clean['enter']).sum()}/{len(clean)}",
        "3. rest reps per 10 min (<= 1)": f"{10 * rest / rest_min:.2f} ({rest} in {rest_min:.0f} min)",
        "idle reps (reported)": f"{int((reps['where'] == 'idle').sum())} in {idle_min:.0f} min" if idle_min else "-",
        "clean exact Enter -> found": f"{(clean['enter'] == clean['true_reps']).sum()} -> "
                                      f"{(clean['stream'] == clean['true_reps']).sum()} /{len(clean)}",
    }


def plot_boxes(r: dict, path, row_s=240.0):
    """The whole recording in rows of row_s seconds: angle, annotated sets shaded, boxes as bars,
    counted reps as ticks under them."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    host = r["df"]["host_t"].to_numpy()
    t0 = host[0]
    x = host - t0
    n_rows = int(np.ceil(x[-1] / row_s))
    fig, axes = plt.subplots(n_rows, 1, figsize=(16, 2.4 * n_rows), squeeze=False)
    shade = {"curl": ("#2a78d6", 0.12), "distractor": ("#6e7781", 0.18), "swing": ("#6e7781", 0.08),
             "discarded": ("#6e7781", 0.05)}
    ann = r["ann"].assign(kind=r["ann"].apply(set_kind, axis=1))
    rep_x = r["reps"]["t"].to_numpy() - t0
    for k, ax in enumerate(axes[:, 0]):
        lo, hi = k * row_s, (k + 1) * row_s
        m = (x >= lo) & (x <= hi)
        for a in ann.itertuples():
            s, e = a.start_host_t - t0, a.end_host_t - t0
            if e >= lo and s <= hi:
                color, alpha = shade[a.kind]
                ax.axvspan(s, e, color=color, alpha=alpha, lw=0)
                ax.text(max(s, lo), 186, f"{a.set_idx} {a.condition if a.kind == 'curl' else a.exercise}",
                        fontsize=7, color="#1f2328", va="bottom")
        for bx in r["boxes"]:
            s, e = bx.start - t0, bx.end - t0
            if e >= lo and s <= hi:
                ax.plot([s, e], [172, 172], color="#eb6834", lw=4, solid_capstyle="butt")
        mine = rep_x[(rep_x >= lo) & (rep_x <= hi)]
        ax.plot(mine, np.full(len(mine), 163), "|", color="#eb6834", ms=6)
        ax.plot(x[m], r["angle"][m], color="#2a78d6", lw=0.9)
        ax.axhline(LIFT_MIN_DEG, color="#6e7781", lw=0.6, ls=":")
        ax.set_xlim(lo, hi); ax.set_ylim(-5, 200); ax.set_yticks([0, 50, 100, 150])
        ax.tick_params(labelsize=7, colors="#6e7781")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0, 0].set_title(f"{r['session']}: forearm angle (blue); annotated sets (blue = curl, gray = distractor/swing); "
                         f"sets found without the Enter presses (orange bars) and the reps counted in them (ticks)",
                         loc="left", fontsize=9)
    axes[-1, 0].set_xlabel("s from recording start", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Find and count sets in whole recordings without the Enter presses.")
    parser.add_argument("sessions", nargs="+")
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--detector", choices=["v1", "v2"], default="v1", help="learned model inside the counters")
    args = parser.parse_args()
    span = {"v1": "dip", "v2": "lift"}[args.detector]
    usb_axis = json.loads(USB_AXIS_FILE.read_text())["axis"]
    rows, results = [], []
    for session in args.sessions:
        r = score(session, usb_axis, span)
        results.append(r)
        rows.append(summary_row(r))
        s, reps = r["per_set"], r["reps"]
        odd = s[(s["result"] != "found once") | (s["stream"] != s["enter"]) | (s["stream"] != s["true_reps"])]
        print(f"== {session}: {len(r['lifts'])} lifts, {len(r['boxes'])} sets found")
        if len(odd):
            print(odd[["set_idx", "condition", "group", "result", "true_reps", "enter", "stream", "edge_reps"]]
                  .to_string(index=False))
        stray = reps[~reps["where"].isin(["curl"])]
        if len(stray):
            print("reps counted outside curl sets: " + ", ".join(
                f"{w} {n}" for w, n in stray["where"].value_counts().items()))
        if args.plot:
            path = REPORTS_DIR / f"{session}_stream.png"
            plot_boxes(r, path)
            print(f"plot: {path.relative_to(REPORTS_DIR.parent)}")
    print()
    print(pd.DataFrame(rows).set_index("session").T.to_string())
    if len(results) > 1:
        print("\npooled:")
        for k, v in pooled_row(results).items():
            print(f"  {k:42s} {v}")


if __name__ == "__main__":
    main()
