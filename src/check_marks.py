"""Scripted marks per movement, lined up with the teacher's measured reps.

make_script wrote which movements should be half reps, drift or pauses (`marked`, by movement
number). This expands every recorded lift set into one row per movement with its scripted kind
(full, half_lift, half_lower, drift, pause). For curls it also attaches the teacher's measurement of the same movement from
labels.py, so the scripted errors can be checked against what the arm actually did, and the
partial/drift yes-no thresholds can be set from them. Each kind shows in its own teacher column:
half_lift in rom_deg, half_lower in end_gap_deg (it goes up fully but stops short of the bottom),
drift in drift_deg, pause in up_s.

Movement N lines up with teacher rep N only if both saw the same number of movements. When
they disagree (a half rep too small to detect, an extra twitch), the set's teacher columns stay
empty and the set is flagged: guessing which movement is missing would put labels on the wrong
one. A set whose typed rep count differs from the script is left out for the same reason.
The exhaustion set has no script to check, so it is left out too.

    python -m src.check_marks dev1
"""

import argparse
import sys

import numpy as np
import pandas as pd

from src.dataio import DATA_DIR, REST_KINDS, annotation_path, load_annotations, movement_count, parse_marked
from src.labels import LABELS_DIR
from src.make_script import LIFT_CONDITIONS, PROTOCOLS_DIR

CHECK_MARKS_DIR = DATA_DIR / "check_marks"
KINDS = ("full", "half_lift", "half_lower", "drift", "pause", *REST_KINDS)
SUMMARY_COLUMNS = ["rom_deg", "end_gap_deg", "drift_deg", "up_s"]  # one per kind
MOVE_COLUMNS = ["set_idx", "exercise", "condition", "movement", "kind"]
TEACHER_COLUMNS = ["rom_deg", "top_gap_deg", "start_gap_deg", "end_gap_deg", "drift_deg", "up_s", "down_s"]


def load_script(session: str) -> pd.DataFrame:
    """protocols/<session>.csv as strings, empty cells kept as "" (not NaN)."""
    return pd.read_csv(PROTOCOLS_DIR / f"{session}.csv", dtype=str, keep_default_na=False)


def expand_marks(script: pd.DataFrame, ann: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """One row per movement of every recorded scripted lift set, plus problems found."""
    if "marked" not in script.columns:
        raise ValueError("script has no `marked` column: it wasn't written by make_script")
    rows, problems = [], []
    for _, a in ann[ann["status"] == "ok"].iterrows():
        i = int(a["set_idx"])
        s = script.iloc[i - 1] if i <= len(script) else None
        # annotations store only set_idx, so check it really points at the same set
        if s is None or (s["exercise"], int(s["reps"])) != (a["exercise"], int(a["scripted_reps"])):
            raise ValueError(f"set {i} was recorded as {a['exercise']} x{a['scripted_reps']} but the "
                             "script row doesn't match: is this the script the session used?")
        if s["condition"] not in LIFT_CONDITIONS:
            continue
        if int(a["actual_reps"]) != int(s["reps"]):
            problems.append(f"set {i}: {int(a['actual_reps'])} full reps done, {s['reps']} scripted: "
                            "can't tell which movement changed, set left out")
            continue
        marks = parse_marked(s["marked"])
        for m in range(1, movement_count(int(s["reps"]), marks) + 1):
            rows.append({"set_idx": i, "exercise": s["exercise"], "condition": s["condition"],
                         "movement": m, "kind": marks.get(m, "full")})
    return pd.DataFrame(rows, columns=MOVE_COLUMNS), problems


def attach_teacher(moves: pd.DataFrame, labels: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Teacher columns onto curl movements by order within the set; a count mismatch leaves the set empty."""
    out = moves.copy()
    out[TEACHER_COLUMNS] = np.nan
    problems = []
    for set_idx, group in out[out["exercise"] == "curl"].groupby("set_idx"):
        reps = labels[labels["set_idx"] == set_idx].sort_values("rep")
        if len(reps) != len(group):
            problems.append(f"set {set_idx}: teacher found {len(reps)} movements, script has "
                            f"{len(group)}: teacher columns left empty")
            continue
        out.loc[group.sort_values("movement").index, TEACHER_COLUMNS] = reps[TEACHER_COLUMNS].to_numpy()
    return out, problems


def summarize(moves: pd.DataFrame) -> pd.DataFrame:
    """Per kind: how many movements, and the teacher's medians where measured."""
    cols = [c for c in SUMMARY_COLUMNS if c in moves]
    table = moves.groupby("kind").agg(n=("movement", "size"), **{c: (c, "median") for c in cols})
    return table.reindex([k for k in KINDS if k in table.index])


def main():
    parser = argparse.ArgumentParser(description="Scripted marks per movement, lined up with teacher reps.")
    parser.add_argument("session")
    args = parser.parse_args()

    script_path = PROTOCOLS_DIR / f"{args.session}.csv"
    if not script_path.exists():
        sys.exit(f"{script_path} not found: check_marks needs the make_script output for this session")
    if not annotation_path(args.session).exists():
        sys.exit(f"Session '{args.session}' hasn't been recorded yet: no annotations file")
    try:
        moves, problems = expand_marks(load_script(args.session), load_annotations(args.session))
    except ValueError as e:
        sys.exit(f"Checking marks failed for {args.session}: {e}")

    labels_path = LABELS_DIR / f"{args.session}.csv"
    if labels_path.exists():
        moves, teacher_problems = attach_teacher(moves, pd.read_csv(labels_path))
        problems += teacher_problems
    else:
        print(f"no teacher labels: run `python -m src.labels {args.session}` to add them")

    CHECK_MARKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CHECK_MARKS_DIR / f"{args.session}.csv"
    moves.insert(0, "session", args.session)
    moves.to_csv(out_path, index=False)
    print(summarize(moves).round(1).to_string())
    print(f"wrote {out_path.relative_to(DATA_DIR.parent)}")
    for p in problems:
        print(f"  problem: {p}")
    sys.exit(1 if problems else 0)


if __name__ == "__main__":
    main()
