"""check_marks tests: a made-up recorded session whose teacher labels are known, so every join can be checked.

Each scripted kind gets a planted value in its own teacher column (half_lift: range 55 deg,
half_lower: stops 55 deg short of the bottom, drift: 20 deg of upper-arm drift, pause: 3.5 s up);
every other movement gets the full-rep value. Any off-by-one in the join shows up as wrong values.
"""

import numpy as np
import pandas as pd
import pytest

from src.check_marks import TEACHER_COLUMNS, attach_teacher, expand_marks, summarize
from src.dataio import movement_count, parse_marked
from src.make_script import DISTRACTORS, LIFT_CONDITIONS, make_script

FULL = {"rom_deg": 110.0, "end_gap_deg": 0.0, "drift_deg": 3.0, "up_s": 1.0}
SIGNATURE = {"half_lift": ("rom_deg", 55.0), "half_lower": ("end_gap_deg", 55.0),
             "drift": ("drift_deg", 20.0), "pause": ("up_s", 3.5)}


def recorded(seed=7):
    """(script, annotations, teacher labels) for a session done exactly as scripted."""
    rows = make_script(["curl", "raise"], seed)
    script = pd.DataFrame(rows).astype(str)
    ann, labels = [], []
    for i, r in enumerate(rows, 1):
        ann.append({"set_idx": i, "exercise": r["exercise"], "scripted_reps": r["reps"],
                    "actual_reps": r["reps"], "condition": r["condition"], "status": "ok"})
        if r["exercise"] != "curl":
            continue
        marks = parse_marked(r["marked"])
        n = movement_count(r["reps"], marks) if r["condition"] != "exhaustion" else 15
        for m in range(1, n + 1):
            values = {c: 1.0 for c in TEACHER_COLUMNS} | FULL
            if m in marks:
                col, value = SIGNATURE[marks[m]]
                values[col] = value
            labels.append({"set_idx": i, "rep": m, **values})
    # a discarded first attempt of set 2 sits in the file too, as the receiver writes it
    ann.insert(1, {**ann[1], "actual_reps": np.nan, "status": "discarded"})
    return script, pd.DataFrame(ann), pd.DataFrame(labels)


def set_of(script, condition, exercise="curl"):
    return int(script.index[(script["condition"] == condition) & (script["exercise"] == exercise)][0]) + 1


def test_every_movement_gets_its_scripted_kind():
    script, ann, _ = recorded()
    moves, problems = expand_marks(script, ann)
    assert problems == []
    lifts = script[script["condition"].isin(LIFT_CONDITIONS)]
    assert moves["set_idx"].nunique() == len(lifts)
    assert set(moves["condition"]) == set(LIFT_CONDITIONS)  # style sets included, exhaustion not
    for set_idx, group in moves.groupby("set_idx"):
        marks = parse_marked(script.iloc[set_idx - 1]["marked"])
        assert list(group["movement"]) == list(range(1, len(group) + 1))
        assert {m: k for m, k in zip(group["movement"], group["kind"]) if k != "full"} == marks
    assert not moves["exercise"].isin(["sync", *DISTRACTORS]).any()


def test_teacher_values_land_on_the_right_movement():
    script, ann, labels = recorded()
    moves, _ = expand_marks(script, ann)
    joined, problems = attach_teacher(moves, labels)
    assert problems == []
    curl = joined[joined["exercise"] == "curl"]
    for kind, (col, value) in SIGNATURE.items():
        assert (curl.loc[curl["kind"] == kind, col] == value).all()
        assert (curl.loc[curl["kind"] != kind, col] == FULL[col]).all()
    assert joined.loc[joined["exercise"] != "curl", TEACHER_COLUMNS].isna().all().all()
    table = summarize(joined)
    assert list(table.index) == ["full", "half_lift", "half_lower", "drift", "pause"]
    for kind, (col, value) in SIGNATURE.items():
        assert table.loc[kind, col] == value


def test_teacher_count_mismatch_empties_only_that_set():
    script, ann, labels = recorded()
    partial = set_of(script, "partial")
    missed = labels.index[(labels["set_idx"] == partial) & (labels["rom_deg"] == 55.0)][0]
    moves, _ = expand_marks(script, ann)
    joined, problems = attach_teacher(moves, labels.drop(missed))  # teacher missed one half rep
    assert len(problems) == 1 and f"set {partial}:" in problems[0]
    assert joined.loc[joined["set_idx"] == partial, TEACHER_COLUMNS].isna().all().all()
    others = joined[(joined["exercise"] == "curl") & (joined["set_idx"] != partial)]
    assert others[TEACHER_COLUMNS].notna().all().all()


def test_set_with_a_changed_rep_count_is_left_out():
    script, ann, _ = recorded()
    drift = set_of(script, "drift")
    row = (ann["set_idx"] == drift) & (ann["status"] == "ok")
    ann.loc[row, "actual_reps"] = ann.loc[row, "actual_reps"].astype(int) - 1
    moves, problems = expand_marks(script, ann)
    assert drift not in set(moves["set_idx"])
    assert len(problems) == 1 and f"set {drift}:" in problems[0]


@pytest.mark.parametrize("bad", ["4:half", "4:halff", "4", "x:half_lift", "4:half_lift;4:drift",
                                 "4:half_lift,5:half_lower"])
def test_malformed_marks_are_refused(bad):
    assert parse_marked("4:half_lift;7:pause;9:drift") == {4: "half_lift", 7: "pause", 9: "drift"}
    with pytest.raises(ValueError):
        parse_marked(bad)


def test_wrong_script_is_refused():
    script, _, _ = recorded(seed=7)
    _, other_ann, _ = recorded(seed=8)
    with pytest.raises(ValueError, match="script row doesn't match"):
        expand_marks(script, other_ann)
