import re

import numpy as np
import pytest

from src.dataio import HALF_KINDS, REST_KINDS, REST_SECONDS, movement_count, parse_marked
from src.make_script import (
    CURL_STYLES, DISTRACTORS, DRIFT_PER_SET, DRIFT_SETS, EXERCISES, LIFT_CONDITIONS, MAX_REPS,
    MIN_REPS, NORMAL_SETS, PARTIAL_SETS, REST_CHANCE, make_script, shares, write_script,
)
from src.record import load_script

SEEDS = range(300)
EXTRA_HALF_SETS = ("normal", *CURL_STYLES)


def body(rows):
    return rows[1:-2]  # between the opening sync and the exhaustion set


def lift_sets(seed, exercises=EXERCISES):
    return [r for r in body(make_script(exercises, seed)) if r["condition"] in LIFT_CONDITIONS]


def test_same_seed_same_script_different_seed_different():
    assert make_script(EXERCISES, 3) == make_script(EXERCISES, 3)
    assert make_script(EXERCISES, 3) != make_script(EXERCISES, 4)


def test_session_shape():
    rows = make_script(EXERCISES, 1)
    assert EXERCISES == ["curl"]
    assert [r["exercise"] for r in (rows[0], rows[-1])] == ["sync", "sync"]
    assert (rows[-2]["exercise"], rows[-2]["condition"]) == ("curl", "exhaustion")
    sets = body(rows)
    assert sorted(r["exercise"] for r in sets if r["condition"] == "distractor") == sorted(DISTRACTORS)
    assert "drink" not in DISTRACTORS and "adjust" in DISTRACTORS
    conditions = sorted(r["condition"] for r in sets if r["condition"] != "distractor")
    expected = ["normal"] * NORMAL_SETS + CURL_STYLES + ["partial"] * PARTIAL_SETS + ["drift"] * DRIFT_SETS
    assert conditions == sorted(expected)
    assert {r["exercise"] for r in sets if r["condition"] != "distractor"} == {"curl"}
    assert len(rows) == 15


def test_other_exercises_only_add_normal_sets():
    sets = lift_sets(1, ["curl", "raise", "press"])
    assert [r["condition"] for r in sets if r["exercise"] != "curl"] == ["normal"] * NORMAL_SETS * 2
    with pytest.raises(ValueError):
        make_script(["raise", "press"], 1)
    with pytest.raises(ValueError):
        make_script(["curl", "squat"], 1)


def test_marks_follow_the_spec_and_stay_off_the_first_and_last_movement():
    for seed in SEEDS:
        for r in lift_sets(seed, ["curl", "raise"]):
            assert MIN_REPS <= r["reps"] <= MAX_REPS
            marks = parse_marked(r["marked"])
            n = movement_count(r["reps"], marks)
            assert all(2 <= m <= n - 1 for m in marks)
            # a lift after a half_lower starts halfway up: it can't be "only halfway" or "hold halfway"
            assert not any(k == "half_lower" and marks.get(m + 1) in ("half_lift", "pause")
                           for m, k in marks.items())
            rests = [k for k in marks.values() if k in REST_KINDS]
            assert len(rests) <= 1 and (not rests or r["exercise"] == "curl")
            kinds = sorted(k for k in marks.values() if k not in REST_KINDS)
            extra = [k for k in kinds if k in HALF_KINDS]
            if r["condition"] == "partial":
                assert kinds == ["half_lift", "half_lift", "half_lower", "half_lower"]
            elif r["condition"] == "drift":
                assert kinds == ["drift"] * DRIFT_PER_SET
            elif r["condition"] == "pause":
                assert kinds.count("pause") == 1 and len(extra) <= 1 and len(kinds) == 1 + len(extra)
            else:
                assert len(extra) <= 1 and len(kinds) == len(extra)


def test_extra_half_rep_in_about_half_the_eligible_sets():
    eligible = [r for seed in SEEDS for r in lift_sets(seed) if r["condition"] in EXTRA_HALF_SETS]
    extras = [k for r in eligible for k in parse_marked(r["marked"]).values() if k in HALF_KINDS]
    rate = len(extras) / len(eligible)
    assert 0.45 < rate < 0.55  # 1500 sets: the 95% band around 0.5 is about ±0.026
    # lift or lower at random, except alternating sets, which only lift halfway
    mixed = [k for r in eligible if r["condition"] != "alternating"
             for k in parse_marked(r["marked"]).values() if k in HALF_KINDS]
    assert 0.4 < mixed.count("half_lift") / len(mixed) < 0.6


def test_notes_name_exactly_the_marked_lifts():
    # The person follows the notes; analysis follows `marked`. If they disagree, labels are wrong.
    for seed in range(100):
        for r in lift_sets(seed, ["curl", "raise"]):
            marks = parse_marked(r["marked"])
            named = re.findall(r"Lifts? ([\d, and]+):", r["notes"])
            numbers = [int(x) for group in named for x in re.findall(r"\d+", group)]
            assert sorted(numbers) == sorted(marks)
            assert re.search(r"(\d+) lifts\.", r["notes"]).group(1) == str(movement_count(r["reps"], marks))
            if "pause" in marks.values():
                assert "hold 2–3 s halfway up" in r["notes"]


def test_default_session_hits_the_edge_case_targets():
    stats = np.array([shares(make_script(EXERCISES, seed)) for seed in range(200)])
    assert 0.10 <= stats[:, 1].mean() <= 0.14  # half reps, target ~12%
    assert 0.05 <= stats[:, 2].mean() <= 0.09  # drift reps, target ~7%


def test_rep_counts_are_not_predictable():
    counts = {r["reps"] for seed in range(50) for r in lift_sets(seed)}
    assert counts == set(range(MIN_REPS, MAX_REPS + 1))


def test_receiver_reads_it_and_it_is_never_silently_overwritten(tmp_path):
    rows = make_script(EXERCISES, 1)
    path = write_script("p1_s1", rows, {"seed": 1}, out_dir=tmp_path)
    loaded = load_script(path)
    assert len(loaded) == len(rows)
    assert [r["marked"] for r in loaded] == [r["marked"] for r in rows]
    assert path.with_suffix(".json").exists()
    with pytest.raises(FileExistsError):
        write_script("p1_s1", rows, {"seed": 1}, out_dir=tmp_path)
    write_script("p1_s1", rows, {"seed": 1}, out_dir=tmp_path, force=True)


def test_alternating_sets_never_lower_halfway():
    # You can't lower one arm halfway, then start the next lift with the other hand.
    for seed in SEEDS:
        for r in lift_sets(seed):
            if r["condition"] == "alternating":
                assert "half_lower" not in parse_marked(r["marked"]).values()


def test_a_rest_in_about_half_the_curl_sets_lengths_mixed_and_never_changing_the_count():
    curl_sets = [r for seed in SEEDS for r in lift_sets(seed) if r["exercise"] == "curl"]
    rests = [k for r in curl_sets for k in parse_marked(r["marked"]).values() if k in REST_KINDS]
    assert abs(len(rests) / len(curl_sets) - REST_CHANCE) < 0.05  # a few drift sets have no room for one
    assert {int(k[4:]) for k in rests} == set(REST_SECONDS)
    for r in curl_sets:
        marks = parse_marked(r["marked"])
        for m, k in marks.items():
            if k in REST_KINDS:
                assert f"Lift {m}: after it, lower all the way and let the arm hang still for {k[4:]} s" in r["notes"]
                assert marks.get(m) == k  # the rested lift carries no other mark: it is a plain full rep
        halves = sum(k in HALF_KINDS for k in marks.values())
        assert movement_count(r["reps"], marks) == r["reps"] + halves  # rests add no lift
