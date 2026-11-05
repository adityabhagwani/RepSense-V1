"""Splicer tests: joins keep every sample at 50 Hz, a recipe's truth is its full pieces, and a
simulated set cut at its rep bottoms and put back together in shuffled order counts the same."""

import numpy as np
import pandas as pd
import pytest

from src.counter import count_set
from src.detector import hump_features
from src.labels import track_down
from src.splice import (FS, LEAD_S, Glue, Library, build_lever_set, cut_points, degrade, null_recipe, plan_levers,
                        relaxed_hang, splice, still_hang_runs, take_hang)
from test_counter import elevation
from test_detector import sensors


def test_splice_keeps_each_window_in_order_at_50_hz():
    raw = pd.DataFrame({"t": np.arange(500) / FS})
    for c in ["ax", "ay", "az", "gx", "gy", "gz"]:
        raw[c] = np.arange(500, dtype=float)
    out = splice(raw, [(5.0, 6.0), (1.0, 1.5)])
    assert out["ax"].tolist() == list(range(250, 300)) + list(range(50, 75))
    assert np.allclose(np.diff(out["t"]), 1 / FS) and (out["host_t"] == out["t"]).all()


def test_a_null_recipe_counts_its_full_pieces_between_two_hangs():
    pieces = pd.DataFrame({"full": [True] * 12 + [False] * 3, "kind": ["full"] * 12 + ["half_lift"] * 3,
                           "clean_ends": True, "condition": "normal", "set_idx": 1,
                           "t_start": np.arange(15) * 3.0, "t_end": np.arange(15) * 3.0 + 2.5})
    glue = Glue(pd.DataFrame({"t_start": [100.0], "t_end": [110.0], "ux": [0.0], "uy": [0.0], "uz": [1.0]}),
                np.array([0.0, 0.0, 1.0]))
    for seed in range(5):
        r = null_recipe(np.random.default_rng(seed), "s", pieces, glue)
        assert r.truth == len(r.parts) - 2 and 5 <= r.truth <= 12
        assert all(LEAD_S[0] <= p[1] - p[0] <= LEAD_S[1] for p in (r.parts[0], r.parts[-1]))
        assert all(a < 45 for a, _ in r.parts[1:-1])  # only full pieces, never the half reps after 36 s


def test_a_set_cut_at_its_rep_bottoms_and_shuffled_counts_the_same():
    tops = [{"top": x} for x in (120, 112, 128, 118, 124, 115, 122)]
    raw = sensors(elevation(tops), mount_seed=4)
    humps = hump_features(raw)
    assert len(humps) == len(tops)
    cuts = cut_points(humps, raw["t"].iloc[0])
    order = np.random.default_rng(0).permutation(len(cuts))
    parts = [(0.0, 0.5)] + [tuple(cuts.iloc[k]) for k in order] + [(raw["t"].iloc[-1] - 0.5, raw["t"].iloc[-1])]
    assert count_set(splice(raw, parts))["angle"].n == len(tops)


def tilted(deg):
    return np.array([0.0, np.sin(np.radians(deg)), np.cos(np.radians(deg))])


def test_still_stretches_are_split_by_posture_and_far_ones_dropped():
    """Between sets people hang ~20-35 deg off the rep-bottom posture; those stretches are kept (to be
    turned), each posture its own clip. One 60 deg off isn't still hanging at all."""
    t = np.arange(600) / FS
    accel = np.vstack([np.tile(tilted(0), (200, 1)), np.tile(tilted(35), (200, 1)), np.tile(tilted(60), (200, 1))])
    runs = still_hang_runs(t, accel, np.zeros((600, 3)), tilted(0), np.ones(600, bool))
    assert runs[["t_start", "t_end"]].values.round(2).tolist() == [[0.0, 4.0], [4.0, 8.0]]
    assert np.allclose(runs[["ux", "uy", "uz"]].values, [tilted(0), tilted(35)], atol=1e-6)


def test_glue_is_turned_onto_the_rep_bottoms_or_left_relaxed_by_the_dose():
    """Regression (dev2b null check): a lead-in hanging 35 deg off moved the learned model's zero. Glue
    now sits where the rep bottoms do; a relaxed start (off_deg) is a lever, along the clip's own lean."""
    raw = pd.DataFrame({"t": np.arange(500) / FS})
    raw[["ax", "ay", "az"]] = np.tile(tilted(35), (500, 1))
    raw[["gx", "gy", "gz"]] = 0.0
    glue = Glue(pd.DataFrame({"t_start": [0.0], "t_end": [10.0], "ux": [tilted(35)[0]], "uy": [tilted(35)[1]],
                              "uz": [tilted(35)[2]]}), tilted(0))
    for off, want in [(0.0, tilted(0)), (20.0, tilted(20)), (35.0, tilted(35))]:
        parts = take_hang(np.random.default_rng(0), glue, 3.0, off_deg=off)
        seg = splice(raw, parts)
        assert len(seg) == 150 and np.allclose(seg[["ax", "ay", "az"]].values, want, atol=1e-6)


def test_a_relaxed_start_eases_into_the_curl_posture_and_the_gyro_agrees():
    """The ease is a turn of the readings, so the gyro must carry the matching rotation: following the
    gyro alone from the relaxed lean has to land where the accelerometer says the arm ends up."""
    raw = pd.DataFrame({"t": np.arange(1000) / FS})
    raw[["ax", "ay", "az"]] = np.tile(tilted(30), (1000, 1))
    raw[["gx", "gy", "gz"]] = 0.0
    glue = Glue(pd.DataFrame({"t_start": [0.0], "t_end": [20.0], "ux": [tilted(30)[0]], "uy": [tilted(30)[1]],
                              "uz": [tilted(30)[2]]}), tilted(0))
    for off in (10.0, 30.0, 40.0):
        seg = splice(raw, relaxed_hang(np.random.default_rng(1), glue, 2.0, off))
        accel = seg[["ax", "ay", "az"]].to_numpy()
        assert np.degrees(np.arccos(accel[0] @ tilted(0))) == pytest.approx(off, abs=0.5)
        assert np.allclose(accel[-1], tilted(0), atol=0.02)
        down = track_down(seg["t"].to_numpy(), seg[["gx", "gy", "gz"]].to_numpy(), accel, tau_s=np.inf)
        assert np.degrees(np.arccos(np.clip(-down[-1] @ accel[-1], -1, 1))) < 1.5


def fake_library():
    rows = []
    for kind, cond, full, n, t0 in [("full", "normal", True, 10, 0), ("full", "fast", True, 3, 100),
                                    ("full", "slow", True, 3, 200), ("full", "alternating", True, 3, 300),
                                    ("pause", "pause", True, 3, 400), ("drift", "drift", True, 3, 500),
                                    ("half_lift", "partial", False, 3, 600)]:
        for k in range(n):
            rows.append({"kind": kind, "condition": cond, "full": full, "clean_ends": True, "set_idx": 1,
                         "t_start": t0 + 3.0 * k, "t_end": t0 + 3.0 * k + 2.5})
    glue = Glue(pd.DataFrame({"t_start": [900.0], "t_end": [920.0], "ux": [0.0], "uy": [0.0], "uz": [1.0]}),
                np.array([0.0, 0.0, 1.0]))
    chunks = {"distractor": [(950.0, 955.0, np.zeros(3))], "edge": [(970.0, 974.0, np.zeros(3))]}
    return Library("fake", None, pd.DataFrame(rows), glue, chunks, {k: ["walk"] * len(v) for k, v in chunks.items()})


def test_levers_never_change_the_truth_and_land_where_planned():
    lib = fake_library()
    for seed in range(20):
        rng = np.random.default_rng(seed)
        plan = plan_levers(rng)
        plan.update({k: d for k, d in [("half_lifts", 2), ("distractor_bit", 1), ("pause_reps", 1), ("drift_reps", 1)]})
        r = build_lever_set(rng, lib, plan)
        starts = [p[0] for p in r.parts]
        assert r.truth == plan["set_length"]
        assert sum(600 <= s < 700 for s in starts) == 2          # the two half lifts are in, but not counted
        assert sum(950 <= s < 960 for s in starts) == 1          # one distractor chunk
        reps = [s for s in starts if s < 600]
        assert len(reps) == r.truth                              # every counted part is a full rep
        assert starts.index(min(s for s in starts if 600 <= s < 700)) > starts.index(reps[0])  # inside the set


def test_degrade_loses_samples_in_bursts_and_saturates_the_gyro():
    seg = pd.DataFrame({"host_t": np.arange(1000) / FS, "t": np.arange(1000) / FS})
    seg[["ax", "ay", "az"]] = 0.0
    seg[["gx", "gy", "gz"]] = np.tile(np.linspace(-500, 500, 1000)[:, None], (1, 3))
    out = degrade(seg, 0.05, 200.0, seed=3)
    assert 0.05 <= 1 - len(out) / 1000 <= 0.07 and np.diff(out["t"]).max() > 1.5 / FS
    assert out[["gx", "gy", "gz"]].abs().max().max() == 200.0
