"""Learned rep detector tests on a simulated wrist (same physics as test_counter): full reps,
half reps, a pause, and "wiggles" (small, jittery, irregular arm movement like walking)."""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

import src.detector as detector
from src.counter import lift_span
from src.detector import (FEATURES, MIN_REPS_GAP_S, combined_count, count_with, hump_features, label_humps,
                          min_reps, train)
from test_counter import FS, FULL, HALF, elevation


def sensors(angle_deg, mount_seed=0, seed=0):
    """Nano readings for a forearm swinging about y by angle_deg, strapped on at a random angle."""
    rng = np.random.default_rng(seed)
    t = np.arange(len(angle_deg)) / FS
    rot = Rotation.from_rotvec(np.outer(np.radians(angle_deg), [0, 1.0, 0]))
    accel = rot.inv().apply([0, 0, 1.0])
    step = (rot[:-1].inv() * rot[1:]).as_rotvec() * FS
    gyro = np.degrees(np.vstack([step, step[-1]]))
    mount = Rotation.random(random_state=mount_seed).as_matrix()
    accel, gyro = accel @ mount, gyro @ mount
    df = pd.DataFrame({"t": t})
    for i, c in enumerate("xyz"):
        df[f"a{c}"] = accel[:, i] + rng.normal(0, 0.01, len(t))
        df[f"g{c}"] = gyro[:, i] + rng.normal(0, 1.0, len(t))
    return df


def wiggles(seed, seconds=20.0):
    """Walking-like arm movement: small humps at irregular gaps, with fast shake on top."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * FS)) / FS
    angle = np.zeros_like(t)
    c = 1.0
    while c < seconds - 1:
        width, height = rng.uniform(0.4, 0.9), rng.uniform(22, 40)
        m = np.abs(t - c) < width / 2
        angle[m] += height * np.cos(np.pi * (t[m] - c) / width) ** 2
        c += width + rng.uniform(0.2, 1.2)
    return angle + 4 * np.sin(2 * np.pi * 6 * t) + rng.normal(0, 1.5, len(t))


def mixed(seed):
    """A curl set: 6 full reps and 2 half reps in random positions, plus its true labels."""
    rng = np.random.default_rng(seed)
    reps = [FULL] * 6 + [HALF] * 2
    order = rng.permutation(len(reps))
    reps = [reps[i] for i in order]
    return reps, np.array([1 if r is FULL else 0 for r in reps])


def training_humps():
    rows = []
    for k in range(4):
        reps, labels = mixed(k)
        f = hump_features(sensors(elevation(reps), mount_seed=k, seed=k))
        rows.append(f.assign(label=labels, session=f"train{k}", set_idx=1))
        w = hump_features(sensors(wiggles(k), mount_seed=10 + k, seed=k))
        rows.append(w.assign(label=0, session=f"train{k}", set_idx=2))
    return pd.concat(rows, ignore_index=True)


def test_labels_follow_the_script():
    # 6 full reps plus a half lift (lift 2) and a half lower (lift 5): 8 lifts in all.
    assert label_humps(8, "partial", "6", "2:half_lift;5:half_lower").tolist() == [1, 0, 1, 1, 0, 1, 1, 1]
    assert label_humps(8, "drift", "8", "3:drift;4:drift;5:drift").tolist() == [1] * 8  # drift reps still count
    assert label_humps(7, "partial", "6", "2:half_lift;5:half_lower") is None          # a hump missing: skip
    assert label_humps(4, "distractor", "0", "").tolist() == [0, 0, 0, 0]
    assert label_humps(9, "exhaustion", "0", "") is None


def test_features_do_not_depend_on_mounting():
    angle = elevation([FULL, HALF, FULL, FULL])
    a = hump_features(sensors(angle, mount_seed=1))
    b = hump_features(sensors(angle, mount_seed=2))
    cols = ["height_deg", "top_deg", "start_deg", "end_deg", "twist_deg", "one_axis_share", "rise_s"]
    assert np.abs(a[cols].to_numpy() - b[cols].to_numpy()).max() < 3


def test_features_tell_full_half_and_wiggle_apart():
    reps = [FULL, HALF, FULL, HALF, FULL]
    curls = hump_features(sensors(elevation(reps)))
    walk = hump_features(sensors(wiggles(0)))
    full, half = curls.iloc[[0, 2, 4]], curls.iloc[[1, 3]]
    assert full["top_deg"].min() > half["top_deg"].max() > walk["top_deg"].median()
    assert walk["jitter_dps"].median() > 2 * curls["jitter_dps"].median()
    assert walk["rhythm_off"].median() > curls["rhythm_off"].median()


def test_detector_counts_full_reps_and_ignores_wiggles_on_unseen_sets():
    model = train(training_humps())
    unseen = []
    for k in range(3):  # new seeds and mountings the model never saw
        reps, labels = mixed(100 + k)
        unseen.append(hump_features(sensors(elevation(reps), mount_seed=50 + k, seed=100 + k))
                      .assign(session="test", set_idx=k, full=int(labels.sum())))
        unseen.append(hump_features(sensors(wiggles(100 + k), mount_seed=60 + k, seed=100 + k))
                      .assign(session="test", set_idx=10 + k, full=0))
    pause = [FULL] * 3 + [{"top": 120, "pause": True}] + [FULL] * 3
    unseen.append(hump_features(sensors(elevation(pause), mount_seed=70)).assign(session="test", set_idx=20, full=7))
    humps = pd.concat(unseen, ignore_index=True)
    counted = count_with(model, humps)
    truth = humps.groupby(["session", "set_idx"])["full"].first()
    assert (counted == truth).all(), pd.DataFrame({"counted": counted, "truth": truth})


def test_held_out_session_never_trains_its_own_model(monkeypatch):
    seen = []
    real_train = detector.train

    def spy(humps):
        seen.append(set(humps["session"]))
        return real_train(humps)

    monkeypatch.setattr(detector, "train", spy)
    humps = training_humps()
    sets = humps.groupby(["session", "set_idx"]).size().reset_index()[["session", "set_idx"]]
    sets["condition"], sets["true_reps"], sets["rule_count"] = "normal", 0, 0
    sets["rule_times"] = [np.array([])] * len(sets)
    detector.leave_one_session_out(humps, sets)
    held_out = list(sets["session"].unique())
    assert len(seen) == len(held_out)
    for session, trained_on in zip(held_out, seen):
        assert session not in trained_on


def test_a_set_that_starts_bent_gives_the_same_humps():
    # Reaching for the laptop with the curling hand: hold 50 deg, drop to hanging, settle, then reps.
    reps = [FULL, HALF, FULL, FULL]
    reach = np.r_[np.full(int(0.5 * FS), 50.0), 50 * (1 + np.cos(np.pi * np.arange(int(0.5 * FS)) / int(0.5 * FS))) / 2,
                  np.zeros(int(0.3 * FS))]
    clean = hump_features(sensors(elevation(reps), mount_seed=3))
    bent = hump_features(sensors(np.r_[reach, elevation(reps)], mount_seed=3))
    assert len(bent) == len(clean) == 4
    cols = ["height_deg", "top_deg", "start_deg", "end_deg"]
    assert np.abs(bent[cols].to_numpy() - clean[cols].to_numpy()).max() < 4


def test_combined_keeps_rule_humps_that_look_like_curls_and_vetoes_the_rest():
    tops = np.array([1.05, 3.0, 5.1])   # learned humps
    p = np.array([0.9, 0.01, 0.3])      # 0.01: not a curl at all; 0.3: unusual, but a curl
    assert combined_count([1.0, 3.0, 5.0], tops, p) == 2


def test_combined_drops_rule_humps_with_no_learned_hump_nearby():
    assert combined_count([1.0, 9.0], np.array([1.0]), np.array([0.9])) == 1
    assert combined_count([1.0], np.array([]), np.array([])) == 0


def test_min_reps_drops_a_rep_with_no_other_rep_near_it():
    # a set of 4, then a stray bend 20 s after its last rep
    assert min_reps([2.0, 4.5, 7.0, 9.5, 29.5]).tolist() == [2.0, 4.5, 7.0, 9.5]
    assert min_reps([5.0]).tolist() == [] and min_reps([]).tolist() == []


def test_min_reps_keeps_reps_exactly_the_gap_apart_and_drops_just_beyond():
    assert min_reps([0.0, MIN_REPS_GAP_S]).tolist() == [0.0, MIN_REPS_GAP_S]
    assert min_reps([0.0, MIN_REPS_GAP_S + 0.1]).tolist() == []


def test_min_reps_keeps_a_slow_set_whole():
    slow = [0.0, 9.7, 19.4, 29.1]  # the longest lift-to-lift time seen in a real set
    assert min_reps(slow).tolist() == slow


def test_lift_span_starts_where_the_lift_leaves_its_dip_not_at_the_dip():
    angle = np.r_[np.zeros(100), np.linspace(0, 100, 51)[1:], np.linspace(100, 0, 51)[1:], np.zeros(100)]
    assert lift_span(angle, 40, 149, 260) == (104, 194)  # within 10 deg of the dips: 5 samples into the rise, 45 into the fall


def test_a_rest_before_a_rep_no_longer_changes_its_measured_lift():
    """Spliced sets (PLAN.md 11-04): v1 measured a rep from the lowest point since the last hump, so a
    10 s rest before it read as a 10 s rise and the veto removed a real rep. v2 measures the lift."""
    def rep_after(rest_s, version):
        angle = np.r_[elevation([FULL, FULL]), np.zeros(int(rest_s * FS)), elevation([FULL, FULL])]
        return hump_features(sensors(angle, mount_seed=5, seed=1), version=version).iloc[2]
    v1_short, v1_long = rep_after(0.2, "v1"), rep_after(10.0, "v1")
    v2_short, v2_long = rep_after(0.2, "v2"), rep_after(10.0, "v2")
    assert v1_long["rise_s"] - v1_short["rise_s"] > 5          # the flaw, as frozen in detector-v1
    for f, tol in [("rise_s", 0.1), ("fall_s", 0.1), ("twist_deg", 3), ("jitter_dps", 1), ("one_axis_share", 0.02)]:
        assert abs(v2_long[f] - v2_short[f]) < tol, f
    assert v2_long["start_deg"] == pytest.approx(v1_long["start_deg"])  # dips themselves are unchanged


def test_v3_rhythm_only_counts_a_hump_that_comes_too_soon():
    """v2 still vetoed the reps either side of a rest: their gap was 2-4x the usual. In v3 a rest is no
    evidence against a rep; a twitch right after a rep still is."""
    rest = np.r_[elevation([FULL] * 3), np.zeros(int(10 * FS)), elevation([FULL] * 3)]
    v2 = hump_features(sensors(rest, mount_seed=6, seed=2), version="v2")
    v3 = hump_features(sensors(rest, mount_seed=6, seed=2), version="v3")
    assert v2["rhythm_off"].iloc[2] > 1 and abs(v3["rhythm_off"].iloc[2]) < 1e-9 and abs(v3["rhythm_off"].iloc[3]) < 1e-9
    bump = 40 * (1 - np.cos(2 * np.pi * np.arange(int(0.6 * FS)) / int(0.6 * FS))) / 2  # a quick 40 deg twitch
    twitch = np.r_[elevation([FULL] * 3)[:-int(0.7 * FS)], bump, elevation([FULL] * 2)]  # ~1.3 s after a rep
    t3 = hump_features(sensors(twitch, mount_seed=6, seed=2), version="v3")
    assert len(t3) == 6 and t3["rhythm_off"].iloc[3] > 0.2 and t3["rhythm_off"].iloc[3] == t3["rhythm_off"].max()


def test_v4_has_no_rhythm_and_v5_only_looks_back():
    bump = 40 * (1 - np.cos(2 * np.pi * np.arange(int(0.6 * FS)) / int(0.6 * FS))) / 2
    twitch = np.r_[elevation([FULL] * 3)[:-int(0.7 * FS)], bump, elevation([FULL] * 2)]
    raw = sensors(twitch, mount_seed=6, seed=2)
    assert (hump_features(raw, version="v4")["rhythm_off"] == 0).all()
    v5 = hump_features(raw, version="v5")["rhythm_off"]
    assert v5.iloc[3] > 0.3 and v5.iloc[:2].eq(0).all()  # the twitch came too soon; the first two have no past
    # looking back only: what comes after a hump never changes its v5 rhythm
    cut = hump_features(sensors(twitch[: int(9.0 * FS)], mount_seed=6, seed=2), version="v5")["rhythm_off"]
    assert np.allclose(cut.iloc[:4], v5.iloc[:4])
