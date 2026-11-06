"""Counter tests on a simulated wrist: the forearm hinges about a fixed axis from hanging (0 deg)
to the top of each rep, and the Nano is strapped on at a random angle. Planted: full reps, half
reps, a pause halfway up, and reps lowered only partway (the next lift starts from there)."""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.counter import count_set

FS = 50.0


def ease(a, b, seconds):
    n = int(seconds * FS)
    return a + (b - a) * (1 - np.cos(np.pi * np.arange(n) / n)) / 2


def elevation(reps):
    """Forearm angle over a set. Each rep is {"top": deg, "pause": bool, "bottom": deg (default 0: lowered all
    the way)}; the next rep lifts from that bottom. The set starts and ends hanging."""
    parts = [np.zeros(int(0.5 * FS))]
    at = 0.0
    for r in reps:
        if r.get("pause"):
            parts += [ease(at, r["top"] / 2, 0.5), np.full(int(2.0 * FS), r["top"] / 2), ease(r["top"] / 2, r["top"], 0.5)]
        else:
            parts.append(ease(at, r["top"], 1.0))
        at = r.get("bottom", 0.0)
        parts += [ease(r["top"], at, 1.0), np.full(int(0.2 * FS), at)]
    parts.append(np.zeros(int(0.5 * FS)))
    return np.concatenate(parts)


def wrist_set(reps, mount_seed=0, seed=0):
    rng = np.random.default_rng(seed)
    angle = elevation(reps)
    t = np.arange(len(angle)) / FS
    rot = Rotation.from_rotvec(np.outer(np.radians(angle), [0, 1.0, 0]))
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


FULL = {"top": 120}
HALF = {"top": 60}


def test_counts_full_reps_both_ways():
    c = count_set(wrist_set([FULL] * 8))
    assert (c["angle"].n, c["gyro"].n) == (8, 8)


def test_half_reps_are_left_out():
    reps = [FULL, FULL, HALF, FULL, FULL, HALF, FULL, FULL, FULL]
    c = count_set(wrist_set(reps))
    assert c["angle"].n == 7
    assert abs(c["gyro"].n - 7) <= 1


def test_a_pause_halfway_up_does_not_split_a_rep():
    reps = [FULL] * 3 + [{"top": 120, "pause": True}] + [FULL] * 4
    c = count_set(wrist_set(reps))
    assert (c["angle"].n, c["gyro"].n) == (8, 8)


def test_mounting_does_not_change_counts():
    reps = [FULL, FULL, HALF, FULL, FULL, FULL]
    counts = {(count_set(wrist_set(reps, mount_seed=s))["angle"].n) for s in range(5)}
    assert counts == {5}


def test_v3_drops_the_rep_that_never_came_back_down():
    # Lift 3 is lowered only to 60 deg and is the tallest top, so height (v2) keeps it and drops lift 4
    reps = [FULL, FULL, {"top": 130, "bottom": 60}, FULL, FULL, FULL]
    seg = wrist_set(reps)
    old, new = count_set(seg, "v2")["angle"], count_set(seg, "v3")["angle"]
    assert old.n == new.n == 5
    assert len(set(np.round(old.times, 1)) ^ set(np.round(new.times, 1))) == 2  # they keep different humps
    assert new.times[2] > old.times[2]                                            # v3 keeps the later one


def test_v3_a_wobble_on_the_way_down_is_not_the_next_lift():
    # Lift 2 is lowered to 60 deg, rises 18 deg, then goes all the way down: one full rep, not a half-lower
    reps = [FULL, {"top": 120, "bottom": 60}, {"top": 78}, FULL, FULL]
    assert count_set(wrist_set(reps), "v3")["angle"].n == 4
