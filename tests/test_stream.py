"""Set-finding tests on a simulated recording: rest, sets and stray movement in one stream, the
Nano strapped on at a random angle. Planted: a normal set, a slow set, a lone bend, walking, two
sets with a rest between, and a bend just before a set."""

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.stream import BOX_GAP_S, PAD_S, draw_boxes, find_lifts, flagged_sets, place_reps, stream_angle
from test_counter import FS, ease


def rest(seconds):
    return np.zeros(int(seconds * FS))


def reps(n, top=120.0, up_s=1.0, down_s=1.0, hold_s=0.2):
    parts = []
    for _ in range(n):
        parts += [ease(0, top, up_s), ease(top, 0, down_s), np.zeros(int(hold_s * FS))]
    return np.concatenate(parts)


def walking(seconds, swing=25.0):
    """The forearm swinging up to `swing` deg about once a second."""
    t = np.arange(int(seconds * FS)) / FS
    return swing * (1 - np.cos(2 * np.pi * t)) / 2


def wrist_stream(*segments, mount_seed=0, tilt_deg=0.0, seed=0):
    """(sensor samples, the USB direction as the board sees it). The USB direction is the forearm's
    up-while-hanging, turned by tilt_deg if the board sits crooked on the arm."""
    rng = np.random.default_rng(seed)
    angle = np.concatenate(segments)
    t = np.arange(len(angle)) / FS
    rot = Rotation.from_rotvec(np.outer(np.radians(angle), [0, 1.0, 0]))
    accel = rot.inv().apply([0, 0, 1.0])
    step = (rot[:-1].inv() * rot[1:]).as_rotvec() * FS
    gyro = np.degrees(np.vstack([step, step[-1]]))
    mount = Rotation.random(random_state=mount_seed).as_matrix()
    accel, gyro = accel @ mount, gyro @ mount
    usb = Rotation.from_rotvec(np.radians(tilt_deg) * mount[0]).apply(mount[2])
    df = pd.DataFrame({"t": t})
    for i, c in enumerate("xyz"):
        df[f"a{c}"] = accel[:, i] + rng.normal(0, 0.01, len(t))
        df[f"g{c}"] = gyro[:, i] + rng.normal(0, 1.0, len(t))
    return df, usb


def boxes_in(df, usb):
    lifts = find_lifts(df["t"].to_numpy(), stream_angle(df, usb))
    return lifts, draw_boxes(lifts)


def test_a_set_is_one_box_padded_into_the_rest_on_both_sides():
    lifts, boxes = boxes_in(*wrist_stream(rest(60), reps(8), rest(60)))
    assert len(boxes) == 1 and len(boxes[0].lifts) == 8
    set_end = 60 + 8 * 2.2
    # starts PAD_S before the first lift begins (at 60 s), not back where the long rest dipped lowest
    assert 60 - PAD_S - 0.5 <= boxes[0].start <= 60 - PAD_S + 0.5
    assert set_end + PAD_S - 0.5 <= boxes[0].end <= set_end + PAD_S + 0.5


def test_slow_reps_nine_seconds_apart_stay_one_set():
    lifts, boxes = boxes_in(*wrist_stream(rest(5), reps(6, up_s=4, down_s=4, hold_s=1), rest(20)))
    assert [len(b.lifts) for b in boxes] == [6]


def test_a_lone_bend_is_a_lift_but_makes_no_box():
    lifts, boxes = boxes_in(*wrist_stream(rest(10), reps(1, top=110), rest(30)))
    assert len(lifts) == 1 and boxes == []


def test_walking_makes_no_lifts():
    lifts, boxes = boxes_in(*wrist_stream(rest(3), walking(20), rest(3)))
    assert lifts == [] and boxes == []


def test_two_sets_with_a_rest_between_are_two_boxes_that_do_not_overlap():
    lifts, boxes = boxes_in(*wrist_stream(rest(5), reps(5), rest(20), reps(5), rest(5)))
    assert [len(b.lifts) for b in boxes] == [5, 5]
    assert boxes[0].end <= boxes[1].start


def test_a_bend_just_before_a_set_joins_it():
    """Known limit (min-reps-v1's open risk): a dumbbell pick-up within BOX_GAP_S of the first rep
    lands in the set's box. Whether it becomes a rep is up to the counters inside the box."""
    lifts, boxes = boxes_in(*wrist_stream(rest(5), reps(1, top=100), rest(3), reps(6), rest(5)))
    assert [len(b.lifts) for b in boxes] == [7]
    assert boxes[0].lifts[1].top - boxes[0].lifts[0].top <= BOX_GAP_S


def test_a_crooked_board_still_sees_every_lift():
    for mount_seed in range(3):
        lifts, boxes = boxes_in(*wrist_stream(rest(5), reps(8), rest(10), mount_seed=mount_seed, tilt_deg=30))
        assert [len(b.lifts) for b in boxes] == [8]


def test_flagged_sets_come_from_the_session_notes(tmp_path):
    path = tmp_path / "sessions.csv"
    pd.DataFrame({"session": ["a", "b", "c"],
                  "notes": ["Operator notes: ... Flagged sets 5, 11 and 14 scored separately (decided before evaluating).",
                            "Flagged sets 8 and 14 scored separately", "Consent given."]}).to_csv(path, index=False)
    assert flagged_sets("a", path) == {5, 11, 14}
    assert flagged_sets("b", path) == {8, 14}
    assert flagged_sets("c", path) == set() and flagged_sets("missing", path) == set()


def test_each_counted_rep_is_placed_by_where_it_fell():
    ann = pd.DataFrame({"set_idx": [1, 2, 3], "exercise": ["sync", "curl", "walk"],
                        "condition": ["gesture", "normal", "distractor"], "status": "ok",
                        "start_host_t": [0.0, 20.0, 60.0], "end_host_t": [5.0, 40.0, 75.0]})
    reps = pd.DataFrame({"t": [2.0, 25.0, 41.0, 50.0, 65.0, 200.0],
                         "box_on": ["swing", "curl", "curl", "nothing", "distractor", "nothing"]})
    assert place_reps(reps, ann, idle_after=100.0).tolist() == ["swing", "curl", "edge", "rest", "distractor", "idle"]
