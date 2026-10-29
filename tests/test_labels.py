"""Label-maker tests on a simulated arm with known answers.

Shoulder fixed; the upper arm swings in the shoulder plane (about y); the elbow hinges about an
axis that can point out of the shoulder plane (people curl in their own plane); the forearm can
roll about its own axis (palm turning) and lean sideways. From chosen angles we compute what
each IMU reads (gravity plus the arm's own acceleration, and rotation rate), then mount the
sensors: the phone at a random angle, the Nano with its USB axis along the forearm toward the
elbow, tipped and spun by known amounts.
Planted by default: 8 curls, rep 6 only half way up, 25 deg of upper-arm drift on reps 3-5.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.transform import Rotation

from src.labels import label_session, track_down
from src.sync import UPPER_ARM

FS, T0, G = 50.0, 1.8e9, 9.81
UPPER_ARM_M, FOREARM_M, PHONE_M = 0.30, 0.28, 0.15
REP_S, TOP_DEG, HALF_DEG = 2.5, 130.0, 65.0
HALF_REP, DRIFT_REPS, LEAN_REP = 6, (3, 4, 5), 7
GAP_S = 2.0  # pause after rep 4 that a sideways swing can use
USB_AXIS = np.array([1.0, 0.0, 0.0])  # the board axis the USB port points along (in this simulation)
WINDOWS = {"sync1": (1.5, 6.5), "curl": (8.5, 31.5), "sync2": (34.5, 39.5)}


def about(axis, degrees):
    return Rotation.from_rotvec(np.outer(np.radians(degrees), axis))


def bump(t, start, length, peak):
    """0 -> peak -> 0 over [start, start + length), smooth at both ends."""
    out = np.zeros_like(t)
    m = (t >= start) & (t < start + length)
    out[m] = peak * (1 - np.cos(2 * np.pi * (t[m] - start) / length)) / 2
    return out


def nano_mount(tip_deg, spin_deg):
    """Sensor -> forearm frame. USB along the forearm toward the elbow (+z), tipped, then spun about it."""
    usb_to_forearm = Rotation.from_rotvec([0, -np.pi / 2, 0])  # board x -> forearm +z
    return (Rotation.from_rotvec([0, 0, np.radians(spin_deg)])
            * Rotation.from_rotvec([np.radians(tip_deg), 0, 0]) * usb_to_forearm).as_matrix()


def timeline(drift_deg, bottom_deg, twist_deg, lean_deg, swing_deg):
    """Angles (deg) over 47 s. Every set starts and ends with the arm hanging."""
    t = np.arange(0, 47, 1 / FS)
    upper, elbow, twist, side = (np.zeros_like(t) for _ in range(4))
    for start in (2.0, 35.0):  # straight-arm pendulum swings, 5 whole swings so they end hanging
        m = (t >= start) & (t < start + 4)
        upper[m] = 40 * (1 - np.cos(2 * np.pi * 1.25 * (t[m] - start))) / 2
    half = REP_S / 2
    for k in range(8):
        t0 = 9 + k * REP_S + (GAP_S if k >= 4 else 0)
        lo_start = 0.0 if k == 0 else bottom_deg
        lo_end = 0.0 if k == 7 else bottom_deg
        top = HALF_DEG if k + 1 == HALF_REP else TOP_DEG
        up, down = (t >= t0) & (t < t0 + half), (t >= t0 + half) & (t < t0 + REP_S)
        elbow[up] = lo_start + (top - lo_start) * (1 - np.cos(np.pi * (t[up] - t0) / half)) / 2
        elbow[down] = top + (lo_end - top) * (1 - np.cos(np.pi * (t[down] - t0 - half) / half)) / 2
        twist += bump(t, t0, REP_S, twist_deg)          # palm turns up on the way up, back on the way down
        if k + 1 in DRIFT_REPS:
            upper += bump(t, t0, REP_S, drift_deg)
        if k + 1 == LEAN_REP:
            side += bump(t, t0, REP_S, lean_deg)        # this rep leans out sideways as it curls
    gap = (t >= 9 + 4 * REP_S) & (t < 9 + 4 * REP_S + GAP_S)
    elbow[gap] = bottom_deg                             # the elbow holds its bottom during the pause
    side += bump(t, 9 + 4 * REP_S, GAP_S, swing_deg)    # pure sideways swing between reps 4 and 5
    return t, upper, elbow, twist, side


def readings(rot, position):
    """Body-frame accelerometer (g, gravity included) and gyro (deg/s) for one segment."""
    accel_world = np.gradient(np.gradient(position, 1 / FS, axis=0), 1 / FS, axis=0) / G + [0, 0, 1]
    step = (rot[:-1].inv() * rot[1:]).as_rotvec() * FS  # body-frame rotation rate between samples
    gyro = np.degrees(np.vstack([step, step[-1]]))
    return rot.inv().apply(accel_world), gyro


def make_session(curl_plane_deg=0.0, drift_deg=25.0, bottom_deg=0.0, twist_deg=0.0, lean_deg=0.0,
                 swing_deg=0.0, tip_deg=0.0, spin_deg=30.0, tip_after_curls=None, seed=0):
    rng = np.random.default_rng(seed)
    t, upper_deg, elbow_deg, twist, side = timeline(drift_deg, bottom_deg, twist_deg, lean_deg, swing_deg)
    y, z = np.array([0.0, 1.0, 0.0]), np.array([0.0, 0.0, 1.0])
    hinge = np.array([np.sin(np.radians(curl_plane_deg)), np.cos(np.radians(curl_plane_deg)), 0.0])
    forward = np.cross(hinge, z)
    upper_rot = about(y, upper_deg)
    fore_rot = upper_rot * about(forward, side) * about(hinge, elbow_deg) * about(z, twist)
    elbow_pos = upper_rot.apply([0, 0, -UPPER_ARM_M])
    f_acc, f_gyr = readings(fore_rot, elbow_pos + fore_rot.apply([0, 0, -FOREARM_M]))
    u_acc, u_gyr = readings(upper_rot, upper_rot.apply([0, 0, -PHONE_M]))

    mount = np.repeat(nano_mount(tip_deg, spin_deg)[None], len(t), axis=0)
    if tip_after_curls is not None:  # the wrist strap gets re-adjusted after the curls
        mount[t > 33] = nano_mount(tip_after_curls, spin_deg)
    w_acc = np.einsum("nij,ni->nj", mount, f_acc)   # sensor reading = mount^T @ body vector
    w_gyr = np.einsum("nij,ni->nj", mount, f_gyr) + [1.1, -0.4, 0.3]  # gyro bias like the Nano's
    phone = Rotation.random(random_state=seed + 100).as_matrix()
    u_acc, u_gyr = u_acc @ phone, u_gyr @ phone

    df = pd.DataFrame({"t": t, "host_t": T0 + t})
    for i, c in enumerate("xyz"):
        df[f"a{c}"] = w_acc[:, i] + rng.normal(0, 0.005, len(t))
        df[f"g{c}"] = w_gyr[:, i] + rng.normal(0, 0.5, len(t))
        df[f"{UPPER_ARM}a{c}"] = u_acc[:, i] + rng.normal(0, 0.005, len(t))
        df[f"{UPPER_ARM}g{c}"] = u_gyr[:, i] + rng.normal(0, 0.1, len(t))
    sets = [("sync", 0, "gesture", WINDOWS["sync1"]), ("curl", 8, "normal", WINDOWS["curl"]),
            ("sync", 0, "gesture", WINDOWS["sync2"])]
    ann = pd.DataFrame([{"set_idx": i + 1, "exercise": ex, "condition": cond, "status": "ok",
                         "actual_reps": reps, "start_host_t": T0 + a, "end_host_t": T0 + b}
                        for i, (ex, reps, cond, (a, b)) in enumerate(sets)])
    return df, ann


def label(**kwargs):
    return label_session(*make_session(**kwargs), usb_axis=USB_AXIS)


def test_track_down_follows_a_turning_sensor():
    t = np.arange(0, 2, 1 / FS)
    rot = about([1.0, 0, 0], 90 * t)                      # turning 90 deg/s about its own x
    true_down = rot.inv().apply([0, 0, -1.0])
    found = track_down(t, np.tile([90.0, 0, 0], (len(t), 1)), -true_down)
    assert np.degrees(np.arccos(np.clip((found * true_down).sum(axis=1), -1, 1))).max() < 1


def test_recovers_reps_range_and_drift():
    reps, info = label()
    assert info.problems == []
    assert len(reps) == 8
    assert np.abs(reps.loc[reps["rep"] != HALF_REP, "rom_deg"] - TOP_DEG).max() < 8
    assert reps.loc[reps["rep"] == HALF_REP, "rom_deg"].item() == pytest.approx(HALF_DEG, abs=8)
    drifted = reps["rep"].isin(DRIFT_REPS)
    assert np.abs(reps.loc[drifted, "drift_deg"] - 25).max() < 5
    assert reps.loc[~drifted, "drift_deg"].max() < 5
    assert reps["twist_deg"].max() < 7 and reps["wobble_deg"].max() < 6  # gyro bias adds ~3 deg per rep


def test_curls_outside_the_shoulder_plane_measure_the_same():
    reps, info = label(curl_plane_deg=50, drift_deg=0)
    assert info.problems == []
    assert len(reps) == 8
    assert np.abs(reps.loc[reps["rep"] != HALF_REP, "rom_deg"] - TOP_DEG).max() < 8


def test_palm_twist_is_measured_and_does_not_change_range():
    reps, info = label(twist_deg=50, drift_deg=0)
    assert info.problems == []
    assert len(reps) == 8
    assert np.abs(reps.loc[reps["rep"] != HALF_REP, "rom_deg"] - TOP_DEG).max() < 8
    assert np.abs(reps["twist_deg"] - 50).max() < 8


def test_a_leaning_rep_counts_and_stands_out():
    # Once the forearm points forward, leaning it sideways IS rolling it about its own axis, so
    # a lean shows up mostly as twist, plus some wobble and plane tilt. It must stand out on all three.
    reps, _ = label(lean_deg=30, drift_deg=0)
    assert len(reps) == 8
    leaned, rest = reps[reps["rep"] == LEAN_REP].iloc[0], reps[reps["rep"] != LEAN_REP]
    assert leaned["twist_deg"] > 15 > rest["twist_deg"].max() + 5
    assert leaned["wobble_deg"] > 4 > rest["wobble_deg"].max()
    assert leaned["plane_tilt_deg"] > 3 * rest["plane_tilt_deg"].max()


def test_a_pure_sideways_swing_is_not_counted():
    reps, info = label(swing_deg=35, drift_deg=0)
    assert len(reps) == 8
    assert any("not counted" in p for p in info.problems)


def test_never_fully_lowered_shows_in_bottom_bend_not_in_range():
    reps, _ = label(bottom_deg=40, drift_deg=0)
    shifted = reps[(reps["rep"] > 1) & ~reps["rep"].isin([HALF_REP, HALF_REP + 1])]
    assert np.abs(shifted["rom_deg"] - (TOP_DEG - 40)).max() < 8   # range is still top minus bottom
    assert np.abs(shifted["bottom_bend_deg"] - 40).max() < 6       # but the bottom is 40 deg from hanging
    assert abs(reps.loc[reps["rep"] == 1, "bottom_bend_deg"].item()) < 6  # the set starts hanging


def test_hang_offset_is_measured_and_mounting_does_not_move_the_labels():
    a, info_a = label(tip_deg=12, spin_deg=30)
    b, info_b = label(tip_deg=12, spin_deg=200)
    c, _ = label(tip_deg=0)
    for info in (info_a, info_b):
        assert info.hang_offset_deg == pytest.approx([12, 12], abs=2)
    cols = ["rom_deg", "drift_deg", "bottom_bend_deg"]
    assert np.abs(a[cols].to_numpy() - b[cols].to_numpy()).max() < 2   # spinning the band: no change
    assert np.abs(a[cols].to_numpy() - c[cols].to_numpy()).max() < 2   # tipping it: no change for the teacher


def test_strap_moved_between_gestures_is_flagged():
    _, info = label(tip_after_curls=25)
    assert info.slip_deg["wrist"] == pytest.approx(25, abs=3)
    assert any("a strap moved" in p for p in info.problems)
