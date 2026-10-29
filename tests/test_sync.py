"""Sync tests: plant a known offset, drift or fault in synthetic data and check it is found.

The synthetic session is 120 s on the Mac clock: a sync gesture at 5 s and 115 s
(irregularly spaced swing bursts, so the shape is distinctive) and repetitive "reps"
in between. Both sensors see the same rotation speed, as on a straight arm.
"""

import numpy as np
import pandas as pd
import pytest

from src.sync import MIN_STRENGTH, UPPER_ARM, align, board_to_unix, lowpass

T0 = 1.8e9                      # Mac Unix time at the start of the session
NANO_HZ, PHONE_HZ = 50.0, 99.63
LATENCY_S = 0.003               # USB delay before host_t is stamped
OFFSET_S, DRIFT = 0.137, 150e-6  # planted: phone clock behind by 137 ms, losing 150 ppm
GESTURES = (5.0, 115.0)


def true_speed(t, gestures=GESTURES):
    """Rotation speed (deg/s) at Mac time t seconds into the session."""
    speed = 2.0 + 100 * np.abs(np.sin(2 * np.pi * 0.6 * t)) * ((t > 20) & (t < 100))
    for center in gestures:
        for k, dt in enumerate([-0.9, -0.5, 0.0, 0.35, 0.8]):
            speed = speed + (250 + 30 * k) * np.exp(-((t - center - dt) / 0.07) ** 2)
    return speed


def make_nano(rng):
    t = np.arange(0, 120, 1 / NANO_HZ)
    df = pd.DataFrame({
        "t": t,
        "host_t": T0 + t + LATENCY_S + rng.uniform(0, 0.004, len(t)),
        "ax": 0.0, "ay": -1.0, "az": 0.0,
        "gx": true_speed(t), "gy": 0.0, "gz": 0.0,
    })
    return df


def make_phone(speed_fn=true_speed, mount_angle=0.7):
    """Phone frames on the phone's own clock, mounted at an angle so its axes differ from the Nano's."""
    mac_t = np.arange(0, 120, 1 / PHONE_HZ)
    unix_t = T0 + mac_t - (OFFSET_S + DRIFT * mac_t)
    speed = speed_fn(mac_t)
    gyro = pd.DataFrame({"unix_t": unix_t, "gx": speed * np.cos(mount_angle),
                         "gy": speed * np.sin(mount_angle), "gz": 0.0})
    accel = pd.DataFrame({"unix_t": unix_t, "ax": 0.0, "ay": 0.0, "az": 1.0})
    return accel, gyro


def make_ann(gestures=GESTURES):
    return pd.DataFrame([
        {"set_idx": i + 1, "exercise": "sync", "status": "ok",
         "start_host_t": T0 + c - 1.2, "end_host_t": T0 + c + 1.0}
        for i, c in enumerate(gestures)
    ])


def expected_offset(mac_t):
    """What align() should find: phone -> Nano shift, including the mean USB delay."""
    return LATENCY_S + 0.002 + OFFSET_S + DRIFT * mac_t


def test_board_to_unix_averages_out_usb_jitter():
    nano = make_nano(np.random.default_rng(0))
    fitted = board_to_unix(nano)
    truth = T0 + nano["t"] + LATENCY_S + 0.002  # mean of the 0-4 ms jitter
    assert np.abs(fitted - truth).max() < 0.001
    assert np.abs(nano["host_t"] - truth).max() > 0.0015  # raw host_t really was jittery


def test_lowpass_removes_fast_noise_without_delay():
    t = np.arange(0, 10, 0.01)
    slow = np.sin(2 * np.pi * 1.0 * t)
    filtered = lowpass(t, slow + 0.5 * np.sin(2 * np.pi * 40 * t))
    middle = slice(100, -100)  # skip filter edge effects
    assert np.abs(filtered[middle] - slow[middle]).max() < 0.02  # 40 Hz gone, 1 Hz intact and not shifted


def test_recovers_planted_offset_and_drift():
    nano = make_nano(np.random.default_rng(1))
    accel, gyro = make_phone()
    aligned, report = align(nano, accel, gyro, make_ann())

    assert report.problems == []
    assert min(report.strengths) > 0.99
    for center, found in zip(GESTURES, report.offsets_s):
        assert found == pytest.approx(expected_offset(center), abs=0.003)
    assert report.drift_ppm == pytest.approx(DRIFT * 1e6, abs=30)

    # After alignment the upper-arm rotation speed matches the wrist's sample for sample.
    ua_speed = np.sqrt((aligned[[UPPER_ARM + c for c in ("gx", "gy", "gz")]] ** 2).sum(axis=1))
    reps = (nano["t"] > 25) & (nano["t"] < 95)
    assert np.corrcoef(ua_speed[reps], nano["gx"][reps])[0, 1] > 0.99


def test_weak_gesture_is_flagged():
    rng = np.random.default_rng(2)

    def first_gesture_is_noise(t):
        speed = true_speed(t, gestures=(GESTURES[1],))
        return speed + np.where(np.abs(t - GESTURES[0]) < 1.5, rng.uniform(0, 300, len(t)), 0)

    accel, gyro = make_phone(first_gesture_is_noise)
    _, report = align(make_nano(rng), accel, gyro, make_ann())
    assert report.strengths[0] < MIN_STRENGTH
    assert any("weak match" in p for p in report.problems)
    assert any("drift is not corrected" in p for p in report.problems)  # only one gesture left


def test_one_gesture_means_no_drift_correction():
    accel, gyro = make_phone()
    _, report = align(make_nano(np.random.default_rng(3)), accel, gyro, make_ann(gestures=GESTURES[:1]))
    assert report.drift_ppm == 0
    assert any("drift is not corrected" in p for p in report.problems)


def test_missing_gesture_rows_is_an_error():
    accel, gyro = make_phone()
    no_sync = make_ann().assign(exercise="curl")
    with pytest.raises(ValueError, match="no sync gesture"):
        align(make_nano(np.random.default_rng(4)), accel, gyro, no_sync)
