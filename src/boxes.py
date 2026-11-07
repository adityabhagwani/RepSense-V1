"""Boxes: where the lifts say a set is. Moved out of stream.py unchanged (stream-v1, frozen 11-04) so the
per-set counter (min-reps-v2, detector.py) can draw the same boxes inside an Enter-press set.

Lifts are humps in the forearm angle measured from the board's USB direction (USB toward the
elbow, so it points up while the arm hangs; data/usb_axis.json). That zero needs no set start.
The board's mounting offset (~30 deg on dev) shifts the angle but barely changes hump heights,
which is all a lift needs. One big arm bend could be anything (grabbing a bottle), two within
BOX_GAP_S of each other start a set, and BOX_GAP_S without one ends it.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from src.counter import MIN_SPACING_S, SMOOTH_HZ, lift_span
from src.dataio import ACCEL, GYRO
from src.labels import segment_angle, track_down
from src.sync import lowpass

LIFT_MIN_DEG = 50.0  # a hump this tall is a lift. Dev: curl-set humps typically 120-138 deg tall,
                     # walking and strap fiddling 24-37; at 50 every dev1b curl hump passes
BOX_GAP_S = 12.0     # lifts whose tops are this close belong to one set. Same 12 s as min-reps-v1:
                     # the longest lift-to-lift time in any dev or held-out set is 9.7 s
PAD_S = 2.0          # a box starts this long before its first lift (and ends this long after its
                     # last), so the arm hangs at its start, like after an Enter press


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


def set_boxes(seg: pd.DataFrame, usb_axis) -> list[tuple[float, float]]:
    """The boxes inside one stretch of recording (an Enter-press set, or a box the replay found), as
    (start, end) in s from its first sample: the clock the counters' rep times use."""
    t = seg["t"].to_numpy()
    return [(b.start, b.end) for b in draw_boxes(find_lifts(t, stream_angle(seg, usb_axis), t - t[0]))]
