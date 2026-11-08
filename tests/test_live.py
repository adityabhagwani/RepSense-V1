from src.boxes import BOX_GAP_S, Box, Lift
from src.live import RISK_FLAG, LiveSet, ready


def box(*tops):
    return Box(tops[0] - 3.0, tops[-1] + 3.0, [Lift(t - 0.8, t, t + 0.8, 100.0) for t in tops])


def test_a_box_closes_only_once_no_lift_can_join_it():
    b = box(10.0, 12.0, 14.0)
    assert ready([b], 14.0 + BOX_GAP_S, float("-inf")) == []
    assert ready([b], 14.0 + BOX_GAP_S + 0.1, float("-inf")) == [b]


def test_a_box_is_counted_once():
    b = box(10.0, 12.0, 14.0)
    assert ready([b], 100.0, counted_until=14.0) == []


def test_only_the_closed_boxes_after_the_last_counted_one_are_ready():
    old, done, open_ = box(10.0, 12.0), box(40.0, 42.0), box(80.0, 82.0)
    assert ready([old, done, open_], now=60.0, counted_until=12.0) == [done]


def test_the_label_flags_the_risky_sets():
    assert LiveSet(1, 0.0, 10.0, 9, RISK_FLAG).label == "check this set"
    assert LiveSet(1, 0.0, 10.0, 9, RISK_FLAG - 0.01).label == "likely right"
