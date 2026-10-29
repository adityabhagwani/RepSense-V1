"""Hand-marking tests on a made-up session in a temp data folder.

Sets: 1 sync ok, 2 curl ok, 3 skipped, 4 discarded then redone ok, 5 discarded only, 6 exhaustion ok.
"""

import csv
import sys

import pytest

from src.dataio import ANNOTATION_COLUMNS
from src.mark import add_marks, current_marks, main, manual_path, show

SESSION = "s1"
SETS = [(1, "sync", "gesture", "ok"), (2, "curl", "normal", "ok"), (3, "curl", "partial", "skipped"),
        (4, "curl", "slow", "discarded"), (4, "curl", "slow", "ok"), (5, "curl", "fast", "discarded"),
        (6, "curl", "exhaustion", "ok")]


@pytest.fixture
def data_dir(tmp_path):
    path = tmp_path / "annotations" / f"{SESSION}.csv"
    path.parent.mkdir()
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS, restval="")
        writer.writeheader()
        for set_idx, exercise, condition, status in SETS:
            writer.writerow({"session": SESSION, "set_idx": set_idx, "exercise": exercise,
                             "condition": condition, "status": status})
    return tmp_path


def in_force(data_dir):
    marks = current_marks(SESSION, data_dir)
    return {(s, m): k for s, m, k in zip(marks["set_idx"], marks["movement"], marks["kind"])}


def test_append_only_and_the_latest_mark_wins(data_dir):
    add_marks(SESSION, 6, [11, 12], "half", note="grinding", data_dir=data_dir)
    before = manual_path(SESSION, data_dir).read_text()
    add_marks(SESSION, 6, [12], "drift", by="observer", data_dir=data_dir)
    after = manual_path(SESSION, data_dir).read_text()
    assert after.startswith(before)                     # nothing rewritten, only appended
    assert len(after.splitlines()) == 1 + 3             # header + every judgment on record
    assert in_force(data_dir) == {(6, 11): "half", (6, 12): "drift"}


def test_ok_clears_a_mark(data_dir):
    add_marks(SESSION, 2, [3], "other", data_dir=data_dir)
    add_marks(SESSION, 2, [3], "ok", data_dir=data_dir)
    assert in_force(data_dir) == {}
    add_marks(SESSION, 2, [3], "half", data_dir=data_dir)  # and can be marked again after clearing
    assert in_force(data_dir) == {(2, 3): "half"}


@pytest.mark.parametrize("set_idx", [99, 3, 5])  # unknown, skipped, discarded only
def test_unknown_or_not_ok_set_is_refused(data_dir, set_idx):
    with pytest.raises(ValueError, match=f"set {set_idx} isn't a recorded ok set"):
        add_marks(SESSION, set_idx, [2], "half", data_dir=data_dir)
    assert not manual_path(SESSION, data_dir).exists()


def test_redone_set_counts_as_ok(data_dir):
    add_marks(SESSION, 4, [2], "half", data_dir=data_dir)
    assert in_force(data_dir) == {(4, 2): "half"}


@pytest.mark.parametrize("movements, kind", [([0], "half"), ([-1], "half"), ([2, 0], "half"),
                                             ([], "half"), ([2], "partial")])
def test_bad_movement_or_kind_is_refused(data_dir, movements, kind):
    with pytest.raises(ValueError):
        add_marks(SESSION, 2, movements, kind, data_dir=data_dir)
    assert not manual_path(SESSION, data_dir).exists()


def test_session_without_annotations_is_refused(data_dir):
    with pytest.raises(ValueError, match="no annotations"):
        add_marks("never_recorded", 1, [2], "half", data_dir=data_dir)


def test_show_reflects_corrections(data_dir):
    assert show(SESSION, data_dir) == f"{SESSION}: no hand marks"
    add_marks(SESSION, 6, [11, 12], "half", note="grinding", data_dir=data_dir)
    add_marks(SESSION, 6, [12], "drift", data_dir=data_dir)
    add_marks(SESSION, 2, [4], "other", data_dir=data_dir)
    add_marks(SESSION, 2, [4], "ok", data_dir=data_dir)
    text = show(SESSION, data_dir)
    assert "set 6 curl exhaustion: 11:half (grinding), 12:drift" in text
    assert "set 2" not in text


def test_command_line_errors_have_no_traceback(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["mark", "no_such_session", "2", "3", "--kind", "half"])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert str(exit_info.value.code).startswith("Not marked: session 'no_such_session' has no annotations")

    monkeypatch.setattr(sys, "argv", ["mark", "no_such_session", "2"])  # no movement, no --kind
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2 and "give a set" in capsys.readouterr().err

    monkeypatch.setattr(sys, "argv", ["mark", "no_such_session", "--show"])
    main()
    assert "no_such_session: no hand marks" in capsys.readouterr().out
