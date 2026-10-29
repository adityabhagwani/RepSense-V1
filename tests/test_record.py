import csv
import time

from src.dataio import RAW_COLUMNS, parse_line
from src.record import Streamer, next_set_idx, run_session, summarize

PERIOD_US = 20_000


class FakeBoard:
    """Stands in for the serial port: a cut first line, an info line, then samples."""

    def __init__(self, first_seq=1000):
        # The cut line parses as a valid sample on purpose: it must still be discarded.
        self.lines = [b"5,6789,1,2,3,4,5,6\n", b"# repsense sensor_logger v1 hz=50\n"]
        self.seq = first_seq

    def readline(self):
        time.sleep(0.001)
        if self.lines:
            return self.lines.pop(0)
        self.seq += 1
        return f"{self.seq},{self.seq * PERIOD_US % 2**32},10,-20,1000,5,-5,0\n".encode()

    def write(self, data):
        pass


def scripted_answers(answers):
    """An input() replacement that holds each recording open long enough to get samples."""
    it = iter(answers)

    def ask(prompt):
        if "RECORDING" in prompt:
            time.sleep(0.15)
        return next(it)

    return ask


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def test_parse_line():
    assert parse_line("1,2,3,4,5,6,7,8\n") == [1, 2, 3, 4, 5, 6, 7, 8]
    assert parse_line("# info line") is None
    assert parse_line("1,2,3") is None
    assert parse_line("1,2,x,4,5,6,7,8") is None
    assert parse_line("") is None


def test_summarize_counts_drops_across_micros_wrap():
    rows = [(0, 2**32 - PERIOD_US), (1, 0), (3, 2 * PERIOD_US)]  # seq 2 lost, clock wrapped
    stats = summarize(rows)
    assert stats["dropped"] == 1
    assert abs(stats["seconds"] - 0.06) < 1e-9


def test_session_end_to_end(tmp_path):
    script = [
        {"exercise": "curl", "reps": "8", "condition": "normal", "notes": ""},
        {"exercise": "curl", "reps": "8", "condition": "pause", "notes": "pause at rep 4"},
        {"exercise": "press", "reps": "6", "condition": "partial", "notes": ""},
    ]
    answers = [
        "", "", "",          # set 1: start, stop, keep 8
        "s",                 # set 2: skip
        "", "", "x",         # set 3: start, stop, discard
        "", "", "7",         # set 3 again: start, stop, actually did 7
    ]
    raw_file_path, ann_file = tmp_path / "raw.csv", tmp_path / "ann.csv"
    logged = []
    with open(raw_file_path, "w", newline="") as raw_file:
        csv.writer(raw_file).writerow(RAW_COLUMNS)
        streamer = Streamer(FakeBoard(), raw_file, log=logged.append)
        streamer.start()
        # The fake board's clock makes 0.15 s look like ~2.8 s; turn the short-set guard off here.
        counts = run_session(streamer, script, "t1", "p1", ann_file, min_seconds=0,
                             ask=scripted_answers(answers), say=lambda *_: None)
        streamer.stop()

    assert counts == {"ok": 2, "skipped": 1, "discarded": 1}
    ann = read_csv(ann_file)
    assert [(r["set_idx"], r["status"], r["actual_reps"]) for r in ann] == [
        ("1", "ok", "8"), ("2", "skipped", ""), ("3", "discarded", ""), ("3", "ok", "7"),
    ]

    raw = read_csv(raw_file_path)
    seqs = [int(r["seq"]) for r in raw]
    assert 5 not in seqs                                   # cut first line was discarded
    assert seqs == list(range(seqs[0], seqs[0] + len(seqs)))  # nothing lost or duplicated
    assert any("repsense" in m for m in logged)            # info line logged, not saved

    # The shared clock is the join: a set's raw rows are the ones between its start and end.
    for row in (r for r in ann if r["status"] == "ok"):
        start, end = float(row["start_host_t"]), float(row["end_host_t"])
        in_set = sum(start <= float(r["host_t"]) <= end for r in raw)
        assert int(row["n_samples"]) > 0
        assert abs(in_set - int(row["n_samples"])) <= 2


def test_short_set_offers_redo(tmp_path):
    script = [{"exercise": "curl", "reps": "8", "condition": "normal", "notes": ""}]
    answers = [
        "", "", "",          # start, stop, too short -> Enter = redo (discarded)
        "", "", "n", "",     # start, stop, too short -> n = keep it, then keep 8 reps
    ]
    ann_file = tmp_path / "ann.csv"
    with open(tmp_path / "raw.csv", "w", newline="") as raw_file:
        csv.writer(raw_file).writerow(RAW_COLUMNS)
        streamer = Streamer(FakeBoard(), raw_file, log=lambda *_: None)
        streamer.start()
        counts = run_session(streamer, script, "t1", "p1", ann_file, min_seconds=1000,
                             ask=scripted_answers(answers), say=lambda *_: None)
        streamer.stop()

    assert counts == {"ok": 1, "skipped": 0, "discarded": 1}
    assert [(r["set_idx"], r["status"], r["actual_reps"]) for r in read_csv(ann_file)] == [
        ("1", "discarded", ""), ("1", "ok", "8"),
    ]


def test_resume_starts_after_last_finished_set(tmp_path):
    ann_file = tmp_path / "ann.csv"
    assert next_set_idx(ann_file) == 0
    with open(ann_file, "w", newline="") as f:
        f.write("set_idx,status\n1,ok\n2,skipped\n3,discarded\n")
    assert next_set_idx(ann_file) == 2  # set 3 was only discarded, so redo it
