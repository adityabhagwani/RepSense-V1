"""Record a scripted session from the RepSense logger.

Two workers, two files:
  - Streamer (background thread) drains the serial port and appends every
    sample to data/raw/<session>.csv. It never waits on the keyboard.
  - The main thread walks you through the set script and appends one row per
    set to data/annotations/<session>.csv.
Both stamp times with the same Mac clock (host_t), which is how a set's rows
are found in the raw file later.

    python -m src.record --session s3 --script protocols/validation.csv
"""

import argparse
import csv
import sys
import threading
import time
from pathlib import Path

import serial
from serial.tools import list_ports

from src.dataio import ANNOTATION_COLUMNS, RAW_COLUMNS, annotation_path, parse_line, raw_path

ARDUINO_VID = 0x2341
FLUSH_EVERY = 25  # samples: 0.5 s at 50 Hz, the most a crash can lose
MIN_SET_SECONDS = 3.0  # a double-pressed Enter records <1 s; no real set is this short


class Streamer(threading.Thread):
    """Worker 1: reads board lines and appends every sample to the raw file."""

    def __init__(self, ser, raw_file, log=print):
        super().__init__(daemon=True)
        self.ser = ser
        self.raw_file = raw_file
        self.writer = csv.writer(raw_file)
        self.log = log
        self.lock = threading.Lock()
        self.running = True
        self.recording = False
        self.set_rows = []  # (seq, t_us) of each sample in the current set
        self.n_samples = 0
        self.got_data = threading.Event()

    def run(self):
        # Opening the port can land mid-line, and a cut line like "45,6789,..."
        # can still parse as a sample, so the first line is thrown away.
        self.ser.readline()
        while self.running:
            line = self.ser.readline()
            if not line:
                continue
            host_t = time.time()
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("#"):
                self.log(f"  board: {text.lstrip('# ')}")
                continue
            fields = parse_line(text)
            if fields is None:
                continue
            with self.lock:
                self.writer.writerow([f"{host_t:.4f}", *fields])
                self.n_samples += 1
                if self.recording:
                    self.set_rows.append((fields[0], fields[1]))
                if self.n_samples % FLUSH_EVERY == 0:
                    self.raw_file.flush()
            self.got_data.set()

    def start_set(self) -> float:
        with self.lock:
            self.set_rows = []
            self.recording = True
            return time.time()

    def stop_set(self) -> tuple[float, list]:
        with self.lock:
            self.recording = False
            self.raw_file.flush()
            return time.time(), self.set_rows

    def stop(self):
        self.running = False
        self.join(timeout=2)


def summarize(rows: list) -> dict:
    """Duration, effective rate and lost lines for one set, from the board's counters."""
    if len(rows) < 2:
        return {"n": len(rows), "seconds": 0.0, "hz": 0.0, "dropped": 0}
    # Per-step modulo handles micros() wrapping past 2^32.
    span_us = sum((b[1] - a[1]) % 2**32 for a, b in zip(rows, rows[1:]))
    dropped = sum(max(0, b[0] - a[0] - 1) for a, b in zip(rows, rows[1:]))
    seconds = span_us / 1e6
    return {"n": len(rows), "seconds": seconds, "hz": (len(rows) - 1) / seconds, "dropped": dropped}


def load_script(path: Path) -> list[dict]:
    """Set script CSV with columns exercise,reps,condition,notes."""
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    for i, row in enumerate(rows, 1):
        if not row.get("exercise") or not (row.get("reps") or "").isdigit():
            raise ValueError(f"{path}: row {i} needs an exercise and a whole-number reps")
    return rows


def next_set_idx(ann_file: Path) -> int:
    """Index of the first script row not yet recorded or skipped, for --resume."""
    if not ann_file.exists():
        return 0
    with open(ann_file, newline="") as f:
        done = [int(r["set_idx"]) for r in csv.DictReader(f) if r["status"] in ("ok", "skipped")]
    return max(done, default=0)  # set_idx is 1-based, so this is the next 0-based index


def ask_actual_reps(scripted: str, ask, say) -> int | None:
    """True rep count: Enter keeps the scripted count, a number overrides it, x discards."""
    while True:
        answer = ask(f"Actual reps [{scripted}] · x = discard > ").strip().lower()
        if answer == "":
            return int(scripted)
        if answer == "x":
            return None
        if answer.isdigit():
            return int(answer)
        say("  Type a number, press Enter to keep it, or x to discard.")


def run_session(streamer, script, session, participant, ann_file, start=0,
                min_seconds=MIN_SET_SECONDS, ask=input, say=print) -> dict:
    """Worker 2: walk through the script, writing one annotation row per set attempt."""
    counts = {"ok": 0, "skipped": 0, "discarded": 0}
    new_file = not ann_file.exists()
    with open(ann_file, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATION_COLUMNS)
        if new_file:
            writer.writeheader()

        def log_row(i, item, status, start_t=None, end_t=None, actual="", stats=None):
            stats = stats or {"n": 0, "dropped": 0}
            writer.writerow({
                "session": session, "participant": participant, "set_idx": i + 1,
                "exercise": item["exercise"], "scripted_reps": item["reps"], "actual_reps": actual,
                "condition": item.get("condition", ""), "notes": item.get("notes", ""),
                "start_host_t": f"{start_t:.4f}" if start_t else "",
                "end_host_t": f"{end_t:.4f}" if end_t else "",
                "n_samples": stats["n"], "dropped": stats["dropped"], "status": status,
            })
            f.flush()  # each label hits disk the moment it exists
            counts[status] += 1

        i = start
        while i < len(script):
            item = script[i]
            say(f"\n[{i + 1}/{len(script)}] {item['exercise']} · {item['reps']} reps · {item.get('condition', '')}")
            if item.get("notes"):
                say(f"       {item['notes']}")
            say("       Press Enter with your OTHER hand; let the sensor arm hang still for a second, then start.")
            choice = ask("Press Enter to START recording, then do the set · s = skip · q = quit > ").strip().lower()
            if choice == "q":
                break
            if choice == "s":
                log_row(i, item, "skipped")
                i += 1
                continue

            start_t = streamer.start_set()
            ask("● RECORDING. Do the set now. Press Enter when finished > ")
            end_t, rows = streamer.stop_set()
            stats = summarize(rows)
            warning = "  ⚠ lines lost" if stats["dropped"] else ""
            say(f"  {stats['seconds']:.1f} s · {stats['n']} samples · {stats['hz']:.1f} Hz · "
                f"{stats['dropped']} dropped{warning}")

            # Catch a mislabeled set now, while redoing it costs one keypress.
            too_short = stats["seconds"] < min_seconds
            if too_short and ask(f"Only {stats['seconds']:.1f} s recorded. Redo this set? [Y/n] > ").strip().lower() != "n":
                actual = None
            else:
                actual = ask_actual_reps(item["reps"], ask, say)
            if actual is None:
                log_row(i, item, "discarded", start_t, end_t, "", stats)
                say("  Discarded. Same set again.")
                continue
            log_row(i, item, "ok", start_t, end_t, actual, stats)
            i += 1
    return counts


def find_port() -> str:
    """First serial port with Arduino's USB vendor id."""
    ports = list_ports.comports()
    for p in ports:
        if p.vid == ARDUINO_VID:
            return p.device
    names = ", ".join(p.device for p in ports) or "none"
    sys.exit(f"No Arduino found (ports: {names}). Plug the board in or pass --port.")


def main():
    parser = argparse.ArgumentParser(description="Record a scripted RepSense session.")
    parser.add_argument("--session", required=True, help="session id, e.g. dev1 or s3")
    parser.add_argument("--script", required=True, type=Path, help="CSV: exercise,reps,condition,notes")
    parser.add_argument("--participant", default="p1")
    parser.add_argument("--port", help="serial port (default: first Arduino found)")
    parser.add_argument("--resume", action="store_true", help="continue a session after a crash or yanked cable")
    args = parser.parse_args()

    try:
        script = load_script(args.script)
    except (OSError, ValueError) as e:
        sys.exit(str(e))

    raw_file_path, ann_file = raw_path(args.session), annotation_path(args.session)
    if (raw_file_path.exists() or ann_file.exists()) and not args.resume:
        sys.exit(f"Session '{args.session}' already exists. Add --resume to continue it, or pick a new --session.")
    start = next_set_idx(ann_file) if args.resume else 0
    if start >= len(script):
        sys.exit(f"Session '{args.session}' already covers all {len(script)} sets in {args.script}.")
    raw_file_path.parent.mkdir(parents=True, exist_ok=True)
    ann_file.parent.mkdir(parents=True, exist_ok=True)

    ser = serial.Serial(args.port or find_port(), 115200, timeout=0.5)
    ser.reset_input_buffer()

    new_raw = not raw_file_path.exists()
    with open(raw_file_path, "a", newline="") as raw_file:
        if new_raw:
            csv.writer(raw_file).writerow(RAW_COLUMNS)
        streamer = Streamer(ser, raw_file)
        streamer.start()
        counts = None
        no_data = not streamer.got_data.wait(timeout=5)
        try:
            if not no_data:
                ser.write(b"i")  # ask the board for its info line
                print(f"Streaming to {raw_file_path}")
                counts = run_session(streamer, script, args.session, args.participant, ann_file, start)
        except KeyboardInterrupt:
            print("\nStopped. A set in progress was not labeled; --resume continues from it.")
        finally:
            streamer.stop()
            ser.close()

    if no_data:
        if new_raw:
            raw_file_path.unlink()  # a failed start leaves nothing behind
        sys.exit("No data from the board in 5 s. Is sensor_logger flashed? "
                 "Is the Arduino Serial Monitor holding the port?")
    if counts:
        print(f"\nSession {args.session}: {counts['ok']} ok, {counts['skipped']} skipped, "
              f"{counts['discarded']} discarded. {streamer.n_samples} samples this run.")


if __name__ == "__main__":
    main()
