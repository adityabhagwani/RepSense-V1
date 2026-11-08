"""Live demo: each set's count from the wrist, about 12 s after its last rep, with a risk label.

The counters judge a hump against the set's typical hump, so a count is only known once the set is
over. A set is over when no lift can join its box any more: BOX_GAP_S after its last lift's top
(stream-v1's rule, src/boxes.py). Every STEP_S this reads the recording so far, finds the lifts and
boxes exactly as the replay does, and counts each box once it closes with the replay's own counter
(stream.count_box: counter-v2, detector-v4, min-reps-v1), so a live count is the replay's count of
the same box. The label is count-error-v1 on that box: "check this set" when its risk is among the
riskiest 20% of the 3,000 spliced sets the model learned from, else "likely right". Experimental:
on p6/p7 (pre-registered) the model was inconclusive.

The data comes from a source with one method, read() -> (samples so far, now): the Nano over USB
(record.Streamer, which also saves the session to data/raw/), or a saved session replayed at any
speed (no board needed; the demo's backup and its test).

    python -m src.live                          # live, from the Nano
    python -m src.live --replay p6_s1 --speed 4 # a saved session through the same loop
"""

import argparse
import csv
import json
import time
from dataclasses import dataclass
from datetime import datetime

import joblib
import numpy as np
import pandas as pd

from src.boxes import BOX_GAP_S, Box, draw_boxes, find_lifts, stream_angle
from src.dataio import RAW_COLUMNS, load_session, raw_path, slice_set
from src.detector import MODELS_DIR, VERSIONS
from src.error_model import symptoms
from src.labels import USB_AXIS_FILE
from src.stream import count_box
from src.sync import REPORTS_DIR

STEP_S = 0.5        # how often the recording is re-read and closed boxes counted
WINDOW_S = 30.0     # seconds of wrist angle on screen
MIN_DATA_S = 3.0    # nothing to look at before this much recording
ERROR_MODEL = MODELS_DIR / "count_error_v1.joblib"
RISK_FLAG = 0.1346  # count-error-v1's 80th-percentile risk on the 3,000 spliced sets it was trained on
                    # (reports/error_rows_spliced_1_v4.csv): 55% of the sets above it were miscounted, 2.4% below


@dataclass
class LiveSet:
    n: int
    start: float     # host clock
    end: float
    reps: int
    risk: float

    @property
    def label(self) -> str:
        return "check this set" if self.risk >= RISK_FLAG else "likely right"


def ready(boxes: list[Box], now: float, counted_until: float) -> list[Box]:
    """Boxes no lift can join any more (BOX_GAP_S since their last top) and not yet counted: those
    starting after `counted_until`, the last top of the last box counted. Boxes come in time order."""
    return [b for b in boxes if now - b.lifts[-1].top > BOX_GAP_S and b.lifts[0].top > counted_until]


class Board:
    """The Nano over USB. record.Streamer writes every sample to data/raw/<session>.csv as it comes."""

    def __init__(self, session: str, port: str | None):
        import serial
        from src.record import Streamer, find_port

        self.session, path = session, raw_path(session)
        if path.exists():
            raise SystemExit(f"Session '{session}' already exists; pick a new --session.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_file = open(path, "w", newline="")
        csv.writer(self.raw_file).writerow(RAW_COLUMNS)
        self.ser = serial.Serial(port or find_port(), 115200, timeout=0.5)
        self.ser.reset_input_buffer()
        self.streamer = Streamer(self.ser, self.raw_file, log=lambda msg: None)
        self.streamer.start()
        if not self.streamer.got_data.wait(timeout=5):
            self.close()
            path.unlink()
            raise SystemExit("No data from the board in 5 s. Is sensor_logger flashed? Is the Serial Monitor open?")
        self.done = False

    def read(self) -> tuple[pd.DataFrame, float]:
        with self.streamer.lock:
            self.raw_file.flush()
        return load_session(self.session), time.time()

    def close(self):
        self.streamer.stop()
        self.ser.close()
        self.raw_file.close()


class Recording:
    """A saved session, replayed: now runs from its first sample at `speed` times real time."""

    def __init__(self, session: str, speed: float):
        self.session, self.speed = session, speed
        self.df = load_session(session)
        self.t0, self.last = self.df["host_t"].iloc[0], self.df["host_t"].iloc[-1]
        self.now = self.t0
        self.done = False

    def read(self) -> tuple[pd.DataFrame, float]:
        if self.now > self.last + BOX_GAP_S + STEP_S:  # the last box has had time to close
            self.done = True
        return self.df[self.df["host_t"] <= self.now], self.now

    def advance(self, step: float):
        self.now += step

    def close(self):
        pass


def count_closed(df: pd.DataFrame, angle: np.ndarray, now: float, usb_axis, model, error_model, sets: list[LiveSet],
                 counted_until: float) -> tuple[list[Box], float]:
    """Find the boxes in the recording so far (`angle`: its stream_angle), count the ones that just
    closed (appending to `sets`), and return all boxes plus the new counted_until."""
    boxes = draw_boxes(find_lifts(df["t"].to_numpy(), angle, df["host_t"].to_numpy()))
    for b in ready(boxes, now, counted_until):
        seg = slice_set(df, {"start_host_t": b.start, "end_host_t": b.end})
        reps = count_box(seg, model, usb_axis, "v4")["min_reps"]
        signs = pd.DataFrame([symptoms(seg, model, usb_axis, "v4")])[error_model["signs"]]
        sets.append(LiveSet(len(sets) + 1, b.start, b.end, reps, float(error_model["model"].predict_proba(signs)[0, 1])))
        counted_until = b.lifts[-1].top
        s = sets[-1]
        print(f"Set {s.n}: {s.reps} reps · {s.label} (risk {s.risk:.2f}) · counted {now - b.lifts[-1].top:.0f} s after the last rep")
    return boxes, counted_until


class Screen:
    """Two panels: the last WINDOW_S of wrist angle with sets shaded, and the list of counted sets."""

    def __init__(self, title: str):
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.fig, (self.ax, self.board) = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw={"width_ratios": [2.4, 1]})
        self.fig.suptitle(title, x=0.01, ha="left", fontsize=12, color="#1f2328")

    def draw(self, df: pd.DataFrame, now: float, boxes: list[Box], sets: list[LiveSet], angle: np.ndarray | None):
        ax, board = self.ax, self.board
        ax.clear(), board.clear()
        if angle is not None and len(df):
            x = df["host_t"].to_numpy() - now
            keep = x >= -WINDOW_S
            ax.plot(x[keep], angle[keep], color="#2a78d6", lw=1.4)
        for s in sets:
            if s.end >= now - WINDOW_S:
                ax.axvspan(s.start - now, s.end - now, color="#2a78d6", alpha=0.12, lw=0)
                ax.text(max(s.start - now, -WINDOW_S), 172, f"Set {s.n}: {s.reps}", fontsize=10, color="#1f2328")
        open_box = [b for b in boxes if now - b.lifts[-1].top <= BOX_GAP_S]
        for b in open_box:
            ax.axvspan(b.start - now, 0, color="#eb6834", alpha=0.10, lw=0)
            ax.text(b.start - now, 172, f"set in progress · count {BOX_GAP_S - (now - b.lifts[-1].top):.0f} s after the last rep",
                    fontsize=9, color="#9a3a10")
        ax.set_xlim(-WINDOW_S, 0)
        ax.set_ylim(-10, 185)
        ax.set_xlabel("seconds ago", color="#6e7781")
        ax.set_ylabel("forearm angle (deg, wrist only)", color="#6e7781")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        board.axis("off")
        board.text(0, 1, "Sets", fontsize=13, fontweight="bold", va="top", color="#1f2328", transform=board.transAxes)
        for k, s in enumerate(sets[-8:]):
            color = "#9a3a10" if s.label == "check this set" else "#1a7f37"
            board.text(0, 0.86 - 0.11 * k, f"Set {s.n}   {s.reps} reps", fontsize=13, va="top", color="#1f2328",
                       transform=board.transAxes)
            board.text(0.62, 0.86 - 0.11 * k, s.label, fontsize=11, va="top", color=color, transform=board.transAxes)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def run(source, step: float, speed: float, show: bool) -> list[LiveSet]:
    usb_axis = json.loads(USB_AXIS_FILE.read_text())["axis"]
    model = joblib.load(VERSIONS["v4"]["path"])
    error_model = joblib.load(ERROR_MODEL)
    screen = Screen(f"RepSense · {'replay of ' + source.session if isinstance(source, Recording) else 'live'}"
                    " · counts appear 12 s after a set's last rep") if show else None
    sets, counted_until = [], -np.inf
    try:
        while not source.done:
            tick = time.time()
            df, now = source.read()
            boxes, angle = [], None
            if len(df) > 1 and df["host_t"].iloc[-1] - df["host_t"].iloc[0] >= MIN_DATA_S:
                angle = stream_angle(df, usb_axis)
                boxes, counted_until = count_closed(df, angle, now, usb_axis, model, error_model, sets, counted_until)
            if screen:
                screen.draw(df, now, boxes, sets, angle)
            if isinstance(source, Recording):
                source.advance(step)
                if np.isfinite(speed):
                    time.sleep(max(0.0, step / speed - (time.time() - tick)))
            else:
                time.sleep(max(0.0, step - (time.time() - tick)))
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        source.close()
    return sets


def main():
    parser = argparse.ArgumentParser(description="Live set counts from the wrist, with a risk label.")
    parser.add_argument("--replay", metavar="SESSION", help="replay a saved session instead of reading the Nano")
    parser.add_argument("--speed", type=float, default=1.0, help="replay speed (inf: as fast as it computes)")
    parser.add_argument("--step", type=float, default=STEP_S, help="seconds between reads")
    parser.add_argument("--session", help="name for the live recording (default live_<date>_<time>)")
    parser.add_argument("--port", help="serial port (default: first Arduino found)")
    parser.add_argument("--no-window", action="store_true", help="print the sets only")
    args = parser.parse_args()
    if args.replay:
        source = Recording(args.replay, args.speed)
    else:
        source = Board(args.session or datetime.now().strftime("live_%Y%m%d_%H%M%S"), args.port)
        print(f"Streaming to {raw_path(source.session)}. Curl; each count appears 12 s after the set's last rep. Ctrl-C stops.")
    sets = run(source, args.step, args.speed, not args.no_window)
    if sets:
        out = REPORTS_DIR / f"{source.session}_live_sets.csv"
        pd.DataFrame([{**s.__dict__, "label": s.label} for s in sets]).to_csv(out, index=False)
        print(f"{len(sets)} sets; list: {out.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
