"""Live demo: the screen says "Resting" or "Set in progress" as you train, and gives each set's final
numbers (reps, time, a risk label) about 12 s after its last rep. Wrist only.

The counters judge a hump against the set's typical hump, so a count is only known once the set is
over. A set is over when no lift can join its box any more: BOX_GAP_S after its last lift's top
(stream-v1's rule, src/boxes.py). Every STEP_S this reads the recording so far, finds the lifts and
boxes exactly as the replay does, and counts each box once it closes with the replay's own counter
(stream.count_box: counter-v2, detector-v4, min-reps-v1), so a live count is the replay's count of
the same box. A set shows as in progress from its second lift (one lift alone could be anything,
the box rule) until it closes. The label is count-error-v1 on that box: "check this set" when its risk is among the
riskiest 20% of the 3,000 spliced sets the model learned from, else "likely right". Experimental:
on p6/p7 (pre-registered) the model was inconclusive.

The data comes from a source with one method, read() -> (samples so far, now): the Nano over USB or
BLE (record.Streamer or ble.BleStreamer, which also save the session to data/raw/), or a saved
session replayed at any speed (no board needed; the demo's backup and its test).

    python -m src.live                          # live, from the Nano over USB
    python -m src.live --ble                    # live, over Bluetooth (firmware v2; board on a power bank)
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
ENDED_S = 6.0       # how long "Set N ended" stays up as the state before "Resting"
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
    seconds: float = 0.0     # first lift's start to last lift's end
    counted_at: float = 0.0  # host clock when the set closed

    @property
    def label(self) -> str:
        return "check this set" if self.risk >= RISK_FLAG else "likely right"


def ready(boxes: list[Box], now: float, counted_until: float) -> list[Box]:
    """Boxes no lift can join any more (BOX_GAP_S since their last top) and not yet counted: those
    starting after `counted_until`, the last top of the last box counted. Boxes come in time order."""
    return [b for b in boxes if now - b.lifts[-1].top > BOX_GAP_S and b.lifts[0].top > counted_until]


def state(boxes: list[Box], sets: list[LiveSet], now: float) -> tuple[str, str]:
    """(kind, text) for the screen: a set in progress (an open box), a set that just ended, or resting."""
    open_boxes = [b for b in boxes if now - b.lifts[-1].top <= BOX_GAP_S]
    if open_boxes:
        return "in progress", f"Set in progress · last lift {now - open_boxes[-1].lifts[-1].top:.0f} s ago"
    if sets and now - sets[-1].counted_at <= ENDED_S:
        s = sets[-1]
        return "ended", f"Set {s.n} ended · {s.reps} reps · {s.seconds:.0f} s · {s.label}"
    return "resting", "Resting"


class Board:
    """The Nano over USB (record.Streamer) or BLE (ble.BleStreamer): either writes every sample to
    data/raw/<session>.csv as it comes."""

    def __init__(self, session: str, port: str | None, ble: bool = False):
        self.session, path = session, raw_path(session)
        if path.exists():
            raise SystemExit(f"Session '{session}' already exists; pick a new --session.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.raw_file = open(path, "w", newline="")
        csv.writer(self.raw_file).writerow(RAW_COLUMNS)
        self.ser = None
        if ble:
            from src.ble import SCAN_S, BleStreamer

            self.streamer, wait_s = BleStreamer(self.raw_file), SCAN_S + 10
        else:
            import serial
            from src.record import Streamer, find_port

            self.ser = serial.Serial(port or find_port(), 115200, timeout=0.5)
            self.ser.reset_input_buffer()
            self.streamer, wait_s = Streamer(self.ser, self.raw_file, log=lambda msg: None), 5
        self.streamer.start()
        if not self.streamer.got_data.wait(timeout=wait_s):
            self.close()
            path.unlink()
            raise SystemExit(f"No data from the board in {wait_s:.0f} s. Is sensor_logger flashed"
                             f"{' (v2) and powered' if ble else '? Is the Serial Monitor open'}?")
        self.done = False

    def read(self) -> tuple[pd.DataFrame, float]:
        with self.streamer.lock:
            self.raw_file.flush()
        return load_session(self.session), time.time()

    def close(self):
        self.streamer.stop()
        if self.ser:
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
        sets.append(LiveSet(len(sets) + 1, b.start, b.end, reps, float(error_model["model"].predict_proba(signs)[0, 1]),
                            b.lifts[-1].end - b.lifts[0].start, now))
        counted_until = b.lifts[-1].top
        s = sets[-1]
        print(f"Set {s.n} ended: {s.reps} reps · {s.seconds:.0f} s · {s.label} (risk {s.risk:.2f})")
    return boxes, counted_until


class Screen:
    """The state on top ("Resting", "Set in progress", "Set N ended"), then two panels: the last WINDOW_S of
    forearm angle with sets shaded, and the list of finished sets."""
    COLORS = {"resting": "#6e7781", "in progress": "#bc4c00", "ended": "#0969da"}

    def __init__(self, title: str):
        import matplotlib.pyplot as plt

        self.plt = plt
        plt.ion()
        self.fig, (self.ax, self.board) = plt.subplots(1, 2, figsize=(13, 5.4), gridspec_kw={"width_ratios": [2.4, 1]})
        self.fig.subplots_adjust(top=0.74, bottom=0.12, left=0.07, right=0.98)
        self.fig.suptitle(title, x=0.01, y=0.98, ha="left", fontsize=11, color="#6e7781")
        self.state = self.fig.text(0.01, 0.85, "", fontsize=22, fontweight="bold", va="center")

    def draw(self, df: pd.DataFrame, now: float, boxes: list[Box], sets: list[LiveSet], angle: np.ndarray | None):
        ax, board = self.ax, self.board
        kind, text = state(boxes, sets, now)
        self.state.set_text(text)
        self.state.set_color(self.COLORS[kind])
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
            ax.axvspan(b.start - now, 0, color="#eb6834", alpha=0.12, lw=0)
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
            board.text(0, 0.86 - 0.11 * k, f"Set {s.n}   {s.reps} reps   {s.seconds:.0f} s", fontsize=13, va="top",
                       color="#1f2328", transform=board.transAxes)
            board.text(0.66, 0.86 - 0.11 * k, s.label, fontsize=11, va="top", color=color, transform=board.transAxes)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def run(source, step: float, speed: float, show: bool) -> list[LiveSet]:
    usb_axis = json.loads(USB_AXIS_FILE.read_text())["axis"]
    model = joblib.load(VERSIONS["v4"]["path"])
    error_model = joblib.load(ERROR_MODEL)
    screen = Screen(f"RepSense · {'replay of ' + source.session if isinstance(source, Recording) else 'live'} · wrist only"
                    " · a set's final numbers appear 12 s after its last rep") if show else None
    sets, counted_until, shown = [], -np.inf, None
    try:
        while not source.done:
            tick = time.time()
            df, now = source.read()
            boxes, angle = [], None
            if len(df) > 1 and df["host_t"].iloc[-1] - df["host_t"].iloc[0] >= MIN_DATA_S:
                angle = stream_angle(df, usb_axis)
                boxes, counted_until = count_closed(df, angle, now, usb_axis, model, error_model, sets, counted_until)
            kind = state(boxes, sets, now)[0]
            if kind == "in progress" and shown != kind:
                print("Set in progress")
            shown = kind
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
    parser.add_argument("--ble", action="store_true", help="read the Nano over Bluetooth (firmware v2), not USB")
    parser.add_argument("--no-window", action="store_true", help="print the sets only")
    args = parser.parse_args()
    if args.replay:
        source = Recording(args.replay, args.speed)
    else:
        source = Board(args.session or datetime.now().strftime("live_%Y%m%d_%H%M%S"), args.port, args.ble)
        print(f"Streaming to {raw_path(source.session)}. A set's final numbers appear 12 s after its last rep. Ctrl-C stops.")
    sets = run(source, args.step, args.speed, not args.no_window)
    if sets:
        out = REPORTS_DIR / f"{source.session}_live_sets.csv"
        pd.DataFrame([{**s.__dict__, "label": s.label} for s in sets]).to_csv(out, index=False)
        print(f"{len(sets)} sets; list: {out.relative_to(REPORTS_DIR.parent)}")


if __name__ == "__main__":
    main()
