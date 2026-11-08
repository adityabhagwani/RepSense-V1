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


# The screen's colors: the dataviz reference palette, dark mode (blue and orange pass its checks as a pair
# on SURFACE). Status chips pair a symbol with the words, so the label never rests on color alone.
PAGE, SURFACE, RAISED = "#0d0d0d", "#1a1a19", "#232322"
INK, INK2, MUTED, GRID, BADGE_GRAY = "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835"
BLUE, ORANGE, GOOD, WARNING = "#3987e5", "#d95926", "#0ca30c", "#fab219"


class Screen:
    """Dark, three cards: the state on top (a badge: "Resting", "Set in progress" with a bar filling
    over the BOX_GAP_S to the set's end, or "Set N ended" with its numbers), the last WINDOW_S of
    forearm angle with sets shaded, and the finished sets, newest first."""

    def __init__(self, subtitle: str):
        import matplotlib.pyplot as plt
        from matplotlib.patches import FancyBboxPatch

        self.plt = plt
        plt.ion()
        plt.rcParams["font.family"] = ["Helvetica Neue", "DejaVu Sans"]  # DejaVu Sans has the check mark
        self.fig = fig = plt.figure(figsize=(13, 6.2), facecolor=PAGE)
        fig.text(0.025, 0.935, "RepSense", fontsize=17, fontweight="bold", color=INK)
        fig.text(0.128, 0.937, subtitle, fontsize=11, color=MUTED)
        for x, y, w, h in [(0.02, 0.70, 0.96, 0.19), (0.02, 0.05, 0.665, 0.62), (0.70, 0.05, 0.28, 0.62)]:
            fig.patches.append(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.012",
                                              transform=fig.transFigure, facecolor=SURFACE, edgecolor="none", zorder=-1))
        self.state_ax = fig.add_axes([0.035, 0.715, 0.93, 0.16])
        self.ax = fig.add_axes([0.07, 0.13, 0.595, 0.49])
        self.list_ax = fig.add_axes([0.715, 0.07, 0.255, 0.58])

    @staticmethod
    def chip(ax, x: float, y: float, label: str, size: float):
        ok = label == "likely right"
        ax.text(x, y, ("✓  " if ok else "!  ") + label, fontsize=size, fontweight="bold", va="center",
                color="#ffffff" if ok else "#0b0b0b",
                bbox=dict(boxstyle="round,pad=0.35,rounding_size=0.5", facecolor=GOOD if ok else WARNING, edgecolor="none"))

    def draw_state(self, kind: str, boxes: list[Box], sets: list[LiveSet], now: float):
        a = self.state_ax
        a.clear()
        a.axis("off")
        a.set_xlim(0, 1)
        a.set_ylim(0, 1)
        text, fill, ink = {"resting": ("Resting", BADGE_GRAY, INK2), "in progress": ("Set in progress", ORANGE, "#ffffff"),
                           "ended": (f"Set {sets[-1].n} ended" if sets else "", BLUE, "#ffffff")}[kind]
        a.text(0.0, 0.62, text, fontsize=24, fontweight="bold", color=ink, va="center",
               bbox=dict(boxstyle="round,pad=0.45,rounding_size=0.6", facecolor=fill, edgecolor="none"))
        if kind == "in progress":
            since = now - [b for b in boxes if now - b.lifts[-1].top <= BOX_GAP_S][-1].lifts[-1].top
            a.text(0.33, 0.70, f"last lift {since:.0f} s ago", fontsize=15, color=INK, va="center")
            a.text(0.33, 0.40, f"the set ends after {BOX_GAP_S:.0f} s with no lift", fontsize=11, color=MUTED, va="center")
            a.barh(0.12, 1.0, height=0.10, color=GRID)
            a.barh(0.12, min(since / BOX_GAP_S, 1.0), height=0.10, color=ORANGE)
        elif kind == "ended":
            s = sets[-1]
            a.text(0.30, 0.62, f"{s.reps}", fontsize=34, fontweight="bold", color=INK, va="center")
            a.text(0.36, 0.62, f"reps   ·   {s.seconds:.0f} s", fontsize=16, color=INK2, va="center")
            self.chip(a, 0.60, 0.62, s.label, 13)
        else:
            a.text(0.20, 0.62, "waiting for the next set", fontsize=14, color=MUTED, va="center")

    def draw(self, df: pd.DataFrame, now: float, boxes: list[Box], sets: list[LiveSet], angle: np.ndarray | None):
        self.draw_state(state(boxes, sets, now)[0], boxes, sets, now)
        ax = self.ax
        ax.clear()
        ax.set_facecolor(SURFACE)
        for y in (0, 90, 180):
            ax.axhline(y, color=GRID, lw=1, zorder=0)
        for s in sets:
            if s.end >= now - WINDOW_S:
                ax.axvspan(max(s.start - now, -WINDOW_S), s.end - now, color=RAISED, lw=0, zorder=0)
                ax.text(max(s.start - now, -WINDOW_S) + 0.4, 176, f"Set {s.n} · {s.reps} reps", fontsize=10,
                        color=INK2, va="top")
        for b in boxes:
            if now - b.lifts[-1].top <= BOX_GAP_S:
                ax.axvspan(max(b.start - now, -WINDOW_S), 0, color=ORANGE, alpha=0.14, lw=0, zorder=0)
        if angle is not None and len(df):
            x = df["host_t"].to_numpy() - now
            keep = x >= -WINDOW_S
            ax.fill_between(x[keep], -10, angle[keep], color=BLUE, alpha=0.10, lw=0, zorder=1)
            ax.plot(x[keep], angle[keep], color=BLUE, lw=2, solid_capstyle="round", solid_joinstyle="round", zorder=2)
            ax.plot([x[keep][-1]], [angle[keep][-1]], "o", ms=9, color=BLUE, mec=SURFACE, mew=2, zorder=3)
        ax.set_xlim(-WINDOW_S, 0)
        ax.set_ylim(-10, 185)
        ax.set_yticks([0, 90, 180], ["0°", "90°", "180°"])
        ax.set_xticks(range(-30, 1, 5), [f"{t} s" if t else "now" for t in range(-30, 1, 5)])
        ax.tick_params(colors=MUTED, labelsize=10, length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title("Forearm angle, from the wrist", loc="left", fontsize=12, color=INK2, pad=10)
        la = self.list_ax
        la.clear()
        la.axis("off")
        la.set_xlim(0, 1)
        la.set_ylim(0, 1)
        la.text(0, 0.97, "Sets", fontsize=13, fontweight="bold", color=INK, va="top")
        if not sets:
            la.text(0, 0.84, "No sets yet", fontsize=11, color=MUTED, va="top")
        for k, s in enumerate(reversed(sets[-6:])):
            y = 0.84 - 0.145 * k
            la.text(0, y, f"{s.reps}", fontsize=22, fontweight="bold", color=INK, va="center")
            la.text(0.17, y + 0.025, f"Set {s.n}", fontsize=11, color=INK2, va="center")
            la.text(0.17, y - 0.03, f"{s.seconds:.0f} s", fontsize=10, color=MUTED, va="center")
            self.chip(la, 0.48, y, s.label, 10)
        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)


def run(source, step: float, speed: float, show: bool) -> list[LiveSet]:
    usb_axis = json.loads(USB_AXIS_FILE.read_text())["axis"]
    model = joblib.load(VERSIONS["v4"]["path"])
    error_model = joblib.load(ERROR_MODEL)
    screen = Screen(f"{'replay of ' + source.session if isinstance(source, Recording) else 'live'} · wrist only") if show else None
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
