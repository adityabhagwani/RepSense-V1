"""Write a session's set script for the receiver: which exercise, how many reps, what to do badly.

A session: sync gesture, the shuffled body, one to-exhaustion curl set, sync gesture.
The exhaustion set always comes last because fatigue would contaminate anything after it.

The body (curls by default: the upper-arm phone can only check curls; raises and presses are
an optional "unfamiliar exercise" test via --exercises, normal sets only):
  - normal, slow, fast, pause and alternating sets, one each; each has a 50% chance of one
    extra half rep (lift or lower halfway, chosen at random). A pause set holds one lift halfway.
  - 2 partial sets with 4 half reps each: 2 lift halfway, 2 lower halfway
  - 2 drift sets with 3 drift reps each
  - DISTRACTORS: movement that isn't reps (reps = 0); the counter should say 0
  - any curl lift set: a REST_CHANCE chance of one rest in the middle (after a lift that isn't the
    first or last, the arm hangs still for 5, 10 or 15 s, then the set carries on). Added 11-04:
    spliced sets showed a rest at the bottom breaks the count most often, and no recorded set had one

`reps` is full reps only: the true count. A movement is any lift, half reps included, numbered
from 1. `marked` lists the scripted ones as movement:kind (see dataio), so analysis never parses
the notes. Marks never land on the first or last movement, where set slicing and peak finding are
least reliable. Rep counts are random (a counter can't win by guessing a fixed number) and the
body is shuffled (fatigue isn't tied to one condition).

The CSV is the record, not the seed: numpy doesn't promise the same random stream across
versions. It is never overwritten, because annotations point into it by row number.

    python -m src.make_script --session p1_s1 --seed 11
"""

import argparse
import csv
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from src.dataio import (HALF_KINDS, MARK_KINDS, REST_KINDS, REST_SECONDS, annotation_path, format_marked,
                        movement_count, parse_marked)

PROTOCOLS_DIR = Path(__file__).resolve().parent.parent / "protocols"
FIELDS = ["exercise", "reps", "condition", "notes", "marked"]

EXERCISES = ["curl"]
MIN_REPS, MAX_REPS = 5, 12
NORMAL_SETS = 1                   # per exercise
CURL_STYLES = ["slow", "fast", "pause", "alternating"]  # one set each, curls only
EXTRA_HALF_CHANCE = 0.5           # normal and style sets: one extra half rep
REST_CHANCE = 0.5                 # any curl lift set: one rest in the middle, REST_SECONDS long (dataio)
PARTIAL_SETS = 2
PARTIAL_KINDS = ["half_lift", "half_lift", "half_lower", "half_lower"]
DRIFT_SETS, DRIFT_PER_SET = 2, 3
LIFT_CONDITIONS = ("normal", *CURL_STYLES, "partial", "drift")  # scripted; exhaustion has no script
# Edge-case targets agreed 2026-10-26: half reps ~12% of movements, drift reps ~7%,
# oversampled on purpose; results are also reported at realistic base rates.

HALF_HOW = {
    "curl": {"half_lift": "lift only until your forearm is level, then lower",
             "half_lower": "lower only until your forearm is level, then start the next lift"},
    "raise": {"half_lift": "lift only half way to shoulder height, then lower",
              "half_lower": "lower only half way, then start the next lift"},
    "press": {"half_lift": "press only to forehead height, then lower",
              "half_lower": "lower only to forehead height, then press again"},
}
OTHER_HOW = {
    "drift": "let your elbow swing forward as you lift, then bring it back",
    "pause": "hold 2–3 s halfway up, then finish it",
    **{k: f"after it, lower all the way and let the arm hang still for {s} s (count it out loud), then carry on"
       for k, s in zip(REST_KINDS, REST_SECONDS)},
}
STYLE_NOTES = {
    "slow": "Slow: 3 s up, 3 s down.",
    "fast": "Fast: as fast as you can while staying in control.",
    "alternating": "Alternate arms; count only the sensor arm's lifts.",
}
DISTRACTORS = {
    "walk": "Walk around for about 15 s, arms natural. Not a set: the counter should say 0.",
    "rerack": "Put the dumbbell down and pick it up again, three times. Not a set.",
    "adjust": "Shake out your arm and fiddle with the strap for about 15 s, without moving the band. Not a set.",
}
SYNC_START = ("Elbow locked, palm facing your thigh: swing the straight arm forward and back like a "
              "pendulum, 3 times, fast. Same plane as a curl, no sideways motion.")
SYNC_END = "Same pendulum swings: elbow locked, palm facing your thigh, forward and back only."
EXHAUSTION = ("Curl until you can't do one more full rep. Count full reps only and TYPE that count "
              "at the prompt (Enter would record 0).")


def place_marks(rng: np.random.Generator, movements: int, kinds: list[str]) -> dict[int, str]:
    """One distinct movement per kind out of 2..movements-1 (never the first or last).

    Redrawn until no half_lift or pause directly follows a half_lower: that lift starts from
    halfway already, so "lift only halfway" or "hold halfway up" would be no movement at all.
    Never marking the last movement also guarantees a half_lower always has a next lift.
    """
    while True:
        picks = rng.choice(np.arange(2, movements), size=len(kinds), replace=False)
        marks = {int(m): kind for m, kind in zip(picks, kinds)}
        if not any(k == "half_lower" and marks.get(m + 1) in ("half_lift", "pause") for m, k in marks.items()):
            return marks  # a rest sits on its own lift, so that lift is always lowered all the way


def lift_numbers(movements: list[int]) -> str:
    words = [str(m) for m in sorted(movements)]
    if len(words) == 1:
        return f"Lift {words[0]}"
    return f"Lifts {', '.join(words[:-1])} and {words[-1]}"


def lift_set(rng: np.random.Generator, exercise: str, condition: str) -> dict:
    reps = int(rng.integers(MIN_REPS, MAX_REPS + 1))
    if condition == "partial":
        kinds = list(PARTIAL_KINDS)
    elif condition == "drift":
        kinds = ["drift"] * DRIFT_PER_SET
    else:
        kinds = ["pause"] if condition == "pause" else []
        if rng.random() < EXTRA_HALF_CHANCE:
            # Alternating: you can't lower one arm halfway, then start the next lift with the other hand.
            options = ["half_lift"] if condition == "alternating" else HALF_KINDS
            kinds.append(str(rng.choice(options)))
    movements = reps + sum(k in HALF_KINDS for k in kinds)
    if exercise == "curl" and rng.random() < REST_CHANCE and len(kinds) < movements - 2:
        kinds.append(f"rest{rng.choice(REST_SECONDS)}")  # skipped only when no free middle lift is left
    marks = place_marks(rng, movements, kinds)

    notes = [STYLE_NOTES[condition]] if condition in STYLE_NOTES else []
    notes.append(f"Count out loud: {movement_count(reps, marks)} lifts.")
    how = {**HALF_HOW[exercise], **OTHER_HOW}
    for kind in MARK_KINDS:
        lifts = [m for m, k in marks.items() if k == kind]
        if lifts:
            notes.append(f"{lift_numbers(lifts)}: {how[kind]}.")
    notes.append("All other lifts: full and strict." if marks else "All full and strict.")
    return {"exercise": exercise, "reps": reps, "condition": condition,
            "notes": " ".join(notes), "marked": format_marked(marks)}


def make_script(exercises: list[str], seed: int) -> list[dict]:
    unknown = [ex for ex in exercises if ex not in HALF_HOW]
    if unknown:
        raise ValueError(f"unknown exercise {unknown[0]!r}: choose from {', '.join(HALF_HOW)}")
    if "curl" not in exercises:
        raise ValueError("the style, partial, drift and exhaustion sets are curls: include curl")
    rng = np.random.default_rng(seed)
    body = [lift_set(rng, ex, "normal") for ex in exercises for _ in range(NORMAL_SETS)]
    body += [lift_set(rng, "curl", style) for style in CURL_STYLES]
    body += [lift_set(rng, "curl", "partial") for _ in range(PARTIAL_SETS)]
    body += [lift_set(rng, "curl", "drift") for _ in range(DRIFT_SETS)]
    body += [{"exercise": name, "reps": 0, "condition": "distractor", "notes": note, "marked": ""}
             for name, note in DISTRACTORS.items()]
    body = [body[i] for i in rng.permutation(len(body))]
    return [
        {"exercise": "sync", "reps": 0, "condition": "gesture", "notes": SYNC_START, "marked": ""},
        *body,
        {"exercise": "curl", "reps": 0, "condition": "exhaustion", "notes": EXHAUSTION, "marked": ""},
        {"exercise": "sync", "reps": 0, "condition": "gesture", "notes": SYNC_END, "marked": ""},
    ]


def write_script(session: str, rows: list[dict], meta: dict, out_dir: Path = PROTOCOLS_DIR,
                 force: bool = False) -> Path:
    """protocols/<session>.csv plus a .json of how it was made. Refuses to overwrite unless forced."""
    path = out_dir / f"{session}.csv"
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists. Annotations point into it by row, "
                              "so regenerating it relabels every recorded set.")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix(".json").write_text(json.dumps(meta, indent=2) + "\n")
    return path


def shares(rows: list[dict]) -> tuple[int, float, float]:
    """Total movements in the scripted lift sets and the share that are half and drift reps."""
    kinds = []
    for r in rows:
        if r["condition"] in LIFT_CONDITIONS:
            marks = parse_marked(r["marked"])
            kinds += [marks.get(m, "full") for m in range(1, movement_count(int(r["reps"]), marks) + 1)]
    n = len(kinds)
    return n, sum(k in HALF_KINDS for k in kinds) / n, kinds.count("drift") / n


def main():
    parser = argparse.ArgumentParser(description="Write a shuffled, seeded set script for one session.")
    parser.add_argument("--session", required=True)
    parser.add_argument("--seed", type=int, required=True, help="use a different seed for every session")
    parser.add_argument("--exercises", nargs="+", default=EXERCISES,
                        help="curl by default; add raise/press only as an unfamiliar-exercise test")
    parser.add_argument("--force", action="store_true", help="overwrite a script that was never recorded")
    args = parser.parse_args()

    if annotation_path(args.session).exists():
        sys.exit(f"Session '{args.session}' has already been recorded; its script can't change.")
    try:
        rows = make_script(args.exercises, args.seed)
        meta = {"session": args.session, "seed": args.seed, "exercises": args.exercises,
                "numpy": np.__version__, "created": date.today().isoformat()}
        path = write_script(args.session, rows, meta, force=args.force)
    except (ValueError, FileExistsError) as e:
        sys.exit(str(e))

    n, half, drift = shares(rows)
    rests = [k for r in rows for k in parse_marked(r["marked"]).values() if k in REST_KINDS]
    print(f"wrote {path.relative_to(PROTOCOLS_DIR.parent)}: {len(rows)} rows, {n} scripted movements, "
          f"{half:.0%} half reps, {drift:.0%} drift reps, {len(rests)} sets with a rest "
          f"({', '.join(sorted(rests)) or 'none'})")


if __name__ == "__main__":
    main()
