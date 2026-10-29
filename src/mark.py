"""Hand-mark bad reps that someone judged by eye, especially in the to-exhaustion set.

The script says which movements were *meant* to be bad; this records which ones a person *saw*
as bad. Marks go in data/manual/<session>.csv, append-only: a correction is a new row, so every
judgment stays on record. The current mark for a movement is its latest row in file order (not
by timestamp, so a clock change can't reorder corrections), and `ok` clears it.

Movement numbers count every lift in the set from 1, as in the script. There is no upper bound,
because the exhaustion set's movement count isn't known in advance.

    python -m src.mark dev1 14 11 12 --kind half --note "grinding, stopped short"
    python -m src.mark dev1 14 12 --kind ok          # correction: 12 was fine
    python -m src.mark dev1 --show
"""

import argparse
import csv
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.dataio import DATA_DIR, annotation_path, load_annotations

HAND_KINDS = ("half", "drift", "other", "ok")
MANUAL_COLUMNS = ["session", "set_idx", "movement", "kind", "note", "by", "marked_at"]


def manual_path(session: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "manual" / f"{session}.csv"


def add_marks(session: str, set_idx: int, movements: list[int], kind: str, note: str = "",
              by: str = "aditya", data_dir: Path = DATA_DIR) -> list[dict]:
    """Validate, then append one row per movement. Nothing is written if any check fails."""
    if kind not in HAND_KINDS:
        raise ValueError(f"kind {kind!r} isn't one of {', '.join(HAND_KINDS)}")
    if not movements:
        raise ValueError("give at least one movement number")
    bad = [m for m in movements if not isinstance(m, int) or m < 1]
    if bad:
        raise ValueError(f"movement {bad[0]!r} isn't a whole number ≥ 1 (movements count every lift from 1)")
    if not annotation_path(session, data_dir).exists():
        raise ValueError(f"session '{session}' has no annotations: record it first")
    ann = load_annotations(session, data_dir)
    ok_sets = sorted(int(i) for i in ann.loc[ann["status"] == "ok", "set_idx"].unique())
    if set_idx not in ok_sets:
        raise ValueError(f"set {set_idx} isn't a recorded ok set in {session} (ok sets: "
                         f"{', '.join(map(str, ok_sets)) or 'none'})")

    marked_at = datetime.now().astimezone().isoformat(timespec="seconds")
    rows = [{"session": session, "set_idx": set_idx, "movement": m, "kind": kind,
             "note": note, "by": by, "marked_at": marked_at} for m in sorted(set(movements))]
    path = manual_path(session, data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANUAL_COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)
    return rows


def current_marks(session: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """The mark in force per (set_idx, movement): latest row wins, `ok` rows removed."""
    path = manual_path(session, data_dir)
    if not path.exists():
        return pd.DataFrame(columns=MANUAL_COLUMNS)
    log = pd.read_csv(path, dtype={"note": str, "by": str}, keep_default_na=False)
    latest = log.drop_duplicates(["set_idx", "movement"], keep="last")
    return latest[latest["kind"] != "ok"].sort_values(["set_idx", "movement"]).reset_index(drop=True)


def show(session: str, data_dir: Path = DATA_DIR) -> str:
    marks = current_marks(session, data_dir)
    if marks.empty:
        return f"{session}: no hand marks"
    names = {}
    if annotation_path(session, data_dir).exists():
        ann = load_annotations(session, data_dir)
        ok = ann[ann["status"] == "ok"]
        names = {int(r["set_idx"]): f"{r['exercise']} {r['condition']}" for _, r in ok.iterrows()}
    lines = [f"{session}: hand marks in force"]
    for set_idx, group in marks.groupby("set_idx"):
        items = [f"{m}:{k}" + (f" ({n})" if n else "") for m, k, n in
                 zip(group["movement"], group["kind"], group["note"])]
        lines.append(f"  set {set_idx} {names.get(set_idx, '')}: " + ", ".join(items))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Hand-mark bad reps judged by eye (append-only).")
    parser.add_argument("session")
    parser.add_argument("set_idx", nargs="?", type=int, help="set number, as in the annotations")
    parser.add_argument("movements", nargs="*", type=int, help="movement numbers, counting every lift from 1")
    parser.add_argument("--kind", choices=HAND_KINDS, help="ok clears an earlier mark")
    parser.add_argument("--note", default="")
    parser.add_argument("--by", default="aditya", help="who judged it")
    parser.add_argument("--show", action="store_true", help="print the marks in force per set")
    args = parser.parse_args()

    if args.show:
        print(show(args.session))
        return
    if args.set_idx is None or not args.movements or not args.kind:
        parser.error("give a set, at least one movement and --kind (or use --show)")
    try:
        rows = add_marks(args.session, args.set_idx, args.movements, args.kind, args.note, args.by)
    except ValueError as e:
        sys.exit(f"Not marked: {e}")
    print(f"set {args.set_idx}: movement{'s' if len(rows) > 1 else ''} "
          f"{', '.join(str(r['movement']) for r in rows)} marked {args.kind} by {args.by}")


if __name__ == "__main__":
    main()
