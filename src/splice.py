"""Spliced workouts: new curl sets made from cut-up real ones, with a true count known by construction.

To find what trips the counter up, we plant suspected trip-ups (levers: a long pause, a half rep, a
distractor movement...) into sets whose true count we know, and measure which ones break the count
most often. Crash testing instead of waiting for crashes.

Reps are cut at their bottoms, where the arm hangs and stops for an instant before the next lift,
like a film cut on a still frame. Two reps from the same session then join cleanly: same strap,
same hanging direction, arm still. A rep is only used if both its ends sit within HANG_DEG of
hanging.

Rests and pauses (the glue) are real still stretches from between sets, turned onto the rep-bottom
posture: between sets people hang their arm ~20-35 deg differently from a rep's bottom (relaxed,
palm turned), and a lead-in like that moves where the learned model takes its zero. Turning a still
clip keeps its real sway and noise and only changes which way it points, like re-strapping the
board for that clip, which the counters are built to ignore. A relaxed start is kept as a lever.

Each spliced set is counted by the frozen counters as one Enter-press set, so this measures count
errors only (is the set a set? is the replay's job). Pieces come only from sets whose labels we
trust: not flagged, not count-fixed, and with as many humps as scripted lifts.

Before any lever: the null check. Full reps only, shuffled, from several sets of one session. They
should count right about as often as real clean sets do; if not, the joins themselves cause errors.

    python -m src.splice --null
"""

import argparse
import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from src.dataio import ACCEL, DATA_DIR, GYRO, load_annotations, load_session, parse_marked, slice_set
from src.detector import VERSIONS, build_dataset
from src.labels import USB_AXIS_FILE
from src.make_script import PROTOCOLS_DIR
from src.stream import UNRESOLVED, count_box, flagged_sets, model_for
from src.sync import REPORTS_DIR

FS = 50.0
HANG_DEG = 15.0      # a rep is cut only if both its bottoms are this close to hanging; real reps' bottoms
                     # are mostly 0-7 deg, a half-lower stops at ~45-90
STILL_DPS = 10.0     # glue: the arm turning slower than this, for at least MIN_CLIP_S, with gravity within
MIN_CLIP_S = 1.0     # MAX_GLUE_TURN_DEG of where it sits at the rep bottoms (~95% of between-set still time)
MAX_GLUE_TURN_DEG = 45.0
GLUE_SPREAD_DEG = 5.0  # a still stretch is split where its gravity has wandered this far, so one turn fits it
LEAD_S = (1.0, 2.0)  # hang before the first rep and after the last, like after an Enter press
NULL_CONDITIONS = ("normal", "partial", "drift", "pause")  # normal-paced sets: their unmarked reps are plain
                                                           # full reps (slow, fast, alternating are levers)
LIBRARY = ("dev1b", "dev2b", "p2_s1", "p4_s1", "p5_s1")

# Levers: suspected trip-ups planted into spliced sets. Each is on with chance P_ON, at a random dose.
P_ON = 0.3
FIRST_LEVERS = ("uneven_rest", "long_pause", "pause_reps", "speed_mix", "alternating", "half_lifts", "drift_reps",
                "distractor_bit", "edge_pickup", "relaxed_start", "dropped_samples", "clipped")
LEVERS = FIRST_LEVERS + ("half_lowers",)  # half_lowers (11-05) draws from its own random stream, so a set
                                          # without it is built exactly as before it existed
SET_LENGTH = (2, 15)      # reps per spliced set, always random
RAMP_S = 0.8              # a relaxed start eases into the curl posture over this long, with the matching
                          # turn added to the gyro (a sudden jump would move gravity with no rotation)
CHUNK_S = (3.0, 8.0)      # distractor movement cut from walk, adjust and rerack sets
SLOW_DPS = 30.0           # a chunk starts and ends where the arm turns slower than this, its gravity at both
                          # ends within HANG_DEG of each other, and is turned so it starts in the curl posture


@dataclass
class Recipe:
    session: str
    parts: list[tuple] = field(default_factory=list)  # (start, end) on the board clock, [start, end); glue
                                                      # parts add a rotation vector that turns them
    truth: int = 0
    levers: dict = field(default_factory=dict)


def cut_points(humps: pd.DataFrame, t0: float) -> pd.DataFrame:
    """Each hump's piece, bottom to bottom, on the board clock. hump_features gives the top's time from
    the set start (t0) and how long the rise before it and the fall after it took."""
    return pd.DataFrame({"t_start": t0 + humps["t_top"] - humps["rise_s"],
                         "t_end": t0 + humps["t_top"] + humps["fall_s"]}, index=humps.index)


def untrusted_sets(session: str) -> set[int]:
    """Sets whose labels we don't trust: flagged, count-fixed, or with no settled count."""
    fixes = pd.read_csv(DATA_DIR / "count_fixes.csv")
    fixed = set(fixes.loc[fixes["session"] == session, "set_idx"])
    return flagged_sets(session) | fixed | {i for s, i in UNRESOLVED if s == session}


def rep_pieces(session: str) -> pd.DataFrame:
    """One row per usable rep: where it is, its label (full or half) and scripted mark, its top."""
    raw, ann = load_session(session), load_annotations(session)
    humps, sets = build_dataset([session])
    script = pd.read_csv(PROTOCOLS_DIR / f"{session}.csv", dtype=str, keep_default_na=False)
    bad = untrusted_sets(session)
    rows = []
    for (set_idx, h) in humps.groupby("set_idx"):
        cond = sets.loc[sets["set_idx"] == set_idx, "condition"].iloc[0]
        if cond == "distractor" or set_idx in bad or h["label"].isna().any():
            continue  # labels exist only when the humps line up one-to-one with the scripted lifts
        a = ann[ann["set_idx"] == set_idx].iloc[0]
        cuts = cut_points(h, slice_set(raw, a)["t"].iloc[0])
        marks = parse_marked(script.iloc[set_idx - 1]["marked"])
        for n, (k, hump) in enumerate(h.iterrows(), start=1):
            rows.append({"session": session, "set_idx": set_idx, "condition": cond, "n": n,
                         "kind": marks.get(n, "full"), "full": bool(hump["label"]),
                         "t_start": cuts.loc[k, "t_start"], "t_end": cuts.loc[k, "t_end"],
                         "top_deg": hump["top_deg"], "start_deg": hump["start_deg"], "end_deg": hump["end_deg"],
                         "clean_ends": max(hump["start_deg"], hump["end_deg"]) <= HANG_DEG})
    return pd.DataFrame(rows)


@dataclass
class Glue:
    clips: pd.DataFrame    # t_start, t_end on the board clock; ux, uy, uz: each clip's median gravity direction
    bottom_up: np.ndarray  # gravity direction (board frame) at the session's rep bottoms


def unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v, axis=-1, keepdims=True)


def turn(u: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Rotation vector that turns unit vector u onto target, the shortest way."""
    axis = np.cross(u, target)
    s = np.linalg.norm(axis)
    return np.zeros(3) if s < 1e-9 else axis / s * np.arctan2(s, u @ target)


def toward(b: np.ndarray, u: np.ndarray, deg: float) -> np.ndarray:
    """The direction `deg` degrees from b, on the great circle from b toward u (past u if deg is larger)."""
    axis = np.cross(b, u)
    if np.linalg.norm(axis) < 1e-9:
        axis = np.cross(b, [1.0, 0.0, 0.0] if abs(b[0]) < 0.9 else [0.0, 1.0, 0.0])
    return Rotation.from_rotvec(unit(axis) * np.radians(deg)).apply(b)


def still_hang_runs(t: np.ndarray, accel: np.ndarray, gyro: np.ndarray, bottom_up: np.ndarray,
                    usable: np.ndarray) -> pd.DataFrame:
    """Still stretches of at least MIN_CLIP_S whose gravity sits within MAX_GLUE_TURN_DEG of the rep
    bottoms' (`bottom_up`), split wherever gravity wanders GLUE_SPREAD_DEG, with each one's median
    gravity direction (ux, uy, uz)."""
    up = unit(accel)
    near = np.degrees(np.arccos(np.clip(up @ bottom_up, -1, 1))) < MAX_GLUE_TURN_DEG
    ok = usable & near & (np.linalg.norm(gyro, axis=1) < STILL_DPS)
    same = np.cos(np.radians(GLUE_SPREAD_DEG))
    runs, start = [], None
    for i in range(len(t)):
        if ok[i] and start is not None and up[i] @ up[start] >= same:
            continue
        if start is not None:
            runs.append((start, i))
        start = i if ok[i] else None
    if start is not None:
        runs.append((start, len(t)))
    rows = [(t[a], t[b - 1] + 1 / FS, *unit(np.median(up[a:b], axis=0))) for a, b in runs
            if t[b - 1] + 1 / FS - t[a] >= MIN_CLIP_S]
    return pd.DataFrame(rows, columns=["t_start", "t_end", "ux", "uy", "uz"])


def hang_clips(session: str, pieces: pd.DataFrame) -> Glue:
    """Still stretches between sets (the glue for lead-ins, rests, pauses) and the gravity direction at
    the session's rep bottoms: the median where its clean reps start and end."""
    raw, ann = load_session(session), load_annotations(session)
    host, t = raw["host_t"].to_numpy(), raw["t"].to_numpy()
    accel = raw[ACCEL].to_numpy()
    ends = pieces.loc[pieces["clean_ends"], ["t_start", "t_end"]].to_numpy().ravel()
    at = np.clip(np.searchsorted(t, ends), 0, len(t) - 1)
    bottom_up = unit(np.median(unit(accel[at]), axis=0))
    usable = np.ones(len(t), bool)
    for a in ann.itertuples():
        usable &= ~((host >= a.start_host_t) & (host <= a.end_host_t))
    return Glue(still_hang_runs(t, accel, raw[GYRO].to_numpy(), bottom_up, usable), bottom_up)


def splice(raw: pd.DataFrame, parts: list[tuple]) -> pd.DataFrame:
    """Join [start, end) windows of one recording into a new one, re-timed at exactly FS, shaped like
    load_session's output so every counter runs on it unchanged.

    A part may carry a rotation vector (third entry) that turns its accelerometer and gyro, like
    re-strapping the board for that window. With a fourth entry the turn eases from the third to the
    fourth across the window (same axis), and the gyro gets the matching rotation: turning the readings
    by an angle growing at rate w about axis n is the board turning at -w about n.
    """
    t = raw["t"].to_numpy()
    values = raw[ACCEL + GYRO].to_numpy()
    chunks = []
    for part in parts:
        v = values[(t >= part[0]) & (t < part[1])]
        if len(part) == 4 and len(v):
            steps = np.linspace(0, 1, len(v))[:, None]
            rv = part[2] + steps * (part[3] - part[2])
            r = Rotation.from_rotvec(rv)
            spin = -np.degrees(part[3] - part[2]) / max(part[1] - part[0], 1 / FS)
            v = np.hstack([r.apply(v[:, :3]), r.apply(v[:, 3:]) + spin])
        elif len(part) == 3 and len(v) and np.any(part[2]):
            r = Rotation.from_rotvec(part[2])
            v = np.hstack([r.apply(v[:, :3]), r.apply(v[:, 3:])])
        chunks.append(v)
    out = pd.DataFrame(np.vstack(chunks), columns=ACCEL + GYRO)
    out.insert(0, "t", np.arange(len(out)) / FS)
    out.insert(0, "host_t", out["t"])
    return out


def take_hang(rng: np.random.Generator, glue: Glue, seconds: float, off_deg: float = 0.0,
              target: np.ndarray | None = None) -> list[tuple]:
    """`seconds` of still arm, from as few clips as it takes (longer clips more likely), each turned so
    its gravity sits off_deg from the rep bottoms': 0 = the rep-bottom posture, ~20-35 = a relaxed hang
    the way that clip actually hung. With `target`, every clip is turned onto that one direction."""
    clips = glue.clips
    length = (clips["t_end"] - clips["t_start"]).to_numpy()
    parts, left = [], seconds
    while left > 1e-9:
        k = rng.choice(len(clips), p=length / length.sum())
        span = min(left, length[k])
        a = clips["t_start"].iloc[k] + rng.uniform(0, max(length[k] - span, 0.0))
        u = clips[["ux", "uy", "uz"]].iloc[k].to_numpy()
        parts.append((a, a + span, turn(u, toward(glue.bottom_up, u, off_deg) if target is None else target)))
        left -= span
    return parts


def relaxed_hang(rng: np.random.Generator, glue: Glue, seconds: float, off_deg: float) -> list[tuple]:
    """A lead-in hanging off_deg off the curl posture, easing into it over its last RAMP_S. The lean's
    direction is the easing clip's own, and any hold before it is turned to the same lean."""
    clips = glue.clips
    long = clips[clips["t_end"] - clips["t_start"] >= RAMP_S]
    c = long.iloc[rng.integers(len(long))]
    a = rng.uniform(c["t_start"], c["t_end"] - RAMP_S)
    u = c[["ux", "uy", "uz"]].to_numpy()
    axis = np.cross(u, glue.bottom_up)
    axis = unit(axis) if np.linalg.norm(axis) > 1e-9 else unit(np.cross(u, [1.0, 0.0, 0.0]))
    beta = np.arccos(np.clip(u @ glue.bottom_up, -1, 1))  # turning u by beta about axis lands on bottom_up
    lean = Rotation.from_rotvec(axis * (beta - np.radians(off_deg))).apply(u)
    hold = take_hang(rng, glue, seconds - RAMP_S, target=lean) if seconds > RAMP_S else []
    return hold + [(a, a + RAMP_S, axis * (beta - np.radians(off_deg)), axis * beta)]


def null_recipe(rng: np.random.Generator, session: str, pieces: pd.DataFrame, glue: Glue) -> Recipe:
    """5-12 plain full reps in random order from the session's normal-paced sets, between two hangs."""
    pool = pieces[pieces["full"] & (pieces["kind"] == "full") & pieces["clean_ends"]
                  & pieces["condition"].isin(NULL_CONDITIONS)]
    reps = pool.iloc[rng.choice(len(pool), size=min(len(pool), int(rng.integers(5, 13))), replace=False)]
    parts = take_hang(rng, glue, rng.uniform(*LEAD_S))
    parts += list(zip(reps["t_start"], reps["t_end"]))
    parts += take_hang(rng, glue, rng.uniform(*LEAD_S))
    return Recipe(session, parts, truth=len(reps), levers={"sets_used": reps["set_idx"].nunique()})


@dataclass
class Library:
    """Everything one session lends to spliced sets."""
    session: str
    raw: pd.DataFrame
    pieces: pd.DataFrame
    glue: Glue
    chunks: dict  # "distractor": walk/adjust/rerack chunks, "edge": rerack chunks; each (start, end, rotvec)
    chunk_from: dict  # the same keys: which distractor (walk, adjust, rerack) each chunk was cut from
    model: object = None
    version: str = "v1"  # detector version inside the counters


def movement_chunks(raw: pd.DataFrame, ann: pd.DataFrame, exercises, bottom_up, rng,
                    per_window=40) -> tuple[list[tuple], list[str]]:
    """CHUNK_S-long pieces of distractor movement that start and end slow, at the same posture, turned so
    they start in the curl posture: a join into a set is then as clean as a join between reps. Also
    which distractor each came from."""
    t, host = raw["t"].to_numpy(), raw["host_t"].to_numpy()
    up = unit(raw[ACCEL].to_numpy())
    slow = np.linalg.norm(raw[GYRO].to_numpy(), axis=1) < SLOW_DPS
    same = np.cos(np.radians(HANG_DEG))
    out, source = [], []
    for a in ann[(ann["condition"] == "distractor") & ann["exercise"].isin(exercises) & (ann["status"] == "ok")].itertuples():
        idx = np.flatnonzero(slow & (host >= a.start_host_t) & (host <= a.end_host_t))
        for _ in range(per_window):
            if len(idx) < 2:
                break
            i = idx[rng.integers(len(idx))]
            ends = idx[(t[idx] >= t[i] + CHUNK_S[0]) & (t[idx] <= t[i] + CHUNK_S[1]) & (up[idx] @ up[i] >= same)]
            if len(ends):
                j = ends[rng.integers(len(ends))]
                out.append((t[i], t[j], turn(up[i], bottom_up)))
                source.append(a.exercise)
    return out, source


def load_library(session: str, seed: int = 0, version: str = "v1") -> Library:
    raw, ann = load_session(session), load_annotations(session)
    pieces = rep_pieces(session)
    glue = hang_clips(session, pieces)
    rng = np.random.default_rng(seed)
    found = {"distractor": movement_chunks(raw, ann, ("walk", "adjust", "rerack"), glue.bottom_up, rng),
              "edge": movement_chunks(raw, ann, ("rerack",), glue.bottom_up, rng)}
    return Library(session, raw, pieces, glue, {k: v[0] for k, v in found.items()}, {k: v[1] for k, v in found.items()},
                   model_for(session, version), version)


def half_lower_pairs(pieces: pd.DataFrame) -> pd.DataFrame:
    """A scripted half-lower (or a run of them) and the full lift after it, as one piece: one rep, the arm
    stopping partway down in between. Used only if it starts and ends within HANG_DEG of hanging and
    every half-lower in it really stopped above that."""
    rows = []
    for _, s in pieces.groupby("set_idx"):  # one session's pieces
        s = s.sort_values("n").reset_index(drop=True)
        k = 0
        while k < len(s):
            if s.loc[k, "kind"] != "half_lower":
                k += 1
                continue
            j = k
            while j < len(s) and s.loc[j, "kind"] == "half_lower":
                j += 1
            if (j < len(s) and s.loc[j, "full"] and s.loc[k, "start_deg"] <= HANG_DEG
                    and s.loc[j, "end_deg"] <= HANG_DEG and (s.loc[k:j - 1, "end_deg"] > HANG_DEG).all()):
                rows.append({"set_idx": s.loc[k, "set_idx"], "n": s.loc[k, "n"], "humps": j - k + 1,
                             "t_start": s.loc[k, "t_start"], "t_end": s.loc[j, "t_end"]})
            k = j + 1
    return pd.DataFrame(rows, columns=["set_idx", "n", "humps", "t_start", "t_end"])


def pool(pieces: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Usable reps of one kind: base (plain full reps from normal-paced sets), fast, slow, alternating,
    pause, drift (full reps), half_lift (not a full rep), half_lower (a half-lower and the lift after
    it: one full rep)."""
    if kind == "half_lower":
        return half_lower_pairs(pieces)
    p = pieces[pieces["clean_ends"]]
    if kind == "half_lift":
        return p[p["kind"] == "half_lift"]
    p = p[p["full"]]
    if kind in ("pause", "drift"):
        return p[p["kind"] == kind]
    p = p[p["kind"] == "full"]
    if kind == "base":
        return p[p["condition"].isin(NULL_CONDITIONS)]
    return p[p["condition"] == kind]  # fast, slow, alternating


def plan_levers(rng: np.random.Generator) -> dict:
    """Which levers are on and at what dose (None = off)."""
    on = {k: rng.random() < P_ON for k in FIRST_LEVERS}
    dose = {
        "uneven_rest": rng.uniform(0.5, 3.0),          # the longest extra rest between reps, s
        "long_pause": rng.uniform(2.0, 15.0),          # one extra rest, s
        "pause_reps": int(rng.integers(1, 4)),         # reps held halfway up
        "speed_mix": rng.uniform(0.25, 1.0),           # share of reps swapped for fast or slow ones
        "alternating": rng.uniform(0.25, 1.0),         # share of reps swapped for alternating-set reps
        "half_lifts": int(rng.integers(1, 4)),         # lifts only to level, inserted
        "drift_reps": int(rng.integers(1, 4)),         # elbow-swing reps (full reps)
        "distractor_bit": 1,                           # one chunk of walk / adjust / rerack inside the set (its
                                                       # length, 3-8 s, is recorded as the dose)
        "edge_pickup": rng.uniform(0.0, 15.0),         # rerack movement this many s before the first rep or after the last
        "relaxed_start": rng.uniform(10.0, 40.0),      # lead-in this many deg off the curl posture
        "dropped_samples": rng.uniform(0.01, 0.05),    # share of samples lost, in bursts of 2-10
        "clipped": rng.uniform(150.0, 400.0),          # gyro saturates at this many deg/s
    }
    plan = {k: (dose[k] if on[k] else None) for k in FIRST_LEVERS}
    plan["set_length"] = int(rng.integers(SET_LENGTH[0], SET_LENGTH[1] + 1))
    plan["speed"] = "fast" if rng.random() < 0.5 else "slow"
    plan["edge_side"] = "before" if rng.random() < 0.5 else "after"
    side = rng.spawn(1)[0]  # a child stream: drawing from it leaves rng where it was
    plan["half_lowers"] = int(side.integers(1, 4)) if side.random() < P_ON else None  # reps swapped for a
    plan["side_rng"] = side                                                          # half-lower and its next lift
    return plan


def draw(rng, p: pd.DataFrame, n: int) -> list[tuple]:
    rows = p.iloc[rng.choice(len(p), size=n, replace=n > len(p))]
    return list(zip(rows["t_start"], rows["t_end"]))


def build_lever_set(rng: np.random.Generator, lib: Library, plan: dict) -> Recipe:
    """Turn a plan into parts. A lever whose material this session lacks is recorded as off."""
    applied = {k: None for k in LEVERS}
    distractor_from = None
    n = plan["set_length"]
    reps = draw(rng, pool(lib.pieces, "base"), n)
    free = list(rng.permutation(n))  # rep positions not yet swapped for another kind

    def swap(kind, count, lever):
        p = pool(lib.pieces, kind)
        count = min(count, len(free))
        if plan[lever] is None or len(p) == 0 or count == 0:
            return
        for pos, piece in zip([free.pop() for _ in range(count)], draw(rng, p, count)):
            reps[pos] = piece
        applied[lever] = plan[lever]

    swap(plan["speed"], max(1, round((plan["speed_mix"] or 0) * n)), "speed_mix")
    swap("alternating", max(1, round((plan["alternating"] or 0) * n)), "alternating")
    swap("pause", plan["pause_reps"] or 0, "pause_reps")
    swap("drift", plan["drift_reps"] or 0, "drift_reps")
    count = min(plan.get("half_lowers") or 0, len(free))
    pairs = pool(lib.pieces, "half_lower") if count else None
    if count and len(pairs):  # last, so the swaps above take the same positions as before it existed
        for pos, piece in zip([free.pop() for _ in range(count)], draw(plan["side_rng"], pairs, count)):
            reps[pos] = piece
        applied["half_lowers"] = plan["half_lowers"]
    seq = [("rep", [r]) for r in reps]  # (kind, parts); only "rep" counts toward the truth

    def insert_inside(item):  # between two reps, never before the first or after the last
        reps_at = [k for k, s in enumerate(seq) if s[0] == "rep"]
        seq.insert(int(rng.integers(reps_at[0] + 1, reps_at[-1] + 1)), item)

    if plan["half_lifts"] is not None and len(pool(lib.pieces, "half_lift")):
        for piece in draw(rng, pool(lib.pieces, "half_lift"), plan["half_lifts"]):
            insert_inside(("half", [piece]))
        applied["half_lifts"] = plan["half_lifts"]
    if plan["distractor_bit"] is not None and lib.chunks["distractor"]:
        k = rng.integers(len(lib.chunks["distractor"]))
        chunk = lib.chunks["distractor"][k]
        insert_inside(("distractor", [chunk]))
        applied["distractor_bit"] = chunk[1] - chunk[0]  # its length, s
        distractor_from = lib.chunk_from["distractor"][k]
    if plan["long_pause"] is not None:
        insert_inside(("glue", take_hang(rng, lib.glue, plan["long_pause"])))
        applied["long_pause"] = plan["long_pause"]
    if plan["uneven_rest"] is not None:
        for k in range(len(seq) - 1, 0, -1):
            extra = rng.uniform(0, plan["uneven_rest"])
            if extra > 0.1:
                seq.insert(k, ("glue", take_hang(rng, lib.glue, extra)))
        applied["uneven_rest"] = plan["uneven_rest"]

    before, after = [], []
    edge = plan["edge_pickup"] is not None and lib.chunks["edge"]
    relaxed = plan["relaxed_start"]
    if edge:
        chunk = lib.chunks["edge"][rng.integers(len(lib.chunks["edge"]))]
        gap = max(plan["edge_pickup"], RAMP_S + 0.2 if relaxed is not None and plan["edge_side"] == "before" else 0.1)
        if plan["edge_side"] == "before":
            before = take_hang(rng, lib.glue, rng.uniform(*LEAD_S)) + [chunk]
            lead_in_s = gap
        else:
            after = take_hang(rng, lib.glue, gap) + [chunk]
            lead_in_s = rng.uniform(*LEAD_S)
        applied["edge_pickup"] = plan["edge_pickup"]
    else:
        lead_in_s = rng.uniform(*LEAD_S)
    if relaxed is not None:
        lead_in = relaxed_hang(rng, lib.glue, max(lead_in_s, RAMP_S + 0.2), relaxed)
        applied["relaxed_start"] = relaxed
    else:
        lead_in = take_hang(rng, lib.glue, lead_in_s)
    parts = before + lead_in + [p for _, ps in seq for p in ps] + after + take_hang(rng, lib.glue, rng.uniform(*LEAD_S))

    applied["dropped_samples"], applied["clipped"] = plan["dropped_samples"], plan["clipped"]
    levers = {**applied, "set_length": n, "speed": plan["speed"] if applied["speed_mix"] else None,
              "edge_side": plan["edge_side"] if applied["edge_pickup"] is not None else None,
              "distractor_from": distractor_from,
              "degrade_seed": int(rng.integers(2**31))}
    return Recipe(lib.session, parts, truth=n, levers=levers)


def degrade(seg: pd.DataFrame, share_dropped, clip_dps, seed: int) -> pd.DataFrame:
    """Signal levers on a finished set: lose share_dropped of the samples in bursts of 2-10 (their times
    stay missing, like real lost samples), and saturate the gyro at clip_dps."""
    rng = np.random.default_rng(seed)
    out = seg.copy()
    if clip_dps is not None:
        out[GYRO] = out[GYRO].clip(-clip_dps, clip_dps)
    if share_dropped is not None:
        keep = np.ones(len(out), bool)
        while (~keep).mean() < share_dropped:
            a = int(rng.integers(len(out)))
            keep[a:a + int(rng.integers(2, 11))] = False
        out = out[keep].reset_index(drop=True)
    return out


def count_recipe(recipe: Recipe, raw: pd.DataFrame, model, usb_axis, version: str = "v1") -> dict:
    seg = splice(raw, recipe.parts)
    lv = recipe.levers
    if "degrade_seed" in lv:
        seg = degrade(seg, lv["dropped_samples"], lv["clipped"], lv["degrade_seed"])
    return {"session": recipe.session, "truth": recipe.truth, **lv,
            **{k: v for k, v in count_box(seg, model, usb_axis, version).items() if not k.startswith("rep_times")}}


def screen(n_sets: int, seed: int, usb_axis, version: str = "v1") -> pd.DataFrame:
    """n_sets spliced sets, each from a random session with random levers. Set i is built from
    default_rng([seed, i]), so any one can be rebuilt later (e.g. for the count-error model's features)."""
    libs = {s: load_library(s, seed, version) for s in LIBRARY}
    rows = []
    for i in range(n_sets):
        rng = np.random.default_rng([seed, i])
        lib = libs[LIBRARY[rng.integers(len(LIBRARY))]]
        rows.append({"set": i, **count_recipe(build_lever_set(rng, lib, plan_levers(rng)), lib.raw, lib.model, usb_axis,
                                              version)})
    return pd.DataFrame(rows)


def lever_effects(df: pd.DataFrame, err: pd.Series, rng: np.random.Generator, n_boot=1000, n_shuffle=200) -> pd.DataFrame:
    """Each lever's effect: error rate with it on minus with it off, a bootstrap 95% interval, and the
    95th percentile of the same difference with the on/off labels shuffled (Aditya's check)."""
    e = err.to_numpy(float)
    rows = []
    for lever in LEVERS:
        on = df[lever].notna().to_numpy()
        diff = lambda o, x=e: x[o].mean() - x[~o].mean()
        boot = [diff(on[b], e[b]) for b in (rng.integers(len(e), size=len(e)) for _ in range(n_boot))]
        shuffled = [diff(rng.permutation(on)) for _ in range(n_shuffle)]
        rows.append({"lever": lever, "sets on": int(on.sum()), "error on": e[on].mean(), "error off": e[~on].mean(),
                     "effect": diff(on), "95% low": np.percentile(boot, 2.5), "95% high": np.percentile(boot, 97.5),
                     "shuffled 95th": np.percentile(shuffled, 95)})
    out = pd.DataFrame(rows).sort_values("effect", ascending=False)
    out["beats shuffle"] = out["effect"] > out["shuffled 95th"]
    return out


DOSE_PANELS = [  # lever, title, x axis, bins for a continuous dose (None: one point per count)
    ("long_pause", "Long rest at the bottom", "rest, s", [2, 5, 8, 10, 12, 13.5, 15]),
    ("distractor_bit", "Walk / adjust / rerack inside the set", "length of the movement, s", [3, 4, 5, 6, 7, 8]),
    ("half_lifts", "Half lifts", "half lifts inserted", None),
    ("half_lowers", "Half-lowers", "reps swapped for a half-lower + next lift", None),
    ("pause_reps", "Reps held halfway up", "reps held", None),
    ("edge_pickup", "Rerack near the first or last rep", "gap to the rep, s", [0, 3, 6, 9, 12, 15]),
    ("uneven_rest", "Uneven rest between reps", "longest extra rest, s", [0.5, 1, 1.5, 2, 2.5, 3]),
]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% interval for a share k/n that stays inside 0-1 even near the edges."""
    if n == 0:
        return np.nan, np.nan
    p, d = k / n, 1 + z * z / n
    mid, half = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return mid - half, mid + half


def plot_doses(df: pd.DataFrame, path) -> None:
    """Error rate against dose for each lever that beat the shuffled check, shipped counter and combined."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter

    ink, muted, grid = "#1f2328", "#6e7781", "#d0d7de"
    series = [("min_reps", "min-reps-v1 (shipped)", "#2a78d6"), ("combined", "combined-v1", "#eb6834")]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharey=True)
    for ax in axes.ravel()[len(DOSE_PANELS):]:
        ax.axis("off")
    for ax, (lever, title, xlabel, bins) in zip(axes.ravel(), DOSE_PANELS):
        on = df[df[lever].notna()]
        key = on[lever].round().astype(int) if bins is None else pd.cut(on[lever], bins)
        for (col, label, color), shift in zip(series, (-0.06, 0.06)):
            pts = []
            for level, g in on.groupby(key, observed=True):
                x = level if bins is None else level.mid
                k, n = int((g[col] != g["truth"]).sum()), len(g)
                pts.append((x, k / n, *wilson(k, n)))
            x, y, lo, hi = map(np.array, zip(*pts))
            width = (x.max() - x.min()) if len(x) > 1 else 1.0
            xs = x + shift * width / 4
            ax.vlines(xs, lo, hi, color=color, lw=1.2, alpha=0.6)
            ax.plot(xs, y, "-o", color=color, lw=2, ms=6, label=label, markeredgecolor="white", markeredgewidth=1.2)
        off = df[df[lever].isna()]
        base = (off["min_reps"] != off["truth"]).mean()
        ax.axhline(base, color=muted, lw=1.2, ls="--", label="min-reps-v1, lever off")
        if lever in ("long_pause", "edge_pickup"):
            ax.axvline(12, color=muted, lw=0.8, ls=":")
            ax.text(12.2, 0.93, "12 s: min-reps gap", fontsize=8, color=muted, va="top")
        if bins is None:
            ax.set_xticks(sorted(key.unique()))
        ax.set_title(f"{title}  (n = {len(on)})", loc="left", fontsize=10.5, color=ink)
        ax.set_xlabel(xlabel, fontsize=9, color=muted)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
        ax.set_ylim(0, 1)
        ax.grid(axis="y", color=grid, lw=0.6)
        ax.tick_params(colors=muted, labelsize=8.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("spliced sets counted wrong", fontsize=9, color=muted)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9)
    fig.suptitle(f"What breaks the count: {len(df)} spliced sets, each lever on in ~30% at a random dose "
                 f"(bars: 95% intervals)", x=0.01, ha="left", fontsize=11, color=ink)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Make spliced curl sets from cut-up real ones and count them.")
    parser.add_argument("--null", action="store_true", help="the null check: shuffled plain full reps only")
    parser.add_argument("--screen", type=int, metavar="N", help="N spliced sets with random levers")
    parser.add_argument("--plot", action="store_true", help="with --screen: dose curves for the levers")
    parser.add_argument("--detector", choices=list(VERSIONS), default="v1", help="learned model inside the counters")
    parser.add_argument("--per-session", type=int, default=40)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    usb_axis = json.loads(USB_AXIS_FILE.read_text())["axis"]
    tag = "" if args.detector == "v1" else f"_{args.detector}"
    counters = ["rule", "learned", "combined", "min_reps"]
    if args.screen:
        df = screen(args.screen, args.seed, usb_axis, args.detector)
        out = REPORTS_DIR / f"splice_screen_{args.seed}{tag}.csv"
        df.to_csv(out, index=False)
        print(f"{len(df)} spliced sets saved to {out.relative_to(REPORTS_DIR.parent)}")
        if args.plot:
            plot = REPORTS_DIR / f"splice_doses_{args.seed}{tag}.png"
            plot_doses(df, plot)
            print(f"plot: {plot.relative_to(REPORTS_DIR.parent)}")
        print("exact: " + ", ".join(f"{c} {(df[c] == df['truth']).mean():.1%}" for c in counters))
        rng = np.random.default_rng(args.seed)
        for c in ("min_reps", "learned"):
            print(f"\nlever effects on {c} errors (error rate on - off):")
            print(lever_effects(df, df[c] != df["truth"], rng).round(3).to_string(index=False))
        return
    rng = np.random.default_rng(args.seed)
    rows = []
    for session in LIBRARY:
        raw, pieces, model = load_session(session), rep_pieces(session), model_for(session, args.detector)
        glue = hang_clips(session, pieces)
        for _ in range(args.per_session):
            rows.append(count_recipe(null_recipe(rng, session, pieces, glue), raw, model, usb_axis, args.detector))
    df = pd.DataFrame(rows)
    hit = df[counters].eq(df["truth"], axis=0)
    print("null check: shuffled plain full reps, share of spliced sets counted exactly right")
    print(hit.groupby(df["session"]).mean().round(2).to_string())
    print(f"all {len(df)} sets: " + ", ".join(f"{c} {hit[c].mean():.0%}" for c in counters))
    wrong = df[df["min_reps"] != df["truth"]]
    if len(wrong):
        print(wrong.to_string(index=False))


if __name__ == "__main__":
    main()
