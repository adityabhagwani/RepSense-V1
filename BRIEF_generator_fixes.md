# Brief: fix the set-script generator and add hand-marking

Repo: `~/Documents/repsense` (Python 3.12, `.venv`, run tests with `.venv/bin/python -m pytest -q`).
Context: `PLAN.md` (plan, decision log), `FEATURES.md` (error-model features), `src/dataio.py` (shared formats).

## How to work with Aditya
- **Explain before code:** before writing any file, say in a few plain sentences what it will do and why, then write it.
- **Keep replies short:** a 1-line TL;DR and at most ~5 bullets, unless he asks for detail.
- Raw data and recorded scripts are never edited. Nothing in `data/` changes.

## Decisions already agreed (don't reopen)
- **Curls only by default.** The upper-arm phone (the "teacher") can only check curls. Raises and presses remain available through `--exercises`, as an optional "unfamiliar exercise" test only.
- **Conditions in every session (curl):** normal ×1, slow ×1, fast ×1, pause ×1, alternating ×1, partial ×2, drift ×2. The band is never rotated mid-session; mounting varies between sessions instead.
- **Edge-case targets:** half reps ≈ 12% of movements, drift reps ≈ 7%. They're oversampled on purpose, and results are also reported at realistic base rates.
  - partial sets: 4 half reps each, **2 "lift halfway" and 2 "lower halfway"**
  - drift sets: 3 drift reps each
  - normal, slow, fast, pause and alternating sets: each has a 50% chance of 1 extra half rep (lift or lower, chosen at random)
  - pause sets: 1 movement with a 2–3 s hold halfway up
  - marks never fall on the first or last movement
- **Distractors (reps = 0):** walk, rerack, **adjust** (shake out your arm and fiddle with the strap for about 15 s). Remove "drink": sipping is mostly still and would FAIL the validator's movement check (≥30% of the set over 20 °/s).
- Sync gesture first and last (pendulum instructions as now). The to-exhaustion curl set comes last before the final sync and is never shuffled.
- Random reps 5–12, shuffled body, CSV + `.json` metadata, never overwrite, refuse once recorded: **keep as is.**

## Fix 1: `src/make_script.py` (+ `src/dataio.py`, `tests/test_make_script.py`)
1. Default `EXERCISES = ["curl"]` and `NORMAL_SETS = 1`.
2. Add the `slow`, `fast`, `pause` and `alternating` curl sets (one each), with clear notes:
   - slow: "3 s up, 3 s down"
   - fast: "as fast as you can while staying in control"
   - pause: "Lift N: hold 2–3 s halfway up, then finish it"
   - alternating: "alternate arms; count only the sensor arm's lifts"
3. Mark kinds become `half_lift`, `half_lower`, `drift` and `pause` (update `MARK_KINDS`). `movement_count` counts both half kinds; a pause is a full rep. Give lower-halfway a clear instruction, e.g. curl: "lower only until your forearm is level, then start the next lift."
4. `shares()` covers every lift condition. Print the half and drift shares as now.
5. Tests:
   - session shape with the new counts;
   - extra-half rate ≈ 50% across all eligible sets;
   - notes name exactly the marked lifts (pause included);
   - each partial set has 2 `half_lift` and 2 `half_lower`;
   - over 200 seeds with the defaults: mean half share 10–14%, mean drift share 5–9%.

## Fix 2: rename the current `src/mark.py` → `src/check_marks.py`
It lines up the script's marks with the teacher's measured reps, which is useful, but it isn't hand-marking. Rename the module, its test (`tests/test_check_marks.py`) and its output folder (`data/check_marks/`). Update it for the new mark kinds. Behaviour otherwise unchanged.

## Fix 3: new `src/mark.py`, hand-marking bad reps
For sets where Aditya or an observer judges reps by eye, especially the to-exhaustion set.
- `python -m src.mark <session> <set_idx> <movement> [<movement> ...] --kind {half,drift,other,ok} [--note "..."] [--by observer]`
- `python -m src.mark <session> --show` prints the current marks per set.
- Storage: `data/manual/<session>.csv`, **append-only**. Columns: `session, set_idx, movement, kind, note, by, marked_at` (ISO time). A correction is a new row; the **latest row** per (set_idx, movement) wins; `ok` clears a mark.
- Validation:
  - the session's annotations must exist;
  - `set_idx` must be an `ok` set in them;
  - movements are integers ≥ 1;
  - kind must be one of the list.
  
  Raise clear errors; the command line prints them without a traceback.
- Works on any set, including the exhaustion set.
- Out of scope: merging hand marks into `check_marks` (a later decision).
- Tests:
  - append-only, and the latest mark wins;
  - `ok` clears;
  - an unknown or not-ok set is refused;
  - a bad movement number is refused;
  - `--show` reflects corrections.

## Fix 4 (small): stale receiver test
`tests/test_record.py::test_session_end_to_end` fails with `StopIteration` because the 3 s short-set guard in `src/record.py` adds a "Redo this set?" prompt the test's scripted answers don't expect (the fake board's clock makes 0.15 s look like ~2.8 s).
- Pass `min_seconds=0` in that test's `run_session` call.
- Add `test_short_set_offers_redo`: with a high `min_seconds`, Enter at "Redo?" discards and repeats the set, and `n` keeps it.

## Done when
- The full test suite passes (currently 31 pass, 1 stale failure).
- `python -m src.make_script --session demo --seed 1` (in a temp folder, or delete it after) prints half ≈ 12% and drift ≈ 7%.
- `PLAN.md` decision log gets rows for: curls-only default, the condition list, half_lift/half_lower, hand-marking design. Its Phase 2 checklist marks the generator and hand-marking done.
