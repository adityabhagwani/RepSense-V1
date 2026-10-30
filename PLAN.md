# RepSense: 36-hour plan

**Question:** can a wrist wearable tell when its own rep count is likely wrong?
**Budget:** 33 planned hours + 3 buffer. Ship gate at H25: one held-out result + 60 s video.

## Phases

| # | Phase | Hours | Ends at | Done when |
|---|---|---|---|---|
| 1 | Setup + logger | 4 | H4 | 60 s recording at 50 ±1 Hz, zero dropped samples |
| 2 | Protocol + dev data | 4 | H8 | ~20 dev sets recorded with true counts |
| 3 | Baseline counter | 5 | H13 | Dev metrics printed; counter frozen (`git tag counter-v1`) |
| 4 | Held-out data | 6 | H19 | 4 sessions, ~90 sets |
| 5 | Uncertainty + evaluation | 6 | H25 | Results table + risk-coverage plot from held-out predictions |
| ✉ | Ship gate | | H25 | Could send the email today |
| 6 | Demo | 4 | H29 | 60 s screen recording |
| 7 | README + outreach | 4 | H33 | README, results.md, email draft |
| | Buffer | 3 → 1 | H36 | 2 h moved to Phase 2 for the phone reference IMU |

## Phase 1 checklist
- [x] Firmware: BMI270 at 50 Hz, `seq` + `t_us`, one write per line (compiles: 11% flash)
- [x] Receiver: two threads, raw + annotations files, resume, 4 tests passing
- [x] Flash the firmware (bench check: 50.00 Hz, 0 seq gaps, 0 late samples, |a| = 1002 mg still)
- [x] Validation session: still, shake, fast curls (`validation` mislabeled, kept as a known-bad case; `validation2` clean)
- [x] Validator: timing, sensor and label checks; FAILs `validation`, PASSes `validation2` (1 expected WARN)
- [ ] Fault-injection tests for the validator
- [x] Update the stale receiver end-to-end test for the 3 s guard (`min_seconds=0`; new `test_short_set_offers_redo`)
- [ ] Wrist strap

## Phase 2 checklist (re-planned 10-26: 4 h → 6 h, paid from buffer)
- [x] Phone test recording: phyphox on iPhone 14 Pro, 99.63 Hz, 0 gaps, 0.08 deg/s gyro offset
- [ ] Rep + form-error definitions approved
- [x] Script generator (`src/make_script.py`): curls by default, 7 lift conditions, scripted form errors, random reps, shuffled, seeded; 9 tests. Default session: 15 rows, half reps 12.1% / drift 6.9% (mean of 1000 seeds). `src/check_marks.py` lines script marks up with teacher reps (7 tests).
- [x] Phone loader (`src/phone.py`, written by Aditya): real export loads, 0.994 g at rest
- [x] Clock sync (`src/sync.py`): 6 tests pass; on synthetic data it recovers planted offsets to 0.06 ms and drift 150.7 vs 150 ppm
- [x] Real two-device sync test: `synctest4` offset +30.5 / +30.4 ms at the two gestures, match 0.974 / 0.977, drift −1.7 ppm. `synctest2` correctly refused (match 0.25 / 0.33, mislabeled windows).
- [x] `src.sync` command line: clean message instead of a traceback when no gesture is usable
- [x] Fixed: `plot_gestures` "wall clocks only" panel shifted the phone the wrong way (+offset instead of −offset); caught by re-deriving the plot from raw phone timestamps
- [x] Elbow-angle label maker (`src/labels.py`): calibrates from the sync gestures; 4 tests pass on a simulated arm (range within 2°, drift within 0.6°, mounting-invariant, strap slip flagged)
- [x] Ran on `synctest4`: 8/8 reps; flagged gesture planarity 0.74 and curl axis 35° off the gesture axis. Diagnostic: the off-axis projection shrank ranges ~19% (median 91° vs 108° on the curl's own axis). Fix: gesture instructions now say pendulum swing in the curl plane.
- [x] `FEATURES.md`: features, ★ core set, nested random-subset selection protocol, fixed before results
- [x] Hand-marking (`src/mark.py`): append-only `data/manual/<session>.csv`, latest row wins, `ok` clears, `--show`; 14 tests
- [ ] `data/sessions.csv` metadata
- [ ] dev1 recorded + validated
- [ ] dev2 recorded + validated

## Hardware facts
- Logged Nano axes are the Arduino library's, not the board's printed ones: for this board the library reports x = −(chip y) and y = −(chip x) (`Arduino_BMI270_BMM150/src/BMI270.cpp`, `TARGET_ARDUINO_NANO33BLE`). The USB runs along the printed +y, which is **−x in our data**; the down test measured exactly that (hand held 10.9° off vertical, snapped). `data/usb_axis.json` = (−1, 0, 0).

- Mounting per session (hanging "up" vs USB axis −x): `dev1` and `dev1b` 35–46°, board turned ~45° in its own plane (USB diagonal across the forearm), same position both sessions; `dev2b` 20–30°, mounted by the rule. Counter and teacher measure from the hanging direction, so they're unaffected; student features that assume USB-along-forearm (#24) aren't, which makes `dev1`/`dev1b` a mounting-robustness test. Even within `dev2b`, the hang direction varies ~20° (gesture vs set starts).

## Teacher error margin: what widens it
| Source | Effect | Size / handling |
|---|---|---|
| Clock sync | Subtracting readings from slightly different instants | ~3° per 20 ms at 150 °/s; measured 0.1–3 ms after sync |
| Armband wobble on the upper arm | Fake upper-arm motion → fake drift | Phone on the outer upper arm, above the elbow, tight |
| Fast motion shaking the accelerometer | "Down" briefly wrong | Gyro carries fast motion; ~2° range error on the simulated arm |
| Mounting error (USB not exactly along the forearm) | Fixed offset in forearm angle from vertical | Measured every session: gesture vs USB axis |
| Drift outside the curl plane | Elbow bend = forearm − upper-arm angle is exact only for in-plane drift | Small for bicep curls (drift is roughly in the curl plane); preacher curls out of scope |
| Twist leaking into the hinge estimate | Some curl counted as wobble or the reverse | Doesn't affect range (range comes from gravity) |
| Straps moved mid-session | Straight-arm reference invalid for later sets | Detected by the start-vs-end gesture check |

## Phase 5 extras (cut-if-behind; the main result doesn't depend on them)
- [ ] Live replay: feed each set sample by sample with a rail/axis that sharpens as reps accumulate; report live-vs-hindsight count disagreement and lag (~1 h)
- [ ] Auto set detection on the continuous stream (rhythmic bursts between rests); report count accuracy with detected vs Enter-press boundaries (~1–2 h)

## Rules
1. Log every block: planned vs actual hours, one line on why they differed.
2. At 1.5× a block's budget, stop and choose: cut scope or spend buffer.
3. Cut order if behind: live demo → replay; friends → just you; third exercise; calibration plot.
4. Never cut: frozen counter, held-out split, comparison against simple heuristics.
5. Explain before code: every component gets a what/why before it's written.

## Decisions
| Date | Decision | Why |
|---|---|---|
| 10-26 | 50 Hz polling of the 100 Hz BMI270 | Rep motion is <5 Hz; matches Fort's battery constraints |
| 10-26 | Integers on the wire (mg, centi-dps) | Float printf support unconfirmed on this core; resolution still at the noise floor |
| 10-26 | One continuous raw file per session + separate annotations | Keeps between-set data for set detection; labels can be redone without touching the recording |
| 10-26 | Scripted protocol is ground truth, video only for spot checks | Cuts annotation from ~a day to ~an hour |
| 10-26 | Firmware sleeps (`delay(1)`) when >2 ms to the next sample, else `yield()` | Busy-waiting starved mbed's USB threads (see Bugs) |
| 10-26 | Add a form layer: phone on the upper arm as a lab-only reference IMU, plus scripted form errors | Wrist sees forearm = upper arm + elbow and can't separate them; the phone supplies the missing term so form labels are measured, not assumed. The wrist stays the only product sensor (teacher/student). |
| 10-26 | Sync gesture (3 straight-arm swings) at the start and end of every session | Phone and Nano have separate clocks; a shared event gives offset (start) and drift (start vs end) |
| 10-27 | Curls only by default; raises and presses only via `--exercises`, as an unfamiliar-exercise test | The upper-arm phone (teacher) can only check curls |
| 10-27 | Every session: curl normal, slow, fast, pause, alternating ×1, partial ×2, drift ×2; distractors walk, rerack, adjust; exhaustion set last before the final sync. Band never rotated mid-session; mounting varies between sessions | Covers the count model's failure modes (tempo, pauses, other-arm motion, non-rep motion) while hitting half ≈12% / drift ≈7%. `drink` dropped: sipping is mostly still and would fail the validator's movement check |
| 10-27 | Half reps split into `half_lift` and `half_lower` (2 each per partial set; extra half reps pick one at random); a lift after a `half_lower` is never `half_lift` or `pause` | They fail differently: lift-halfway shows in range of motion, lower-halfway in how far short of the bottom the rep stopped. After a lower-halfway the next lift already starts halfway up, so those two instructions would be no movement |
| 10-28 | Label maker v2: home from each set's rep bottoms (range); straight-arm home from the gesture (teacher) and from the fixed USB-along-forearm mounting (student); the gap between them measured per session as mounting error | The gesture's shoulder plane isn't the curl plane, so axis calibration failed (35°, ~55°). Gravity-based tilt from home works in any curl plane. Fixed mounting gives the student the same reference without calibration, and measuring the gap turns that assumption into a number |
| 10-28 | Mounting rule: USB toward the elbow, board flat on the back of the forearm, two fingers above the wrist bone, line drawn on the strap; don't touch straps mid-session | Makes the along-forearm direction known to the student; re-adjusting mid-session changes the mounting |
| 10-27 | Hand-marking is a separate, append-only log (`data/manual/`), latest row wins, `ok` clears; not merged into `check_marks` yet | Scripted marks say what was meant; hand marks say what was seen (needed for the unscripted exhaustion set). Append-only keeps every judgment and correction on record |

| 10-29 | Alternating sets never get `half_lower` marks (lift-halfway is still allowed) | You can't lower one arm halfway and then start the next lift with the other hand (dev2b set 8) |
| 10-29 | Count ground truth = typed full reps, corrected by an append-only fix log when the raw wrist data clearly shows otherwise | dev2b sets 8 and 14: the ax humps show the typed count was off. Raw annotations are never edited |

| 10-30 | Protocol: press Enter with the OTHER hand and let the sensor arm hang still for a second before lifting (receiver prompts it every set) | Pressing Enter with the curling hand left the arm bent at set start |
| 10-30 | Teacher and learned detector measure from the hang at the set's own rep bottoms, and ignore everything before the forearm first comes back near it (`labels.settle`). Bottom vs top end of the swing picked by the USB direction when known, else by time spent near each end | Sets starting bent (dev2b 4–7) had every rep read 35° short: the detector rejected whole sets, the teacher added a lift at the start |

| 10-30 | Three frozen rep counters, judged on held-out people (p2 onward): `counter-v2` (rule: hump height vs the set's typical, cut 0.825), `detector-v1` (logistic regression on 12 wrist features per hump, trained on all dev humps), `combined-v1` (a rule-counted hump counts only if the learned model gives it p ≥ 0.05, gate fixed before seeing results) | Rule never misses a real rep but counts any wiggle (33 fake reps on dev distractors); learned ignores wiggles (5) but drops unusual full reps (first, paused, slow). Dev, learned parts held out: rule 20/28 sets right (optimistic), learned 21/28, combined 23/28. Evaluate with `python -m src.detector <sessions> --evaluate` |

## Bugs and lessons
| Date | Symptom | Cause | Fix | Lesson |
|---|---|---|---|---|
| 10-26 | First flash OK, every later upload `exit status 1`; board invisible to the Mac; orange LED off | `loop()` busy-waited on `micros()` and never gave up the CPU, starving mbed OS threads, including the USB thread that handles the 1200-baud "reset for upload" | Sleep/yield while waiting; recover with a double-tap reset into the bootloader | On an RTOS, busy-waiting takes CPU from other threads. Confirmed: after the fix, a normal upload works with no double-tap. Open: why the LED stayed off (the loop should still have toggled it) |
| 10-26 | Failed receiver start left an empty raw file, blocking the next run | File created before the first sample arrived | Delete the file if no data arrives | A failed start should leave no trace |

| 10-29 | Rail fix on `dev1b`: fake lifts gone (alternating 7/7), but 3 sets miss one lift and the forearm sits 20–45° off its fitted rail on typical reps | The rail assumes the forearm's "down" moves on one great circle. Real curls twist the forearm (palm turning) during the lift, which moves "down" off any single rail | A twist-proof measure: the forearm's angle from vertical, using its long axis (USB along the forearm). Twisting around that axis doesn't change it, and a forearm can't hang below vertical, so no sign is needed | Test on real data early: the simulated arm had no forearm twist, so 52 passing tests couldn't catch this |

| 10-29 | Label maker v3 (forearm angle from its hanging axis; per-lift twist, wobble, plane tilt; lifts tilted >45° off the set's curl axis not counted) on `dev1b`: lift counts match the script in 8 of 9 scripted sets (pause set +1); twist 21–48° per rep, which explains why the rail failed; drift still marks lifts 4, 7, 10 of set 2 | Forearm angle from vertical can't be changed by twist or curl plane | Freeze the teacher here; check the pause set's extra lift later | A sideways lean shows up mostly as twist once the forearm points forward: from the wrist, lean and twist can't be fully told apart near horizontal (observability limit, report it) |

| 10-30 | Nano angles on real data lagged ~1 s and flattened humps; teacher ranges wrong (half reps measured larger than full), counter 52% exact | The Arduino library's remap (x = -chip y, y = -chip x) mirrors the frame: accelerometer directions survive, gyro turns come out reversed, so following "down" with the gyro turned the wrong way and the 2% accelerometer nudge dragged it back late | Undo the mirror at load (flip z on accel and gyro); validator `check_frame` FAILs if gyro-only "down" drifts >20° in 3 s (as logged 92°, fixed 5°); clipped windows skipped | Simulated sensors were correctly handed, so tests couldn't see it. Check sensor conventions on real data with a physics consistency test. After the fix: teacher 9/9 lift counts on dev1b, sane ranges on dev2b; counter 90–95% exact on dev |

| 10-30 | Learned detector zeroed whole sets (dev2b 4–7, dev1b 5, 12): 57% exact held-out | Every angle was measured from the first instant of the set; you press Enter with the curling hand, so that instant was bent. Reps read ~35° short = "lower halfway" | `settle`: zero from the set's rep bottoms, pre-set jitter dropped. Held-out: 76% exact, 90% within one, 5 fake reps in distractors (rule counter: 33). Teacher: 9/9 and 8/9 (the miss is the set where 7 lifts were done, not the scripted 8) | Numbers measured from a reference inherit its mistakes; a learned model leaning on them breaks whole sets at once. Simulation needed a realistic reach (smooth, with a hang) to test it fairly |

## Commands
```bash
arduino-cli board list   # port name changes with the USB socket (usbmodem101, usbmodem1101, ...)
arduino-cli compile --upload -p /dev/cu.usbmodem1101 --fqbn arduino:mbed_nano:nano33ble firmware/sensor_logger
.venv/bin/python -m src.record --session validation --script protocols/validation.csv
.venv/bin/python -m pytest -q
```

## Time log
| Date | Block | Planned h | Actual h | Why different |
|---|---|---|---|---|
| 10-26 | Phase 1: firmware, receiver, flashing, validation recording, validator | 4.0 | 2.0 | Code generation was fast. Time went to the RTOS/USB bug and the mislabeled first recording. Still open: wrist strap, validator tests, stale receiver test. |
| 10-26 → 10-28 | Phase 2 so far: phone test, sync, label maker v1, feature plan, generator review and briefs, `dev1b` recorded and processed, calibration redesign | 6.0 (whole phase) | 5.5 | Design iterations (calibration twice, teacher/student consistency) and safety-filter interruptions. Still open: label maker v2, `dev2`. Running total 7.5 h of 36. |
| 10-29 | Phase 2 finish (label maker v3, `dev2b`) + Phase 3 (wrist-only counter, frozen as `counter-v1`) | Phase 3: 5.0 | 3.0 | Counter design was quick once the teacher work had shown the forearm angle is the clean signal. Running total 10.5 h of 36; Phase 3 was planned to end at H15, so ~4.5 h ahead. |
