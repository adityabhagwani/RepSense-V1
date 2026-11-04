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
| 10-30 | No neural net for judging humps: keep the logistic regression (13 parameters). Rule for any future net: parameters ≤ 50% of the training examples | Tested on dev, held out by session: a one-hidden-layer net on the same 12 features, as big as the cap allows per fold (2–6 hidden units, 29–85 parameters), 10 random starts. Curl sets exact: learned 10/21 median (6–16) vs logistic 16/21; combined 14/21 (9–16) vs 19/21. Fake reps 4 vs 5–8. 185 labeled humps from one arm is too little for a net, and its answer changed with the random start. Revisit with many people's data |
| 10-30 | Drift from the wrist alone: yes/no works, exact degrees don't (exploratory, dev only, p2 not looked at) | Drift adds to forearm tilt, so drifting reps top out higher: median top 157° at ≥25° drift vs 142° under 15°. `top_deg` alone separates drift reps at cutoffs 15–37.5° (AUC 0.83–0.98; shuffled-label 95th percentile 0.59–0.75); holds without partial and exhaustion sets (0.77–0.94), and against the set's own typical top (0.84–0.96). No dev rep reaches 40° (max 39.4°), so the 40° cut can't be tested yet. Exact degrees: a ridge on 13 features misses by 5.2° on average vs 6.6° for guessing the mean, and normal reps already read ~11° (teacher noise floor). Hinge residual (#20) not built yet |
| 10-30 | Frozen before looking at p2's drift numbers: `drift-flag-v1` = a rep drifted if its forearm top angle (`top_deg`, from the hang at the set's rep bottoms) ≥ **154°** (dev: best split for ≥25° drift; catches 81%, false alarm 3%). Scored once on `p2_s1`, every teacher-labeled rep, no exclusions. **Primary:** AUC of `top_deg` at drift cutoffs 15–40° (2.5° steps, cutoffs with ≥4 drifting reps), each beating the 95th percentile of 200 shuffled-label runs = pass. **Secondary:** the 154° flag's catch and false-alarm rates at ≥25° (and ≥40° if p2 has 4+ such reps); `top_vs_set` AUC | A one-shot held-out test of the dev finding, on a different person |
| 10-30 | **`drift-flag-v1` on p2_s1, held out: primary PASS.** The ranking works, but the fixed 154° cut doesn't transfer between people | AUC 0.89–1.00 at all 11 cutoffs (shuffled 95th 0.57–0.69), including ≥40° (7 reps, AUC 1.00). The 154° flag: at ≥40° catches 7/7 with 6% false alarms; at ≥25° catches only 13/48 (0 false alarms). p2 tops out lower than you (median 134° vs ~142°) and drifts more (median 23°, max 79°), so the cut needs to be per person. `top_vs_set` 0.73: p2 drifted through whole sets, so the set's own typical top already includes drift. Teacher "drift" = upper arm tilting from hanging, so a torso lean counts too |
| 10-30 | Drift detection stays exploratory; not needed for the Fort pitch (reps, velocity loss, failure). Torso lean counts as a bad rep too | A per-person cut needs clean reps, and some people drift on every rep (p2). One calibration-free lead, exploratory (dev and p2 already seen): an elbow bends ~150° at most, so a forearm past 150° from hanging means the upper arm moved by at least the excess. Held on all 290 dev + p2 reps (teacher drift ≥ top − 150°, 0 violations); above 155°, every p2 rep drifted ≥30°. One-sided: misses moderate drift |
| 11-02 | Teacher phone goes in a running armband on the outer upper arm, lower half, from p2_s2 on (was taped inside the biceps) | band_test vs dev2b: same curl-angle humps and lift counts, drift reps stand out the same (28–40° vs ~10° strict), cleaner set starts, best sync yet (0.995), and no tape to break with the pump. Cost: fast reps read ~8° more drift (19° vs 11° median), either the phone rocking or real arm swing; acceptable while drift is exploratory. Plot: `reports/band_vs_tape.png` |
| 11-04 | **Frozen counters on p4_s1 and p5_s1 (new people, armband), counts settled and flagged sets fixed before evaluating: combined 14/14 clean curl sets exact.** Pooled held-out (p2, p4, p5): combined 22/22 clean, 6/8 flagged; rule 22/22 and 6/8 but 34 fake reps in 9 distractors; learned 18/22 and 3/8, 1 fake rep. `combined-v1` stays the headline | p4+p5: rule 14/14 clean, 4/6 flagged, 19 fake reps in 6 distractors; learned 11/14, 2/6, 1; combined 14/14, 4/6, 4. The two flagged misses: p5 set 2, 13 fast lifts all ruled half reps (tops 109–131° vs p5's 140–166°): every counter says 13 (learned 9), because each hump is judged against the set's own typical hump, so a set where every rep is short looks normal. Needs the person's own full-range baseline from earlier sets (not built). p5 set 4: 8 vs 7, one half rep counted. The unfrozen rest/set flag (rule − combined ≥ 2) would catch 4/6 distractors here, 0 real sets wrongly flagged |
| 11-04 | Counter failure modes on held-out p2, p4, p5, and what we do about each | **FM1** the rule judges each hump against its own set's typical hump, so a set of uniformly small movements passes (the rule counted reps in all 9 distractors; p5 set 2, 13 short fast lifts ruled half reps, counted ~13 by every counter): **left**, Fort's long-term tracking learns each person's range, 10 sets/person can't. **FM2** the learned model's degrees come from one arm (dev tops ~140–150°; p4's reps ~115–125° score p ≈ 0.5): fixed by the same per-person range. Its fast-lowering penalty is a dev-data artifact (quick lowering came from half reps there), but combined only vetoes p < 0.05 and these reps scored 0.21–0.49, so FM2 never reaches the headline counter. **FM3** half-lowers (learned counts both odd humps or neither; the rule drops one, sometimes the wrong one): **out of scope**, nobody does them naturally and only the total count matters. **FM4** one real elbow bend inside a distractor (dumbbell pick-up, strap fiddling; tops 82–139°, p 0.06–0.64) survives the 0.05 veto, exactly one per distractor: **handled by `min-reps-v1`**. The veto zone held: of 49 held-out humps with p < 0.05, 0 were real reps |
| 11-04 | Rest/set flag (rule − combined ≥ 2) dropped for `min-reps-v1` | Flag: distractors dev 6/6, held-out 6/9 (p4/p5 alone 4/6), real sets flagged 0/50. A simpler rule beats it (≤1 combined-counted rep = not a set: 15/15 distractors, 0/50 real sets) because FM4 leaks one bend per distractor. The flag needs a whole set's counts, so it can't run continuously; min-reps can |
| 11-04 | Frozen before the next sessions: **`min-reps-v1`** = `combined-v1`, then a counted rep counts only if another counted rep peaks within **12 s** of it, on either side. Applied inside each recorded set; the continuous replay applies the same rule to the whole stream. Scored once on the next new sessions, every set, flagged sets reported separately. **Pass:** fewer fake reps in distractors than `combined-v1`, and no real set's count further from the truth than `combined-v1`'s. Report: distractors at 0, real sets changed, exact counts before/after | 12 s: the longest lift-to-lift time in any real set is 9.7 s (dev2b slow; p5 8.4 s), the nearest leaked bend to a real rep is 16 s. On sets already seen (dev with the learned part held out, p2, p4, p5): distractors 15/15 at 0 (6 fake reps → 0); real sets changed 1/50 (dev1b set 5, a scripted partial set opening with a half-lower: 5 → 4, now wrong). Open risk: a dumbbell pick-up or put-down within 12 s of a set adds a rep |
| 11-04 | Frozen before any held-out stream is replayed: **`stream-v1`** (`src/stream.py`) finds sets in a whole recording without the Enter presses: a lift = a forearm hump ≥ 50° tall (angle from the USB direction); 2+ lifts with tops ≤ 12 s apart = a set, padded 2 s; each set counted by `combined-v1` + `min-reps-v1`, unchanged. Scored once on p2_s1, p3_s1, p4_s1, p5_s1; flagged sets and p3's unsettled alternating set left out (23 clean curl sets). **Pass = all three:** (1) ≥ 22/23 clean sets found exactly once; (2) ≥ 21/23 clean sets get the same count as with the Enter presses; (3) ≤ 1 fake rep per 10 min of between-set rest, pooled. Reported, no bar: reps in p3's 84-min idle stretch, flagged sets, exact counts vs truth, distractor and swing reps, set start/end timing | Dev (learned part held out): 20/20 found once; 19/20 same count (the miss: dev1b set 10, a half-lower pair where the earlier set start flipped which hump the rule dropped, then the veto removed the half); 0 rest reps in 15 min; 0 edge reps; swing gestures 7 reps in dev1b; set start/end within ~4 s of the Enter presses |
| 11-04 | **`stream-v1` on held-out p2_s1, p3_s1, p4_s1, p5_s1: FAIL** (criterion 3). (1) clean sets found once 23/23, pass; (2) same count as with the Enter presses 22/23, pass; (3) rest reps 1.58 per 10 min (6 in 38 min), fail (bar ≤ 1). Reported: p3 idle stretch 0 reps in 84 min; clean exact vs truth 23 → 22/23 | All of the failure is p5: p2, p3, p4 had 0 rest reps in 24 min. p5 made big curl-like bends (to ~150°) outside the sets: 3 at 0:28–0:41 (before the first swing gesture) and 3 at 12:12–12:19 (a 3-min break between sets 10 and 11) = the 6 rest reps; 7 at 4:01–4:35, just before set 4 (partial), which the set finder merged into set 4 (14 vs 8 with Enter); 2 in the rerack distractor (4 reracks, two bends 6 s apart, so min-reps keeps both). Cause unknown without video; by the pre-registered rule they count as fakes. The count miss (p2 set 3) is a scripted half-lower set, like dev1b set 10 |

## Bugs and lessons
| Date | Symptom | Cause | Fix | Lesson |
|---|---|---|---|---|
| 10-26 | First flash OK, every later upload `exit status 1`; board invisible to the Mac; orange LED off | `loop()` busy-waited on `micros()` and never gave up the CPU, starving mbed OS threads, including the USB thread that handles the 1200-baud "reset for upload" | Sleep/yield while waiting; recover with a double-tap reset into the bootloader | On an RTOS, busy-waiting takes CPU from other threads. Confirmed: after the fix, a normal upload works with no double-tap. Open: why the LED stayed off (the loop should still have toggled it) |
| 10-26 | Failed receiver start left an empty raw file, blocking the next run | File created before the first sample arrived | Delete the file if no data arrives | A failed start should leave no trace |

| 10-29 | Rail fix on `dev1b`: fake lifts gone (alternating 7/7), but 3 sets miss one lift and the forearm sits 20–45° off its fitted rail on typical reps | The rail assumes the forearm's "down" moves on one great circle. Real curls twist the forearm (palm turning) during the lift, which moves "down" off any single rail | A twist-proof measure: the forearm's angle from vertical, using its long axis (USB along the forearm). Twisting around that axis doesn't change it, and a forearm can't hang below vertical, so no sign is needed | Test on real data early: the simulated arm had no forearm twist, so 52 passing tests couldn't catch this |

| 10-29 | Label maker v3 (forearm angle from its hanging axis; per-lift twist, wobble, plane tilt; lifts tilted >45° off the set's curl axis not counted) on `dev1b`: lift counts match the script in 8 of 9 scripted sets (pause set +1); twist 21–48° per rep, which explains why the rail failed; drift still marks lifts 4, 7, 10 of set 2 | Forearm angle from vertical can't be changed by twist or curl plane | Freeze the teacher here; check the pause set's extra lift later | A sideways lean shows up mostly as twist once the forearm points forward: from the wrist, lean and twist can't be fully told apart near horizontal (observability limit, report it) |

| 10-30 | Nano angles on real data lagged ~1 s and flattened humps; teacher ranges wrong (half reps measured larger than full), counter 52% exact | The Arduino library's remap (x = -chip y, y = -chip x) mirrors the frame: accelerometer directions survive, gyro turns come out reversed, so following "down" with the gyro turned the wrong way and the 2% accelerometer nudge dragged it back late | Undo the mirror at load (flip z on accel and gyro); validator `check_frame` FAILs if gyro-only "down" drifts >20° in 3 s (as logged 92°, fixed 5°); clipped windows skipped | Simulated sensors were correctly handed, so tests couldn't see it. Check sensor conventions on real data with a physics consistency test. After the fix: teacher 9/9 lift counts on dev1b, sane ranges on dev2b; counter 90–95% exact on dev |

| 10-30 | Learned detector zeroed whole sets (dev2b 4–7, dev1b 5, 12): 57% exact held-out | Every angle was measured from the first instant of the set; you press Enter with the curling hand, so that instant was bent. Reps read ~35° short = "lower halfway" | `settle`: zero from the set's rep bottoms, pre-set jitter dropped. Held-out: 76% exact, 90% within one, 5 fake reps in distractors (rule counter: 33). Teacher: 9/9 and 8/9 (the miss is the set where 7 lifts were done, not the scripted 8) | Numbers measured from a reference inherit its mistakes; a learned model leaning on them breaks whole sets at once. Simulation needed a realistic reach (smooth, with a hang) to test it fairly |
| 11-02 | Strap check flagged band_test (14°) and both taped sessions; the wrist "slipped" 20–37° too | It measured each sensor's hanging "up" change alone, so the whole arm hanging at a new angle at the last gesture looked like two straps moving | Flag only when one sensor's change exceeds the other's by >10° (elbow locked in the gesture, so posture turns both equally). Now: band_test OK (14.0 vs 14.4°), p2_s1 OK, dev1b wrist flagged (+18°), dev2b taped phone flagged (+16°) | A check on one sensor can't tell its mount from the body; compare against a second sensor that shares the body's motion but not the mount. Sizes only, not directions: a strap move on top of a posture change can partly hide |
| 11-04 | Settling p4_s1 set 5 from raw ax: proposed "7 or 9" because lifts 5 and 8 topped at 0.6–0.7 g vs ~0.9 | ax ≈ −cos(angle) only if the board's x runs along the forearm. p4's board sat ~33° off (hanging read −0.83 g, not −1), so level read +0.55 g. In degrees the two lifts topped at 94–95°: plain half lifts | Settle plots show forearm degrees from the hang (0 = hanging, dashed line at level). Raw ax is still fine for counting humps | Raw axes mix the motion with how the sensor sits; judge heights on a mount-independent quantity (angle from the hang) |

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
| 10-30 | Axis fix, counter-v2, learned detector + combined (frozen), protocol, p2_s1 and p3_s1 collected and scored | Phase 4: 6.0 (whole phase) | 4.0 | Mirrored-axes bug and the start-pose bug took most of it; collection itself was quick. Running total 14.5 h of 36. Phase 4 was planned to end at H19 and is ~40% done (1.5 of 4 sessions), so roughly on schedule. |
| 11-03 | Data collection: `p4_s1` and `p5_s1` (new people, armband teacher), scripts, two-person run order | Phase 4 (cont.) | 2.0 | Phase 4's 6 h is now used (4.0 + 2.0). Collected: p2_s1, p3_s1 (cut short), p4_s1, p5_s1; second sessions not recorded. Running total 16.5 h of 36, ahead of the H19 end of Phase 4. |
