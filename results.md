# RepSense results

**Question.** Can one sensor on the wrist count dumbbell curl reps, and tell when its own count is likely wrong?

**Short answer.**
- **Counting: yes, on two new people.** The counter was locked before their data was scored. It got 19 of 20 curl sets within 1 rep (16 of 20 exact). It counted 4 false reps in 6 non-curl sets.
- **Knowing when it is wrong: not shown on real sets yet.** The error flag works on synthetic sets (AUC 0.91). Real data has only 9 wrong counts in 72 sets, which is too few to test it.

Every number below comes from `PLAN.md`. That file is the dated log of each decision, each test and its pass bar, each result and each bug.

## 1. Setup

| Item | What we used |
|---|---|
| Sensor | Arduino Nano 33 BLE Sense Rev2: BMI270 accelerometer and gyroscope at 50 Hz |
| Position | Back of the forearm, two fingers above the wrist bone, USB toward the elbow |
| Exercise | Dumbbell biceps curls, one arm |
| People | 7. p1 is the author (3 design sessions). p2 to p7 are friends (1 session each). Anonymous IDs only |
| Lab reference | A phone on the upper arm measures the elbow angle and elbow drift. Only the labels use it. The counter does not use it |

**Sessions are scripted.** Each set has an instruction: normal, slow, fast, pause, alternating arms, partial (half reps), drift (elbow moves forward), a rest in the middle, or exhaustion (reps until failure). Each session also has non-curl sets: walk, rerack the dumbbell, adjust the strap.

**True counts.**
- The true count is the number of full reps that the person typed after the set.
- A fix log (`data/count_fixes.csv`, append only) corrects a typed count when the raw angle clearly shows a different count.
- A lift is a full rep only if the forearm comes back to within 15° of hanging before the next lift. The hang is measured for each session.
- A **flagged** set is a set where the person's paper note says the set did not go as scripted. Flagged sets are scored separately. All other sets are **clean**.

## 2. The counter

The counter has three steps. Each step was tagged in git before it was scored.

1. **Height rule (`counter-v2`).** It tracks the forearm angle from hanging, from gravity and the gyroscope. Twist does not change this angle. A hump counts if it is at least 0.825 × the set's median hump height.
2. **Checker (`detector-v4`).** A logistic regression on 12 features of each hump (height, top angle, rise time, twist and others). It removes a rule hump when p < 0.05. It was trained on p1's 3 design sessions only.
3. **Lone-rep check (`min-reps-v1`).** A counted rep stays only if another counted rep peaks within 12 s of it. This removes single bends, for example picking up a dumbbell.

A set ends 12 s after its last lift.

**Why three steps.** On p6 and p7, each part alone fails:
- The height rule alone counted 30 false reps in the 6 non-curl sets.
- The checker alone got only 8 of 20 curl sets exact.
- Together, the counter got 16 of 20 exact and counted 4 false reps.

## 3. How we tested

- **Design data ("seen").** This is p1's sessions plus p2 to p5. Each of p2 to p5 was a held-out test one time (section 5). After its test, each became design data.
- **Final test.** This is p6 and p7. Before scoring:
  - Each test and its pass bar were written down (`PLAN.md`, rows dated 11-04 and 11-07).
  - The code was tagged.
  - The true counts were settled from the raw angle only. No counter output was looked at.
  - Then each test was scored one time.

## 4. Main result: p6 and p7 (pre-registered, scored once)

**The shipped counter:** height rule v2, checker v4 and the 12 s check.

| Sets | Exact | Within 1 rep |
|---|---|---|
| Clean curl sets (16) | 14/16 | 15/16 |
| Flagged curl sets (4) | 2/4 | 4/4 |
| **All curl sets (20)** | **16/20 (80%)** | **19/20 (95%)** |
| False reps in 6 non-curl sets | 4 | |

In 1 of 20 curl sets (5%), the count was off by more than 1 rep.

| Pre-registered test | Pass bar | Result |
|---|---|---|
| Checker v4 against checker v1, in the full counter | Clean exact ≥ v1; false reps ≤ v1 | **PASS:** 14/16 against 13/16; 4 false reps against 4 |
| 12 s check (`min-reps-v1`) | Fewer false reps than without it, and no real set further from the truth | **FAIL:** false reps 6 → 4, but p6 set 8 went from 9 (right) to 8 |
| Per-box lone-rep check (`min-reps-v2`) against v1 | v2 ≥ v1 on four measures | **PASS by a tie:** the same count on every set |
| Error flag (`count-error-v1`) | Lower AUC bound > 0.5; fewer than 5 wrong sets = inconclusive | **Inconclusive:** 4 wrong sets; AUC 0.64 (95% interval 0.37–0.89) |

The 12 s check failed its own bar, but we keep it. It removes a third of the false reps (6 → 4), and the per-box version gives the same counts.

**Rests in the middle of a set.** 7 scripted sets had a rest of 5, 10 or 15 s between two reps.
- With checker v4: 5/7 exact, 7/7 within 1.
- With checker v1: 4/7 exact, 5/7 within 1.

Checker v1 removed a real rep after a rest in 3 of the 7 sets. Checker v4 did this in none of them.

## 5. Earlier held-out rounds

Each round was scored one time, with the counter locked before.

| Round | People | Counter | Result |
|---|---|---|---|
| 10-30 and 11-04 | p2, p4, p5 | Rule + checker v1 (no 12 s check) | Clean curl sets 22/22 exact; flagged 6/8 exact. The rule alone counted 34 false reps in 9 non-curl sets |
| 10-30 | p2 | Elbow drift flag (`drift-flag-v1`) | The ranking passed: AUC 0.89–1.00 at all 11 drift cutoffs. The fixed 154° cut did not transfer between people, so the cut must be per person |

p3's session stopped after 3 sets, because the tape on the phone broke.

## 6. Whole recordings without set marks (replay)

In the scored sessions, the person pressed Enter at the start and end of each set. The replay (`stream-v1`) finds the sets in the whole recording without the Enter presses. Then it counts each set with the same counter.

| Data | Clean sets found once | Exact | Within 1 | False reps in rest |
|---|---|---|---|---|
| p2 to p5 (pre-registered) | 23/23 | 22/23 | | 1.58 per 10 min: **FAIL** (bar ≤ 1). All of them are from p5's bends between sets |
| p6, p7 (reported, no bar) | 14/16 | 12/16 | 13/16 | 1.68 per 10 min (3 in 18 min) |

On p6 and p7, a mid-set rest split 2 sets in two (p6 set 8, p7 set 9). The rest put more than 12 s between two rep tops.

## 7. The shipped counter's 8 errors on p6 and p7

We traced each error hump by hump, after scoring.

| # | Set | Counted / true | Cause | How sure |
|---|---|---|---|---|
| 1 | p6 set 8 (fast, rest before the last rep) | 8 / 9 | The 12 s check dropped the last rep. It came 13.2 s after the rep before it | Certain |
| 2 | p7 set 13 (drift) | 3 / 5 | 3 of 5 reps drifted and rose to 151–153°. The height bar is a share of the set's typical rep, so it rose above the 2 normal reps | Certain |
| 3 | p6 set 12 (flagged, partial) | 13 / 12 | A half lift counted. The rule measures from the set's first 0.3 s, and the arm was 29° off hanging at that time. So full reps looked shorter, and the half lift reached 0.91 of a typical rep (really 0.76) | Certain. A known fix (measure from the session's hang) is not locked yet |
| 4 | p7 set 14 (flagged, exhaustion) | 9 / 8 | The failed 9th rep rose to 0.88 of a typical rep, above the 0.825 cut | Certain |
| 5–6 | p6 set 4 (adjust the strap) | 2 false reps | Two bends to 127–134°, 1.7 s apart | Probably the wrist rose to reach the strap. There is no video to confirm |
| 7–8 | p6 set 11 (rerack) | 2 false reps | Two dumbbell pick-ups to 105–121°. A loaded pick-up is a real elbow bend | Certain about the movement |

**Four causes:**
1. A long rest against the 12 s check (error 1).
2. The relative height bar (errors 2 and 3).
3. A nearly complete rep (error 4).
4. Curl-like bends outside a set (errors 5 to 8).

## 8. Can it tell when its count is wrong?

**The error flag (`count-error-v1`)** is a logistic model on three signs for each set:
- movement that is not a rep,
- how much the three counters (rule, checker, combined) disagree,
- seconds for each counted rep.

It gives the risk that the counter got the set wrong.

| Data | Sets (wrong) | AUC (95% interval) | Wrong sets after hiding the riskiest 20% |
|---|---|---|---|
| **Synthetic:** spliced sets, held out by source session | 3,000 (13%) | 0.91 | 13% → 3.0% |
| Real, seen during design (p1 to p5) | 52 (5) | 0.76 (0.57–0.94) | |
| Real, held out (p6, p7) | 20 (4) | 0.64 (0.37–0.89) | Inconclusive |

**What a synthetic set is.** We cut real reps at their bottoms and join them into new sets. Then we add trip-ups at random: long rests, half reps, other movement inside the set, reracks and others (13 in total). The true count is known by construction. Synthetic sets hold only the trip-ups that we thought of, so they are easier than real sets.

A pre-specified model on 6 other signs scored AUC 0.85 (0.83–0.87) on the synthetic sets: 13% → 5.6%. The three signs above were selected from 15 candidates on synthetic data.

**The live label.** The screen shows "check this set" when the risk is ≥ 0.1346, the 80th percentile on synthetic sets. On real sets, the label gives many false alarms:
- In the p7 replay, it flagged 6 of 8 sets, and 4 of those 6 were counted correctly.
- In the demo video, it flagged set 2, which was counted correctly.

**Conclusion.** The flag ranks wrong counts well on synthetic sets. On real sets, we have too few wrong counts (9 in 72) to show that it works. It needs more real sets with errors.

## 9. Crash test (synthetic)

On 3,000 synthetic sets, the shipped counter got 80.2% exact and 91.8% within 1.

- **Half-lowers caused the largest number of errors:** +26 points of error when they were present.
- **Long rests at the bottom of a rep:** checker v4 fixed most of this damage. After a 10–15 s rest, the rule + checker counter was wrong on 5% of sets with v4, against 68% with v1.

## 10. Live demo over Bluetooth

- Firmware v2 sends each sample over Bluetooth Low Energy as one 20-byte message.
- **Link check:** 50.00 Hz, 0 samples lost in 60 s; 95% of samples arrived within 23 ms.
- **The live screen** shows "Resting", "Set in progress", or "Set N ended" with the reps, the time and the label. It uses the same code as the replay, so the live counts are the replay counts. We checked this on p6 and p7.
- **Live runs with true counts** (p1 only): 4 of 4 sets exact (7, 10, 9 and 6 reps). The sets had half lifts, half-lowers and pauses. During the runs, the link ran at 49.93–49.99 Hz and lost 0.02–0.12% of samples.

This is a demo check (1 person, 4 sets), not a result.

## 11. Limits

1. **Small data.** 7 people, 1 session each for p2 to p7, one exercise. The checker learned from p1's arm only (185 humps).
2. **The height bar is relative to the set.** A set where most reps drift high, or where every rep is short, misleads it (p7 set 13; p5 set 2). A per-person range would fix this. That needs many sets from each person.
3. **A rest of approximately 10 s or more at the bottom of a rep splits a set**, because the gap between rep tops becomes more than 12 s. Then the counter can drop a rep.
4. **Curl-like bends outside a set can count:** dumbbell pick-ups and strap adjustments. Between sets, this gave 1.6–1.7 false reps per 10 min of rest.
5. **The error flag is not proven on real data**, and it gives many false alarms.
6. **Elbow drift** from the wrist alone is exploratory. It is not in the counter or the demo.
7. **Live speed.** The live screen processes the whole recording at each step, so each step gets slower by approximately 0.04 s for each minute of recording. Restart the program between takes.

## 12. Repeat these numbers

```bash
.venv/bin/python -m pytest -q                                              # 117 tests
.venv/bin/python -m src.detector p6_s1 p7_s1 --evaluate --detector v4      # section 4 (Enter-press sets)
.venv/bin/python -m src.stream p6_s1 p7_s1 --detector v4 --min-reps v1     # section 6 (replay)
.venv/bin/python -m src.live --replay p7_s1                                # the live screen on a saved session
```

The `min_reps_count` column is the shipped counter.

## Appendix: every version on every real set

**Format:** exact / within 1 / false reps.
- **Seen data:** p1 design sessions and p2 to p5. 53 curl sets and 16 non-curl sets. For p1's sessions, the checker was held out by session.
- **New data:** p6 and p7. 20 curl sets and 6 non-curl sets.

This table counts all curl sets, flagged sets included. Every row uses the 0.05 checker limit. **Only the two bold rows were locked before p6 and p7.** We designed every other row with the seen data, so their p6 and p7 numbers are exploratory.

| Combination | Seen data | New data (p6, p7) |
|---|---|---|
| Height rule v2 alone | 49 / 50 / 67 | 17 / 19 / 30 |
| Height rule v3 alone | 49 / 50 / 69 | 17 / 19 / 27 |
| Checker v1 alone | 38 / 45 / 6 | 7 / 11 / 1 |
| Rule v2 + checker v1 | 48 / 50 / 13 | 15 / 18 / 6 |
| Rule v2 + checker v1 + 12 s check | 47 / 50 / 6 | 15 / 17 / 4 |
| Rule v2 + checker v1 + per-box check | 48 / 50 / 4 | 15 / 18 / 4 |
| Checker v2 alone | 38 / 46 / 6 | 6 / 10 / 4 |
| Rule v2 + checker v2 | 48 / 50 / 10 | 15 / 17 / 6 |
| Rule v2 + checker v2 + 12 s check | 47 / 50 / 4 | 15 / 16 / 4 |
| Rule v2 + checker v2 + per-box check | 48 / 50 / 4 | 15 / 17 / 4 |
| Checker v3 alone | 34 / 44 / 5 | 5 / 11 / 5 |
| Rule v2 + checker v3 | 47 / 50 / 12 | 15 / 19 / 6 |
| Rule v2 + checker v3 + 12 s check | 46 / 50 / 8 | 14 / 19 / 4 |
| Rule v2 + checker v3 + per-box check | 47 / 50 / 6 | 14 / 19 / 4 |
| Checker v4 alone | 38 / 45 / 6 | 8 / 11 / 2 |
| Rule v2 + checker v4 | 48 / 50 / 9 | 17 / 19 / 6 |
| **Rule v2 + checker v4 + 12 s check (shipped)** | **47 / 50 / 6** | **16 / 19 / 4** |
| **Rule v2 + checker v4 + per-box check (locked 11-07)** | **48 / 50 / 6** | **16 / 19 / 4** |
| Checker v5 alone | 34 / 46 / 5 | 10 / 14 / 3 |
| Rule v2 + checker v5 | 47 / 50 / 9 | 16 / 18 / 6 |
| Rule v2 + checker v5 + 12 s check | 47 / 49 / 6 | 15 / 18 / 4 |
| Rule v2 + checker v5 + per-box check | 47 / 50 / 6 | 15 / 18 / 4 |
| Checker v6 alone | 36 / 45 / 6 | 8 / 11 / 4 |
| Rule v3 + checker v6 | 49 / 50 / 13 | 17 / 19 / 6 |
| Rule v3 + checker v6 + 12 s check | 47 / 49 / 8 | 16 / 19 / 4 |
| Rule v3 + checker v6 + per-box check | 49 / 50 / 6 | 16 / 19 / 4 |

**What each version changes:**
- **Checker v1:** features measured from the lowest point since the last hump.
- **Checker v2:** the same features, measured over the lift only.
- **Checker v3 and v5:** v2 with one-sided rhythm features.
- **Checker v4:** v2 with no rhythm feature.
- **Checker v6:** v4 with the rule v3 height.
- **Rule v3:** a hump's size is how far it comes back down before the next real lift.
