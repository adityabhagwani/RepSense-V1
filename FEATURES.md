# Error-model features and selection protocol

Fixed on 2026-10-26, **before any results exist**. Changing the core set after seeing
held-out results would make the main result meaningless, so changes go in the log at the bottom.

## Rules every feature must follow
1. **Wrist only:** the product has no upper-arm sensor.
2. **Invariant to rotation around the forearm:** the same rep with the band turned around the wrist gives the same value (enforced by a test). The direction *along* the forearm is fixed by how the band is worn (USB toward the elbow), as on a product band; its human error is measured every session (teacher gesture vs USB axis).
3. **Relative within the set or rep:** no "position 0"; differences between the rep's own bottom and top, or comparisons with the set's typical rep.

★ = core set (the confirmatory model).

## Count model: one prediction per set, target = count wrong (yes/no)
| # | Feature | Computation | Catches |
|---|---|---|---|
| 1 ★ | Rhythm irregularity | Std ÷ mean of time between reps | Pauses, uneven cadence |
| 2 | Longest gap ratio | Longest gap ÷ median gap | Missed rep, long pause |
| 3 | Shortest gap ratio | Shortest gap ÷ median gap | Pause bump counted twice |
| 4 ★ | Weakest-peak ratio | Smallest peak height ÷ median peak height | Partial rep or noise counted |
| 5 | Peak-height spread | Std ÷ mean of peak heights | Mixed full and partial reps |
| 6 | Close calls | Share of candidate peaks within ±20% of the threshold | Counts decided by a hair |
| 7 ★ | Rhythm strength | Autocorrelation at the rep period (0–1) | No clear rhythm |
| 8 | Period vs count | Duration ÷ period, compared with the count | Counter and rhythm disagree |
| 9 ★ | Accel vs gyro disagreement | Difference between the two counters | Ambiguous sets |
| 10 ★ | Threshold stability | Does the count change at threshold ±15%? | Fragile counts |
| 11 | Motion outside reps | Share of the set's movement outside detected reps | Re-racking, adjusting, walking |
| 12 ★ | Seconds per rep | Duration ÷ count | Impossible pace |
| 13 | Signal quality | Lost and late samples, % clipped | Bad data, not bad counting |
| 14 | Exercise type | Curl, raise or press | Exercise difficulty |

## Form model: one prediction per rep, targets from the two-sensor teacher (range in degrees, partial yes/no, drift in degrees)
| # | Feature | Computation | Catches |
|---|---|---|---|
| 15 ★ | Gravity sweep | Angle between gravity direction at rep bottom and top | Range of motion |
| 16 ★ | Sweep vs typical rep | Rep sweep ÷ set median sweep | Partial reps |
| 17 | Which end fell short | Rep bottom and top directions vs set typical | Lifted vs lowered halfway |
| 18 | Integrated rotation | Gyro rotation summed over the rep | Range, a second source |
| 19 ★ | Sweep vs rotation agreement | #15 vs #18 | Forearm twist, upper arm moving |
| 20 ★ | Hinge residual | Measured accel vs pure-elbow-hinge prediction | Elbow drift, body swing |
| 21 | Rotation-axis wobble | How much the rotation axis wanders within the rep | Twist, drift |
| 22 ★ | Tempo | Up time, down time, ratio | Rushed lowering |
| 23 | Peak rotation speed | Fastest rotation in the rep; its decline across a set | Effort, near failure |
| 24 | Bottom tilt from straight | Angle between the USB (along-forearm) axis and straight down at the rep bottom | Never fully lowered, even when every rep in the set is shifted |

Features 19–21 also feed the form model's confidence.

## Teacher labels and their wrist clues
No teacher label without a wrist clue. What the wrist can't see becomes the student's uncertainty.

| Teacher label | Wrist clue | Expected recovery | When the clue fails |
|---|---|---|---|
| Rep count and boundaries | Forearm rotation (#1–#12) | Good | Rhythm and peak features flag messy sets |
| Range per rep (elbow bend from the set's rep bottoms) | Forearm tilt from home (#15, #16, #18) = elbow bend + upper-arm tilt | Good on reps with a still elbow | On drifting reps the wrist sees the sum; #19–#21 lower confidence |
| Drift (upper-arm tilt during the rep) | Hinge residual (#20), axis wobble (#21) | Partial: knows the pivot moved, not by how much | Becomes uncertainty on range |
| Bottom distance from straight | Bottom tilt from straight (#24), with fixed mounting | Good if the measured mounting error is small | Reported with the mounting-error number |
| ~~Curl plane~~ | None | Not a target | Not bad form, not observable |

## Selection protocol
**Main result (confirmatory):** the ★ core set, L2-regularized logistic regression, leave-one-session-out. One run, reported whatever it says.

**Exploratory: random-subset ranking (Aditya's method), run nested:**
1. For each held-out session, use **only the training sessions**.
2. Fit 100 models on random 8-feature subsets, each scored by leave-one-session-out within the training sessions.
3. Each feature's contribution = mean score of subsets that include it − mean score of subsets that don't.
4. **Shuffled-label check:** repeat steps 2–3 at least 20 times with the labels shuffled. A feature counts only if its real contribution beats the 95th percentile of its shuffled contributions.
5. **Stability:** repeat on bootstrap resamples and keep features in the top list in ≥70% of runs.
6. Fit the kept features on the training sessions and score the held-out session **once**.
7. Report: kept features per fold, how often each was kept, and the nested score next to the core set's score.

Correlated features (e.g. #1 and #2) split credit, so one of a pair being dropped is expected.

**Baselines both must beat:** each ★ feature on its own with a single threshold.

## Change log
| Date | Change | Reason |
|---|---|---|
| 2026-10-26 | Created | |
| 2026-10-28 | Rule 2 amended (fixed along-forearm direction); feature #24; teacher-label wrist-clue table | Two-home label maker; fixed USB-along-forearm mounting |
