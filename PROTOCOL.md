# RepSense data collection protocol: p2 and p3

**Goal:** the first fair test of the three frozen rep counters (`counter-v2`, `detector-v1`, `combined-v1`).
They have never seen p2 or p3, and **nothing gets tuned on this data**.

**Plan:** 2 friends × 2 sessions each, about 20 minutes per session. If someone is short on time, session 1 alone still counts.

| Session | Participant | Script | Mounting |
|---|---|---|---|
| `p2_s1` | p2 | `protocols/p2_s1.csv` | Position A |
| `p2_s2` | p2 | `protocols/p2_s2.csv` | Position B |
| `p3_s1` | p3 | `protocols/p3_s1.csv` | Position A (cut short after set 3: tape broke) |
| `p3_s1b` | p3 | `protocols/p3_s1b.csv` | Position A (redo of `p3_s1`) |
| `p3_s2` | p3 | `protocols/p3_s2.csv` | Position B |
| `p4_s1` | p4 | `protocols/p4_s1.csv` | Position A |
| `p4_s2` | p4 | `protocols/p4_s2.csv` | Position B |

---

## 1. The three rules that matter most
1. **You run the laptop; your friend only lifts.** You press Enter, read out the next set and type the count. That way their arm is never bent from reaching for the keyboard, which caused the bad set starts in `dev2b`.
2. **Nobody touches the straps or the phone mid-session.** If one slips or falls off, stop at the end of that set, fix it, and write the set number down. The start and end swing gestures will flag it anyway.
3. **Stay blind until the true counts are settled.** Don't run `src.detector --evaluate` or `src.counter`, and don't read the teacher's lift counts, until every set's true count is final (section 8).

---

## 2. Before anyone arrives (~20 min)
**Gear**
- [ ] Nano with the long micro-USB **data** cable, plugged into the laptop
- [ ] Laptop on a table at about elbow height, charger connected
- [ ] Phone with phyphox, **charged above 50%**
- [ ] Phone armband
- [ ] Wrist strap and a **marker** to draw the alignment line
- [ ] **One dumbbell** light enough for 12 strict reps. Use the same one for both friends and note its weight.
- [ ] Space to walk a few steps (for the "walk" set)
- [ ] Paper and pen for notes: set number plus what went wrong

**Phone settings** (iPhone)
- [ ] Settings → Display & Brightness → **Auto-Lock: Never**. If the phone locks, phyphox can stop recording and the session loses its phone data.
- [ ] **Do Not Disturb** on
- [ ] phyphox experiment with **accelerometer (with g) and gyroscope at the highest rate**, the same one used for `dev2b`

**Laptop check (2 min)**
```bash
cd ~/Documents/repsense && .venv/bin/python -m pytest -q
```
All tests should pass. The Nano's orange LED should blink once a second.

---

## 3. When a friend arrives (~12 min before recording)
**Consent and ID (2 min).** Tell them, in your own words:
- We record motion from a sensor on the wrist and a phone on the upper arm. No video, no name; they're **p2** (or **p3**) in the files.
- It's a college project; the data stays in the project.
- They can stop any time.

Then add their row to `data/sessions.csv` once the session is recorded.

**Mounting (5 min).** Use the arm they normally curl with, the same arm for both sensors.
- **Nano:** USB pointing **toward the elbow**, board flat on the **back of the forearm**. Strap snug. Draw a line across the strap and the skin.
  - **Position A** (session 1): two fingers above the wrist bone.
  - **Position B** (session 2): three fingers above the wrist bone, band turned about 30° around the forearm. USB still toward the elbow.
- **Phone:** in the armband on the **outer side of the upper arm, just above the elbow**, tight. Any orientation is fine.

**Briefing (3 min).** Read this out:

| Term | Meaning |
|---|---|
| Full rep | Arm nearly straight → fully bent → back to nearly straight |
| Count out loud | Count **every lift**, half reps included. The script says how many lifts to expect. |
| "Lift only until your forearm is level" | Raise until the forearm is horizontal, then lower all the way. **Not a full rep.** |
| "Lower only until your forearm is level" | Lift fully, come down only to horizontal, then start the next lift from there. **Not a full rep.** |
| "Let your elbow swing forward" | Let the upper arm drift forward as you lift, then bring it back. **Still a full rep.** |
| "Hold 2–3 s halfway up" | Stop halfway, hold, then finish the lift. **Still a full rep.** |
| Slow / fast | Slow: 3 s up, 3 s down. Fast: as fast as you can with control. |
| Alternating | Take turns with both arms; only the **sensor arm's** lifts count |
| Distractor sets | Walk, re-rack the dumbbell, fiddle with the strap. Not exercise: just move naturally. |
| Exhaustion (always last) | Curl until you can't do one more full rep. Tell me how many full reps you did. |
| Swing gesture (first and last) | Elbow locked, palm facing the thigh: swing the straight arm forward and back like a pendulum, 3 times, fast. **No sideways motion.** |
| Before every set | Let the arm **hang still for a second**, then start |

**Practice (2 min, not recorded).** One swing gesture, one full rep, one "lift only to level," one "lower only to level," one elbow drift. Check they understand "level" means a horizontal forearm.

---

## 4. Recording a session
1. Start the **phyphox recording**. Check the timer is running.
2. Start the receiver, with the session name and participant both matching:
   ```bash
   .venv/bin/python -m src.record --session p2_s1 --participant p2 --script protocols/p2_s1.csv
   ```
3. For every set:
   1. Read the set and its note out loud.
   2. Check their arm is **hanging still**.
   3. Press **Enter**. They do the set and count out loud.
   4. Press **Enter** when they finish.
   5. At **"Actual reps"**:
      - Press **Enter** if they did exactly what the script asked.
      - Otherwise type the number of **full** reps.
      - Type `x` to discard and redo the set (it's still logged as discarded).
   6. ~60 s rest between curl sets. Distractors need little rest.
4. **Exhaustion set:** ask them for their full-rep count and **type it**. Enter would record 0.
5. After the final swing gesture: wait for the session summary, then **stop phyphox**.

**While recording, watch for:**
- A set that went wrong (wrong condition, lost count, interrupted): `x` and redo it.
- Unsure of a count: type your best guess and **write the set number and what happened on paper**. It gets settled later from the raw data.
- A strap or the armband moved: write down the set number.
- The cable pulled tight: give it slack. If it came out, plug it back in and continue with `--resume` (section 9).
- phyphox still recording: glance at it every few sets.

---

## 5. After each session (~5 min)
1. **Export** the phyphox recording (CSV) and AirDrop it to the Mac.
2. Put the export folder at `data/phone/<session>/`, e.g. `data/phone/p2_s1/Accelerometer.csv`.
3. Add a line to `data/sessions.csv`: session, date, participant, dumbbell kg, arm, position A/B, phone position, notes. Include anything from your paper notes and "consent given."
4. **Data-quality check** (allowed: it checks the data, not the counters):
   ```bash
   .venv/bin/python -m src.validate p2_s1 --no-plot
   ```
   It should show no FAIL. The "axes" line should say PASS.

Between session 1 and session 2: a 5-minute break, then re-strap to **position B** with a new line drawn.

---

## 6. Mock trajectory: what to expect when
**The whole day (both friends, 2 sessions each): about 3 hours.** With one session each, about 1 h 45.

| Clock | Who | What |
|---|---|---|
| 0:00 | You | Setup and checks (section 2) |
| 0:20 | p2 | Arrives: consent, mounting (position A), briefing, practice |
| 0:33 | p2 | Start phyphox, start `p2_s1` |
| 0:53 | p2 | `p2_s1` ends: export, `sessions.csv`, validate |
| 0:58 | p2 | Break; re-strap to position B |
| 1:05 | p2 | Start `p2_s2` |
| 1:25 | p2 | `p2_s2` ends: export, `sessions.csv`, validate. p2 done. |
| 1:30 | p3 | Arrives: consent, mounting, briefing, practice |
| 1:43 | p3 | `p3_s1` |
| 2:05 | p3 | Break, re-strap, then `p3_s2` |
| 2:35 | p3 | Export, `sessions.csv`, validate. p3 done. |
| 2:40 | You | Processing and settling true counts (sections 7–8) |

**Inside one session: `p2_s1`, about 20 minutes.** Times are approximate, from the start of recording.

| ~Time | # | Set | Lifts | What they do |
|---|---|---|---|---|
| 0:00 | 1 | Swing gesture | | 3 pendulum swings, elbow locked |
| 0:35 | 2 | Re-rack | | Put the dumbbell down and pick it up, 3 times |
| 1:20 | 3 | Slow | 13 | 3 s up, 3 s down; lift 8 lowers only to level (12 full reps) |
| 3:45 | 4 | Drift | 12 | Lifts 6, 7 and 11: elbow swings forward (12 full reps) |
| 5:25 | 5 | Normal | 7 | All strict |
| 6:55 | 6 | Fast | 11 | All strict, fast |
| 8:20 | 7 | Adjust | | Shake out the arm, fiddle with the strap ~15 s, **don't move the band** |
| 9:00 | 8 | Partial | 16 | Lifts 4 and 8 lift only to level; lifts 6 and 14 lower only to level (12 full reps) |
| 10:50 | 9 | Alternating | 10 | Both arms take turns; count only the sensor arm |
| 12:50 | 10 | Drift | 7 | Lifts 2, 4 and 5: elbow swings forward |
| 14:20 | 11 | Walk | | Walk around ~15 s, arms natural |
| 15:00 | 12 | Partial | 15 | Lifts 2 and 4 lift only to level; lifts 6 and 7 lower only to level (11 full reps). Lift 7 starts from level. |
| 16:50 | 13 | Pause | 11 | Lift 6: hold 2–3 s halfway up; lift 10 lowers only to level (10 full reps) |
| 18:30 | 14 | Exhaustion | ? | To failure; **type** their full-rep count |
| 19:40 | 15 | Swing gesture | | Same 3 pendulum swings |

The other sessions have the same kinds of sets in a different, shuffled order:
- `p2_s2`: swing → slow → partial → re-rack → normal → pause → drift → drift → alternating → adjust → fast → walk → partial → exhaustion → swing
- `p3_s1`: swing → alternating → slow → normal → adjust → walk → drift → partial → fast → pause → drift → partial → re-rack → exhaustion → swing
- `p3_s2`: swing → walk → fast → alternating → partial → slow → partial → normal → drift → adjust → pause → drift → re-rack → exhaustion → swing

### Two people at once (11-03: p3 and p4), about 2 h 10
Alternate people so each gets ~45 min between their own sessions, not 5: the first session's exhaustion set
would otherwise tire the second. All the session-1s go first, so if someone leaves early they still have one.

| Clock | What |
|---|---|
| 0:00 | Both: consent, briefing and practice together (~10 min) |
| 0:10 | Mount p3, position A. `p3_s1b` (p4 watches: it's their practice) |
| 0:32 | Export, validate. Move the Nano and armband to p4, position A |
| 0:40 | `p4_s1` |
| 1:02 | Export, validate. Mount p3, **position B**, new line |
| 1:10 | `p3_s2` |
| 1:32 | Export, validate. Mount p4, **position B**, new line |
| 1:40 | `p4_s2` |
| 2:02 | Export, validate. Done |

---

## 7. Processing (after everyone has left, ~15 min)
For each session:
```bash
.venv/bin/python -m src.validate p2_s1 --no-plot
.venv/bin/python -m src.sync p2_s1
.venv/bin/python -m src.labels p2_s1
```
Sync should show a match of **0.8 or higher** at both gestures. Labels may report problems; note them, don't fix anything.

## 8. Settle the true counts, then evaluate
1. Start from the typed counts.
2. For each set on your paper notes (and the exhaustion sets), **look only at the raw wrist data** (ax humps). Not the counters or the teacher's lift count.
3. Where the raw data clearly shows a different number of full reps, add a row to `data/count_fixes.csv`: session, set_idx, full_reps, source, note, date.
4. Commit the fixes. **Then** run the comparison:
   ```bash
   .venv/bin/python -m src.detector p2_s1 p2_s2 p3_s1 p3_s2 --evaluate
   ```
   This prints, per set, the true count and what each frozen counter said, plus a summary for each counter.

## 9. If something goes wrong
| Symptom | What to do |
|---|---|
| "No data from the board in 5 s" | Check the cable; close the Arduino Serial Monitor if it's open; unplug and replug the Nano |
| Cable pulled out mid-session | Replug, then rerun the same command with `--resume`. It continues from the first unfinished set. |
| Short set: "Only 0.8 s recorded. Redo?" | Press Enter to redo (usually Enter was pressed twice by accident) |
| phyphox stopped or paused | Finish the session, then re-record it under a new name (`p2_s1b`) with a new script, since a paused phone recording has a time gap the sync can't bridge |
| Validate shows "axes FAIL" | Stop and tell me before recording more. Something in the Nano's frame changed. |
| Sync "weak match" at a gesture | The swing probably went sideways or the elbow bent. The session is still usable; note it. |
| Labels: "a strap moved" | Expected if you wrote down a strap problem. Otherwise note it in `sessions.csv`. |
| Friend needs to stop | Press `q`. Everything recorded so far is kept; finish later with `--resume`. |
| Session cut short for good | Press `q` **right away**. A receiver left waiting keeps logging the idle arm (`p3_s1` logged 84 min that way). |
