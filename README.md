# RepSense

A rep counter for dumbbell curls that uses one sensor on the wrist. It also tells you when to check its count.

**Demo (2 min 27 s, live over Bluetooth):** VIDEO_LINK

**Result on two people that the counter never saw, with the counter locked before their data was scored:**
- 19 of 20 curl sets within 1 rep; 16 of 20 exact.
- 4 false reps in 6 non-curl sets (walk, rerack, adjust the strap).

The full numbers, every error and the limits are in [results.md](results.md).

## How it works

```
wrist IMU, 50 Hz ─► forearm angle from hanging ─► height rule ─► checker ─► 12 s check ─► reps per set
(accelerometer +     (gravity + gyroscope;        (hump ≥ 0.825   (logistic     (a rep needs a
 gyroscope)           twist does not change it)    × the set's     regression,   neighbour within
                                                   median hump)    12 features)  12 s)
                                                                                      │
                                                    error flag (3 set-level signs) ◄──┘ ─► "likely right" / "check this set"
```

1. **Forearm angle.** The board's long axis follows the forearm. Its angle from hanging comes from gravity, and the gyroscope tracks it during fast movement. Twisting the forearm does not change this angle.
2. **Height rule.** Each hump in the angle is a candidate rep. A hump counts if it is at least 0.825 × the set's median hump.
3. **Checker.** A logistic regression scores each hump (height, top angle, rise time, twist and others). Humps with p < 0.05 are removed. This step removes most wiggles.
4. **12 s check.** A rep with no other counted rep within 12 s is removed. This removes single bends, for example picking up a dumbbell.
5. **Sets.** A set ends 12 s after its last lift.
6. **Error flag.** A logistic model on three signs gives the risk that the set's count is wrong. The three signs are: movement that is not a rep, disagreement between the counters, and seconds for each rep. It works on synthetic sets but is not proven on real sets yet (see results.md, section 8).

Each step was locked with a git tag before it was tested on new people: `counter-v2`, `detector-v4`, `min-reps-v1`, `count-error-v1`.

## Hardware

- Arduino Nano 33 BLE Sense Rev2 (BMI270 IMU).
- A strap on the back of the forearm, two fingers above the wrist bone, USB toward the elbow.
- Power: a USB power bank (Bluetooth) or a USB cable to the computer.
- Lab only: a phone in an armband on the upper arm measured the elbow angle for the labels. The counter does not use it.

## Set up

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m pytest -q
```

Flash the firmware (it sends each sample over USB and over Bluetooth):

```bash
arduino-cli core install arduino:mbed_nano
arduino-cli lib install Arduino_BMI270_BMM150 ArduinoBLE
arduino-cli compile --upload -p /dev/cu.usbmodem1101 --fqbn arduino:mbed_nano:nano33ble firmware/sensor_logger
```

The port name changes with the USB socket. Use `arduino-cli board list` to find it.

## Run

| Task | Command |
|---|---|
| Live screen over Bluetooth | `.venv/bin/python -m src.live --ble` |
| Live screen over USB | `.venv/bin/python -m src.live` |
| Live screen on a saved session | `.venv/bin/python -m src.live --replay p7_s1` |
| Check the Bluetooth link (60 s) | `.venv/bin/python -m src.ble --seconds 60` |
| Record a scripted session | `.venv/bin/python -m src.record --session p8_s1 --participant p8 --script protocols/<script>.csv` |
| Score the counters on sessions | `.venv/bin/python -m src.detector p6_s1 p7_s1 --evaluate --detector v4` |
| Find and count sets in whole recordings | `.venv/bin/python -m src.stream p6_s1 p7_s1 --detector v4 --min-reps v1` |

Restart the live screen between takes. Each step gets slower as the recording gets longer (results.md, section 11).

## Files

| Path | What it holds |
|---|---|
| `firmware/sensor_logger/` | Nano firmware: 50 Hz samples over USB and Bluetooth |
| `src/record.py`, `src/ble.py` | Receivers: USB and Bluetooth into one raw CSV per session |
| `src/counter.py` | Forearm angle and the height rule |
| `src/detector.py` | The checker, the 12 s check and the evaluation |
| `src/boxes.py`, `src/stream.py` | Set finding in whole recordings |
| `src/error_model.py`, `src/splice.py` | The error flag and the synthetic sets |
| `src/live.py` | The live screen |
| `src/labels.py`, `src/sync.py`, `src/phone.py` | Lab-only labels from the upper-arm phone |
| `src/make_script.py`, `src/validate.py`, `src/mark.py` | Session scripts, recording checks, hand marks |
| `data/raw/` | Raw recordings, one CSV per session |
| `data/annotations/`, `data/count_fixes.csv`, `data/sessions.csv` | Set marks, typed counts, count fixes, session notes |
| `models/` | The trained checkers and the error flag |
| `PLAN.md` | The dated log: each decision, pre-registration, result and bug |
| `PROTOCOL.md` | How a session is run |
| `FEATURES.md` | The features and how they were selected |

The people are anonymous (p1 to p7). p1 is the author.

## Author

Aditya Bhagwani, UW Electrical and Computer Engineering. Built in approximately 30 hours of work, logged in `PLAN.md`.
