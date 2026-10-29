"""Load a phyphox export (the phone strapped to the upper arm) in the Nano's units.

phyphox writes one folder per recording:
  Accelerometer.csv   Time (s), X/Y/Z in m/s^2, gravity included
  Gyroscope.csv       Time (s), X/Y/Z in rad/s
  meta/time.csv       START / PAUSE events with the phone's Unix time
Accelerometer and gyroscope have their own timestamps, so they stay separate
frames here; sync.py interpolates both onto the Nano's sample times.

Convention: copy the export folder to data/phone/<session>/.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from src.dataio import ACCEL, DATA_DIR, GYRO

STANDARD_GRAVITY = 9.80665  # m/s^2 per g


def phone_dir(session: str, data_dir: Path = DATA_DIR) -> Path:
    return data_dir / "phone" / session


def load_phyphox(folder: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(accel, gyro) frames: unix_t on the phone's clock, then ax..az in g or gx..gz in deg/s.

    unix_t = START system time + "Time (s)", which only holds for an unbroken run.
    phyphox's clock stops while paused, so after a resume every sample would land
    early by the length of the pause. A second START is therefore an error.
    """
    folder = Path(folder)
    events = pd.read_csv(folder / "meta" / "time.csv")
    starts = events.loc[events["event"] == "START", "system time"]
    if len(starts) > 1:
        raise ValueError(
            f"{folder}: {len(starts)} START events in meta/time.csv. "
            "The recording was paused and has a gap: re-record it without pausing."
        )
    if len(starts) == 0:
        raise ValueError(f"{folder}: no START event in meta/time.csv, so no clock anchor.")
    start = float(starts.iloc[0])

    accel = _read_sensor(folder / "Accelerometer.csv", start, ACCEL, "m/s^2", 1 / STANDARD_GRAVITY)
    gyro = _read_sensor(folder / "Gyroscope.csv", start, GYRO, "rad/s", 180 / np.pi)
    return accel, gyro


def _read_sensor(path: Path, start: float, names: list[str], unit: str, scale: float) -> pd.DataFrame:
    """One phyphox sensor file -> unix_t plus three axes multiplied by `scale`.

    Columns are picked by full header, unit included, so a file in other units
    fails here instead of being scaled wrong.
    """
    columns = [f"{axis} ({unit})" for axis in "XYZ"]
    raw = pd.read_csv(path, usecols=["Time (s)", *columns])
    df = pd.DataFrame({"unix_t": start + raw["Time (s)"].to_numpy(float)})
    for name, col in zip(names, columns):
        df[name] = raw[col].to_numpy(float) * scale
    return df
