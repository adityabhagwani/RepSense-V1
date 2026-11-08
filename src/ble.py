"""BLE receiver for the RepSense logger: the same samples as USB, over Bluetooth, into the same raw file.

The firmware (sensor_logger v2) notifies one 20-byte packet per sample on SAMPLE_UUID: seq and t_us
(uint32), acceleration in mg and rotation in deci-degrees/s (int16 each, little-endian). The gyro
travels at 0.1 deg/s steps, not USB's 0.01, to fit the default 20-byte packet (gyro noise is ~0.1
deg/s); it is written back in centi-degrees/s so the raw file keeps the USB format and every reader
works unchanged. host_t is the Mac clock on arrival: BLE delivers in bursts, so it jitters by about
a connection interval (15-30 ms); the board's t_us does not.

    python -m src.ble --seconds 60    # link check: record a minute, report rate and lost samples
"""

import argparse
import asyncio
import csv
import struct
import threading
import time
from datetime import datetime

import numpy as np

from src.dataio import RAW_COLUMNS, load_session, raw_path

NAME = "RepSense"
SERVICE_UUID = "a4130001-4d59-462f-8eaa-2423dcd49f52"
SAMPLE_UUID = "a4130002-4d59-462f-8eaa-2423dcd49f52"
PACKET = struct.Struct("<IIhhhhhh")  # seq, t_us, ax, ay, az (mg), gx, gy, gz (deci-deg/s)
FLUSH_EVERY = 25                     # samples, as record.Streamer
SCAN_S = 15.0


def unpack(data: bytes) -> list[int]:
    """One packet -> the 8 board fields in the USB line's units (gyro back to centi-deg/s)."""
    seq, t_us, ax, ay, az, gx, gy, gz = PACKET.unpack(data)
    return [seq, t_us, ax, ay, az, gx * 10, gy * 10, gz * 10]


class BleStreamer(threading.Thread):
    """record.Streamer's interface over BLE: appends host_t and the board fields of every sample to
    raw_file; lock, got_data and n_samples work the same way."""

    def __init__(self, raw_file, name: str = NAME, log=print):
        super().__init__(daemon=True)
        self.raw_file, self.name, self.log = raw_file, name, log
        self.writer = csv.writer(raw_file)
        self.lock = threading.Lock()
        self.got_data = threading.Event()
        self.n_samples = 0
        self.running = True

    def run(self):
        asyncio.run(self._stream())

    async def _stream(self):
        from bleak import BleakClient, BleakScanner

        device = await BleakScanner.find_device_by_name(self.name, timeout=SCAN_S)
        if device is None:
            self.log(f"No BLE device named {self.name} in {SCAN_S:.0f} s. Is the board powered and flashed (v2)?")
            return
        async with BleakClient(device) as client:
            await client.start_notify(SAMPLE_UUID, self._on_sample)
            while self.running and client.is_connected:
                await asyncio.sleep(0.2)
            if not client.is_connected:
                self.log("BLE link lost.")

    def _on_sample(self, _sender, data: bytearray):
        host_t = time.time()
        with self.lock:
            self.writer.writerow([f"{host_t:.4f}", *unpack(bytes(data))])
            self.n_samples += 1
            if self.n_samples % FLUSH_EVERY == 0:
                self.raw_file.flush()
        self.got_data.set()

    def stop(self):
        self.running = False
        self.join(timeout=3)


def link_report(session: str) -> str:
    """Rate, lost samples (seq gaps) and arrival jitter of a BLE recording."""
    df = load_session(session)
    seq = df["seq"].to_numpy(np.int64)
    lost = int(np.clip(np.diff(seq) - 1, 0, None).sum())
    seconds = df["t"].iloc[-1] - df["t"].iloc[0]
    late = np.diff(df["host_t"].to_numpy()) - np.diff(df["t"].to_numpy())
    return (f"{len(df)} samples in {seconds:.1f} s ({(len(df) - 1) / seconds:.2f} Hz), lost {lost} "
            f"({lost / (len(df) + lost):.2%}); arrival jitter 95th percentile {np.percentile(np.abs(late), 95) * 1000:.0f} ms")


def main():
    parser = argparse.ArgumentParser(description="Record from the RepSense logger over BLE and check the link.")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--session", help="default ble_test_<date>_<time>")
    args = parser.parse_args()
    session = args.session or datetime.now().strftime("ble_test_%Y%m%d_%H%M%S")
    path = raw_path(session)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as raw_file:
        csv.writer(raw_file).writerow(RAW_COLUMNS)
        streamer = BleStreamer(raw_file)
        streamer.start()
        if not streamer.got_data.wait(timeout=SCAN_S + 10):
            streamer.stop()
            raise SystemExit("No BLE data. Is the board powered, flashed with sensor_logger v2, and not connected elsewhere?")
        print(f"Recording {args.seconds:.0f} s over BLE to {path}")
        time.sleep(args.seconds)
        streamer.stop()
    print(link_report(session))


if __name__ == "__main__":
    main()
