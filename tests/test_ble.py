import struct

from src.ble import PACKET, unpack


def test_a_packet_is_20_bytes_and_unpacks_to_the_usb_fields():
    # seq, t_us, accel in mg, gyro in deci-deg/s (as the firmware sends it)
    data = struct.pack("<IIhhhhhh", 4_000_000_000, 123_456_789, -1000, 25, 980, 12345, -20000, 7)
    assert PACKET.size == len(data) == 20
    # gyro back in centi-deg/s, like the USB line: 1234.5 deg/s -> 123450
    assert unpack(data) == [4_000_000_000, 123_456_789, -1000, 25, 980, 123450, -200000, 70]
