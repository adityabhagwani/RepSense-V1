// RepSense sensor logger for the Arduino Nano 33 BLE Sense Rev2 (BMI270 IMU).
//
// Streams accelerometer + gyroscope at SAMPLE_HZ over USB serial, one CSV line
// per sample:
//
//   seq,t_us,ax_mg,ay_mg,az_mg,gx_cdps,gy_cdps,gz_cdps
//
//   seq      sample counter since boot; a gap means the host lost lines
//   t_us     micros() when the sample was read (uint32, wraps every ~71.6 min)
//   a*_mg    acceleration in milli-g (library range: +-4 g)
//   g*_cdps  angular rate in centi-degrees/s (library range: +-2000 dps)
//
// The same samples go out over BLE (v2, 11-08): service SERVICE_UUID, one 20-byte notification per
// sample on SAMPLE_UUID, little-endian: seq (uint32), t_us (uint32), a*_mg (int16 x3),
// g*_ddps (int16 x3, deci-degrees/s: 0.1 deg/s steps, so the gyro fits 20 bytes). Advertised as
// "RepSense". USB lines go out only while a computer has the port open, so the board also runs
// from a power bank.
//
// Lines starting with '#' are info, not data. Send 'i' to get the info line.
// The built-in LED toggles once per second while sampling, and blinks fast if
// the IMU fails to start.

#include <ArduinoBLE.h>
#include <Arduino_BMI270_BMM150.h>

const uint32_t SAMPLE_HZ = 50;
const uint32_t PERIOD_US = 1000000UL / SAMPLE_HZ;

uint32_t seq = 0;
uint32_t nextSampleUs;
uint32_t missed = 0;  // sample slots skipped because the loop fell behind

const char *SERVICE_UUID = "a4130001-4d59-462f-8eaa-2423dcd49f52";
const char *SAMPLE_UUID = "a4130002-4d59-462f-8eaa-2423dcd49f52";
const int PACKET_BYTES = 20;
BLEService repService(SERVICE_UUID);
BLECharacteristic sampleChar(SAMPLE_UUID, BLERead | BLENotify, PACKET_BYTES, true);
bool bleOk = false;

int16_t clamp16(long v) {
  return (int16_t)constrain(v, -32768L, 32767L);
}

// The sample as the BLE packet: little-endian, as the nRF52840 stores it.
void packSample(uint8_t *buf, uint32_t s, uint32_t tUs, const int16_t a[3], const int16_t g[3]) {
  memcpy(buf, &s, 4);
  memcpy(buf + 4, &tUs, 4);
  memcpy(buf + 8, a, 6);
  memcpy(buf + 14, g, 6);
}

void printInfo() {
  char buf[128];
  int n = snprintf(buf, sizeof(buf),
                   "# repsense sensor_logger v2 hz=%lu acc_odr=%d gyr_odr=%d missed=%lu ble=%d\n",
                   (unsigned long)SAMPLE_HZ, (int)IMU.accelerationSampleRate(),
                   (int)IMU.gyroscopeSampleRate(), (unsigned long)missed, bleOk ? 1 : 0);
  Serial.write(buf, n);
}

void setup() {
  Serial.begin(115200);
  pinMode(LED_BUILTIN, OUTPUT);

  if (!IMU.begin()) {
    while (true) {
      Serial.println("# error: IMU.begin() failed");
      digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
      delay(100);
    }
  }
  if (BLE.begin()) {
    BLE.setLocalName("RepSense");
    BLE.setAdvertisedService(repService);
    repService.addCharacteristic(sampleChar);
    BLE.addService(repService);
    BLE.advertise();
    bleOk = true;
  } else {
    Serial.println("# warning: BLE.begin() failed; USB only");
  }
  nextSampleUs = micros();
}

void loop() {
  if (bleOk) {
    BLE.poll();  // the BLE stack runs here: connections, subscriptions
  }
  if (Serial.available() && Serial.read() == 'i') {
    printInfo();
  }

  // Never busy-wait: a loop that keeps the CPU starves the core's USB threads
  // (including the one that handles the 1200-baud "reset for upload" request).
  // Sleep while the next sample is far off, then yield for the last stretch
  // so timing stays precise.
  uint32_t now = micros();
  int32_t waitUs = (int32_t)(nextSampleUs - now);
  if (waitUs > 0) {
    if (waitUs > 2000) {
      delay(1);
    } else {
      yield();
    }
    return;
  }

  // If the loop fell a full period behind, skip the lost slots instead of
  // bursting samples back-to-back; the gap stays visible in t_us.
  uint32_t behind = (now - nextSampleUs) / PERIOD_US;
  missed += behind;
  nextSampleUs += (behind + 1) * PERIOD_US;

  float ax, ay, az, gx, gy, gz;
  IMU.readAcceleration(ax, ay, az);
  IMU.readGyroscope(gx, gy, gz);

  // One write per line: the USB CDC driver sends each write as its own transfer.
  if (Serial) {
    char line[96];
    int n = snprintf(line, sizeof(line), "%lu,%lu,%ld,%ld,%ld,%ld,%ld,%ld\n",
                     (unsigned long)seq, (unsigned long)now,
                     lroundf(ax * 1000.0f), lroundf(ay * 1000.0f), lroundf(az * 1000.0f),
                     lroundf(gx * 100.0f), lroundf(gy * 100.0f), lroundf(gz * 100.0f));
    Serial.write(line, n);
  }
  if (bleOk && BLE.connected() && sampleChar.subscribed()) {
    int16_t a[3] = {clamp16(lroundf(ax * 1000.0f)), clamp16(lroundf(ay * 1000.0f)), clamp16(lroundf(az * 1000.0f))};
    int16_t g[3] = {clamp16(lroundf(gx * 10.0f)), clamp16(lroundf(gy * 10.0f)), clamp16(lroundf(gz * 10.0f))};
    uint8_t packet[PACKET_BYTES];
    packSample(packet, seq, now, a, g);
    sampleChar.writeValue(packet, PACKET_BYTES);  // a full queue drops the sample; the seq gap shows it
  }
  seq++;

  if (seq % SAMPLE_HZ == 0) {
    digitalWrite(LED_BUILTIN, !digitalRead(LED_BUILTIN));
  }
}
