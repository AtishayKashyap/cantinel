"""Synthetic passenger-car model used to generate realistic CAN traffic.

The model has two parts:

* ``generate_trace`` - a physically plausible driving scenario (idle, launch,
  gear shifts, cruising, braking, parking manoeuvres in reverse) sampled every
  10 ms.
* ``MESSAGE_CATALOG`` - 24 periodic / event-driven CAN messages sent by eight
  ECUs, with signal encodings, rolling counters and checksums similar to what
  is found in production vehicles (IDs 0x316 / 0x43F mirror the RPM and gear
  frames of the public HCRL Car-Hacking dataset).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .checksums import compute

DT = 0.01  # trace resolution (s)

# gear selector encoding (high nibble of the gear byte)
SEL_P, SEL_R, SEL_N, SEL_D = 0, 1, 2, 3
GEAR_SPEEDS = [0, 18, 35, 55, 75, 95]          # km/h up-shift points
GEAR_RATIO = [0, 110, 65, 45, 34, 28, 23]       # rpm per km/h per gear


@dataclass
class DrivingTrace:
    t: np.ndarray
    speed: np.ndarray      # km/h
    rpm: np.ndarray
    selector: np.ndarray   # SEL_*
    gear: np.ndarray       # 0..6
    throttle: np.ndarray   # %
    brake: np.ndarray      # bar
    steer: np.ndarray      # deg
    yaw: np.ndarray        # deg/s
    ax: np.ndarray         # m/s^2
    coolant: np.ndarray    # deg C
    fuel: np.ndarray       # %
    odo: np.ndarray        # km
    lights: np.ndarray     # bitfield

    def state(self, t: float) -> dict:
        i = min(max(int(t / DT), 0), len(self.t) - 1)
        return {k: getattr(self, k)[i] for k in (
            "speed", "rpm", "selector", "gear", "throttle", "brake", "steer",
            "yaw", "ax", "coolant", "fuel", "odo", "lights")}


def generate_trace(duration: float, rng: np.random.Generator, odo0: float | None = None,
                   fuel0: float | None = None) -> DrivingTrace:
    n = int(duration / DT) + 2
    speed = np.zeros(n)
    sel = np.zeros(n, dtype=int)
    throttle = np.zeros(n)
    brake = np.zeros(n)

    k = 0
    v = 0.0
    cur_sel = SEL_P
    # start parked
    dur = int(rng.uniform(1.5, 3.0) / DT)
    sel[k:k + dur] = SEL_P
    k += dur
    while k < n:
        if v < 0.5:
            seg = rng.choice(["idle", "accelerate", "reverse"], p=[0.25, 0.6, 0.15])
        else:
            choices = ["cruise", "brake"] + (["accelerate"] if v < 115 else [])
            probs = [0.35, 0.4, 0.25] if v < 115 else [0.45, 0.55]
            seg = rng.choice(choices, p=probs)

        if seg == "idle":
            m = int(rng.uniform(2, 6) / DT)
            cur_sel = SEL_P if rng.random() < 0.3 else SEL_D
            for _ in range(m):
                if k >= n:
                    break
                speed[k] = 0
                sel[k] = cur_sel
                k += 1
            v = 0.0
        elif seg == "accelerate":
            cur_sel = SEL_D
            target = rng.uniform(v + 10, min(130, v + 60))
            a = rng.uniform(1.0, 3.0) * 3.6 * DT  # km/h per step
            thr = rng.uniform(20, 70)
            while v < target and k < n:
                v = min(target, v + a * (1 - 0.4 * v / 140))
                speed[k] = v
                sel[k] = cur_sel
                throttle[k] = thr + rng.normal(0, 2)
                k += 1
        elif seg == "cruise":
            cur_sel = SEL_D
            m = int(rng.uniform(3, 12) / DT)
            for _ in range(m):
                if k >= n:
                    break
                v = max(5.0, v + rng.normal(0, 0.03))
                speed[k] = v
                sel[k] = cur_sel
                throttle[k] = 12 + 0.1 * v + rng.normal(0, 1.5)
                k += 1
        elif seg == "brake":
            cur_sel = SEL_D
            target = 0.0 if rng.random() < 0.55 else rng.uniform(10, max(11, v - 10))
            d = rng.uniform(1.5, 4.5)
            step = d * 3.6 * DT
            while v > target and k < n:
                v = max(target, v - step)
                speed[k] = v
                sel[k] = cur_sel
                brake[k] = d * 11 + rng.normal(0, 1.0)
                k += 1
            v = max(v, 0.0)
        elif seg == "reverse":
            # P -> R, creep backwards, stop, back to P
            top = rng.uniform(4, 8)
            prof = np.concatenate([
                np.zeros(int(0.5 / DT)),
                np.linspace(0, top, int(2 / DT)),
                np.full(int(rng.uniform(1.5, 3) / DT), top),
                np.linspace(top, 0, int(1.5 / DT)),
            ])
            for x in prof:
                if k >= n:
                    break
                speed[k] = x
                sel[k] = SEL_R
                throttle[k] = 8 if x > 0 else 0
                brake[k] = 15 if x == 0 else 0
                k += 1
            m = int(1.0 / DT)
            sel[k:k + m] = SEL_P
            k += m
            v = 0.0

    speed = np.clip(speed[:n], 0, 140)
    sel = sel[:n]
    throttle = np.clip(throttle[:n], 0, 100)
    brake = np.clip(brake[:n], 0, 80)

    # automatic gearbox
    gear = np.zeros(n, dtype=int)
    g = 1
    for i in range(n):
        if sel[i] == SEL_D:
            while g < 6 and speed[i] > GEAR_SPEEDS[g] + 3:
                g += 1
            while g > 1 and speed[i] < GEAR_SPEEDS[g - 1] - 3:
                g -= 1
            gear[i] = g
        elif sel[i] == SEL_R:
            gear[i] = 1
            g = 1
        else:
            gear[i] = 0
            g = 1

    ratio = np.array(GEAR_RATIO)[gear]
    rpm = np.where(gear == 0, 750.0, 800.0 + speed * ratio) + throttle * 6
    rpm += rng.normal(0, 15, n)
    rpm = np.clip(rpm, 650, 6200)

    # steering: smoothed random walk, narrower at speed
    steer = np.zeros(n)
    s = 0.0
    for i in range(n):
        lim = 400 if speed[i] < 15 else max(20, 200 - speed[i] * 1.4)
        s = 0.995 * s + rng.normal(0, 0.8 if speed[i] < 15 else 0.25)
        s = float(np.clip(s, -lim, lim))
        steer[i] = s
    yaw = (speed / 3.6) * np.tan(np.radians(steer / 15.0)) / 2.7 * 57.3
    ax = np.gradient(speed / 3.6, DT)
    ax = np.convolve(ax, np.ones(10) / 10, mode="same")

    t = np.arange(n) * DT
    coolant = 90 - 28 * np.exp(-t / 150) + rng.normal(0, 0.2, n)
    fuel = (fuel0 if fuel0 is not None else rng.uniform(30, 90)) - t * 0.002
    odo = (odo0 if odo0 is not None else rng.uniform(10_000, 90_000)) + np.cumsum(speed / 3.6 * DT) / 1000
    lights = np.zeros(n, dtype=int)
    cur = 0
    for i in range(0, n, 100):
        if rng.random() < 0.05:
            cur ^= 1 << int(rng.integers(0, 4))
        lights[i:i + 100] = cur

    return DrivingTrace(t, speed, rpm, sel, gear, throttle, brake, steer, yaw,
                        ax, coolant, fuel, odo, lights)


# --------------------------------------------------------------------------
# Message catalogue
# --------------------------------------------------------------------------

def _u16(v: float) -> bytes:
    x = int(np.clip(round(v), 0, 0xFFFF))
    return bytes((x & 0xFF, x >> 8))


def _s16(v: float) -> bytes:
    x = int(np.clip(round(v), -32768, 32767)) & 0xFFFF
    return bytes((x & 0xFF, x >> 8))


def _b(v: float) -> int:
    return int(np.clip(round(v), 0, 255))


def _with_ctr_crc(body: list[int], ctr: int, ctr_mode: str, crc: str | None) -> bytes:
    """body has 8 ints; byte 6 carries the counter, byte 7 the checksum."""
    if ctr_mode == "nibble":
        body[6] = (body[6] & 0xF0) | (ctr & 0x0F)
    elif ctr_mode == "byte":
        body[6] = ctr & 0xFF
    data = bytes(body)
    if crc:
        data = data[:7] + bytes((compute(crc, data, 7),))
    return data


@dataclass
class MessageSpec:
    can_id: int
    name: str
    ecu: str
    period: float | None            # None -> event driven
    dlc: int
    encode: Callable[[dict, int], bytes]
    counter: str | None = None      # nibble | byte | None
    checksum: str | None = None     # xor | sum | crc8 | None


def _wheel(s, c):
    v = s["speed"] * 100
    d = s["steer"] * 0.02
    return _u16(v + d) + _u16(v - d) + _u16(v + d * 0.8) + _u16(v - d * 0.8)


def _brake(s, c):
    body = list(_u16(s["brake"] * 10)) + [1 if s["brake"] > 2 else 0, 0, 0x00, 0x00, 0x10, 0]
    return _with_ctr_crc(body, c, "nibble", "crc8")


def _torque(s, c):
    tq = s["throttle"] * 3.2 + 20
    body = list(_u16(tq * 10)) + [_b(s["throttle"] * 2), 0x7D, 0x00, 0x00, 0, 0]
    return _with_ctr_crc(body, c, "byte", "xor")


def _esc(s, c):
    body = list(_u16(s["yaw"] * 100 + 32768)) + list(_u16(s["ax"] * 100 + 32768)) + [0x00, 0x00, 0x20, 0]
    return _with_ctr_crc(body, c, "nibble", "sum")


def _eng_status(s, c):
    return bytes((0x0B, 0x21, _b(s["throttle"]), _b(s["coolant"] + 40), 0, 0, 0, 0))


def _steer(s, c):
    rate = 0  # filled from angle change in a stateless way (coarse)
    body = list(_s16(s["steer"] * 10)) + [rate, 0x00, 0x07, 0x00, 0x00, 0]
    return _with_ctr_crc(body, c, "nibble", "xor")


def _accel_pedal(s, c):
    return bytes((_b(s["throttle"] * 2.5), 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, (c & 0x0F) << 4))


def _eng2(s, c):
    return bytes((0x05, 0x20 if s["rpm"] < 3000 else 0x21, 0x13, 0x00, 0x00, 0x00, 0x00, 0x00))


def _tcu(s, c):
    body = [(int(s["selector"]) << 4) | int(s["gear"]), 0x00, _b(s["rpm"] / 40), 0x00, 0x00, 0x00, 0x00, 0]
    return _with_ctr_crc(body, c, "nibble", "crc8")


def _rpm(s, c):  # mirrors HCRL 0x316 layout: RPM*4 in bytes 2-3 (LE)
    return bytes((0x05, 0x21)) + _u16(s["rpm"] * 4) + bytes((0x0F, _b(s["throttle"] * 2), 0x00, 0x7F))


def _eng_temp(s, c):
    return bytes((0x40, _b(s["coolant"] + 48), 0x7F, 0x14, 0x11, 0x20, 0x00, 0x14))


def _eps(s, c):
    body = list(_s16(s["steer"] * 3)) + [0x00, 0x80, 0x00, 0x00, 0x00, 0]
    return _with_ctr_crc(body, c, "nibble", "sum")


def _gear(s, c):  # mirrors HCRL 0x43F gear frame
    return bytes(((int(s["selector"]) << 4) | int(s["gear"]), 0x45, 0x60, 0xFF, 0x6B, 0x00, 0x00, 0x00))


def _tcu2(s, c):
    return bytes((0xFF, 0x00, 0x00, 0x00, 0xFF, 0x00, 0x00, 0x00))


def _vspeed(s, c):
    body = list(_u16(s["speed"] * 100)) + list(_u16(s["speed"] * 100)) + [0x00, 0x00, 0x00, 0]
    return _with_ctr_crc(body, c, "nibble", "crc8")


def _bms(s, c):
    return bytes((0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00))


def _eng_misc(s, c):
    return bytes((0xD8, 0x00, 0x00, 0x8B, 0x00, 0x00, 0x00, 0x00))


def _fuel(s, c):
    return bytes((_b(s["fuel"] * 2.5), 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00))


def _lights(s, c):
    return bytes((int(s["lights"]) & 0x0F, 0x00, 0x00, 0x00))


def _body(s, c):
    return bytes((0x00, 0x40, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00))


def _climate(s, c):
    return bytes((0x00, 0x16, 0x00, 0x00, 0x2C, 0x00, 0x00, 0x00))


def _odo(s, c):
    x = int(s["odo"] * 10)
    return bytes((x & 0xFF, (x >> 8) & 0xFF, (x >> 16) & 0xFF, (x >> 24) & 0xFF, 0, 0, 0, 0))


def _tpms(s, c):
    return bytes((0x23, 0x23, 0x24, 0x23, 0x00, 0x00, 0x00, 0x00))


def _airbag(s, c):
    body = [0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0]
    return _with_ctr_crc(body, c, "nibble", "xor")


MESSAGE_CATALOG: list[MessageSpec] = [
    MessageSpec(0x0A0, "WHEEL_SPEEDS", "ABS", 0.010, 8, _wheel),
    MessageSpec(0x0B0, "BRAKE", "ABS", 0.020, 8, _brake, "nibble", "crc8"),
    MessageSpec(0x110, "ENGINE_TORQUE", "ECM", 0.010, 8, _torque, "byte", "xor"),
    MessageSpec(0x153, "ESC", "ABS", 0.010, 8, _esc, "nibble", "sum"),
    MessageSpec(0x18F, "ENGINE_STATUS", "ECM", 0.010, 8, _eng_status),
    MessageSpec(0x1F1, "STEERING_ANGLE", "SAS", 0.010, 8, _steer, "nibble", "xor"),
    MessageSpec(0x220, "ACCEL_PEDAL", "ECM", 0.010, 8, _accel_pedal, "nibble-hi", None),
    MessageSpec(0x260, "ENGINE_2", "ECM", 0.010, 8, _eng2),
    MessageSpec(0x2A0, "TCU_STATUS", "TCU", 0.010, 8, _tcu, "nibble", "crc8"),
    MessageSpec(0x316, "ENGINE_RPM", "ECM", 0.010, 8, _rpm),
    MessageSpec(0x329, "ENGINE_TEMP", "ECM", 0.010, 8, _eng_temp),
    MessageSpec(0x370, "EPS", "EPS", 0.020, 8, _eps, "nibble", "sum"),
    MessageSpec(0x43F, "GEAR", "TCU", 0.010, 8, _gear),
    MessageSpec(0x440, "TCU_2", "TCU", 0.020, 8, _tcu2),
    MessageSpec(0x4B0, "VEHICLE_SPEED", "ABS", 0.020, 8, _vspeed, "nibble", "crc8"),
    MessageSpec(0x4F0, "BMS", "BCM", 0.050, 8, _bms),
    MessageSpec(0x545, "ENGINE_MISC", "ECM", 0.010, 8, _eng_misc),
    MessageSpec(0x580, "FUEL", "BCM", 0.100, 8, _fuel),
    MessageSpec(0x5A0, "LIGHTS", "BCM", None, 4, _lights),
    MessageSpec(0x5F0, "BODY", "BCM", 0.200, 8, _body),
    MessageSpec(0x690, "CLIMATE", "BCM", 0.100, 8, _climate),
    MessageSpec(0x6B0, "CLUSTER_ODO", "IC", 1.000, 8, _odo),
    MessageSpec(0x350, "TPMS", "BCM", 0.200, 8, _tpms),
    MessageSpec(0x3E9, "AIRBAG", "ACU", 0.100, 8, _airbag, "nibble", "xor"),
]

CATALOG_BY_ID = {m.can_id: m for m in MESSAGE_CATALOG}
