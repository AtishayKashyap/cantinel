"""Live monitoring (real CAN interface via python-can) and a classroom demo.

``cantinel live``  - attach to SocketCAN / PCAN / Vector / any python-can bus.
``cantinel demo``  - trains a profile on simulated traffic, then streams an
                     attack capture through a python-can *virtual* bus in
                     real time and prints alerts as they happen.
"""
from __future__ import annotations

import threading
import time
from collections import Counter

from .engine import Engine, Profile
from .frame import CANFrame

RED, YEL, GRN, DIM, BOLD, RST = "\033[31m", "\033[33m", "\033[32m", "\033[2m", "\033[1m", "\033[0m"


class Console:
    """Aggregates alerts so a flood does not scroll the terminal away."""

    def __init__(self, every: float = 1.0, color: bool = True):
        self.every = every
        self.color = color
        self.last_print = None
        self.window = Counter()
        self.example = {}
        self.win_flagged = 0
        self.frames = 0
        self.flagged = 0
        self.total_alerts = Counter()

    def c(self, code, s):
        return f"{code}{s}{RST}" if self.color else s

    def feed(self, f: CANFrame, alerts):
        self.frames += 1
        bad = [a for a in alerts if a.flags]
        if bad:
            self.flagged += 1
            self.win_flagged += 1
        for a in alerts:
            key = (a.can_id, a.kind)
            self.window[key] += 1
            self.total_alerts[a.kind] += 1
            self.example.setdefault(key, a)
        if self.last_print is None:
            self.last_print = f.timestamp
        if f.timestamp - self.last_print >= self.every:
            self.flush(f.timestamp)

    def flush(self, t):
        if self.win_flagged:
            status = self.c(RED, f"UNDER ATTACK ({self.win_flagged:,} frames flagged)")
        elif self.window:
            status = self.c(YEL, "normal (low-severity notices only)")
        else:
            status = self.c(GRN, "normal")
        print(self.c(DIM, f"[t={t:7.2f}s] frames={self.frames:>8,} flagged={self.flagged:>7,} ") + status)
        rank = {"high": 0, "medium": 1, "low": 2, "info": 3}
        for (cid, kind), n in sorted(self.window.items(),
                                     key=lambda kv: (rank[self.example[kv[0]].severity], -kv[1]))[:4]:
            a = self.example[(cid, kind)]
            col = RED if a.severity == "high" else YEL if a.severity == "medium" else DIM
            print("    " + self.c(col, f"{kind:<16}") + f" 0x{cid:03X} x{n:<5} {a.message}")
        self.window.clear()
        self.example.clear()
        self.win_flagged = 0
        self.last_print = t


def monitor_bus(profile: Profile, interface: str, channel: str, bitrate: int | None = None,
                alerts_path: str | None = None, color: bool = True):
    import json

    import can

    kw = {"interface": interface, "channel": channel}
    if bitrate:
        kw["bitrate"] = bitrate
    eng = Engine(profile)
    con = Console(color=color)
    out = open(alerts_path, "a") if alerts_path else None
    print(f"CANtinel monitoring {interface}:{channel} - Ctrl+C to stop")
    with can.Bus(**kw) as bus:
        try:
            for msg in bus:
                if msg.is_error_frame or msg.is_remote_frame:
                    continue
                f = CANFrame(msg.timestamp, msg.arbitration_id, msg.dlc, bytes(msg.data))
                alerts = eng.process(f)
                con.feed(f, alerts)
                if out:
                    for a in alerts:
                        out.write(json.dumps(a.to_dict()) + "\n")
        except KeyboardInterrupt:
            pass
    if out:
        out.close()
    print(f"done - {con.frames:,} frames, {con.flagged:,} flagged, alerts: {dict(con.total_alerts)}")


def demo(attack: str = "spoof_gear", duration: float = 40.0, speed: float = 1.0, seed: int = 1,
         color: bool = True):
    from .simulator import simulate

    print("1) learning the vehicle (simulated 180 s drive + 60 s validation drive)...")
    prof = Profile.train(simulate(180, None, seed=100 + seed), simulate(60, None, seed=150 + seed))
    print("   " + prof.summary().replace("\n", "\n   "))
    frames = simulate(duration, None if attack == "none" else attack, seed=200 + seed, first_attack=5.0)
    atk = [f for f in frames if f.label]
    if atk:
        print(f"2) streaming a {duration:.0f} s capture; first '{attack}' burst starts at t={atk[0].timestamp:.1f}s")
    eng = Engine(prof)
    con = Console(color=color)
    try:
        import can
        tx = can.Bus(interface="virtual", channel="cantinel-demo", preserve_timestamps=True)
        rx = can.Bus(interface="virtual", channel="cantinel-demo", preserve_timestamps=True)
    except Exception:  # python-can missing -> stream in-process
        tx = rx = None

    def producer():
        start = time.time()
        for f in frames:
            lag = f.timestamp / speed - (time.time() - start)
            if lag > 0.005:
                time.sleep(lag)
            tx.send(can.Message(timestamp=f.timestamp, arbitration_id=f.can_id, data=f.data,
                                is_extended_id=False))
        tx.send(can.Message(arbitration_id=0x7FF, data=b"EOF", is_extended_id=False, timestamp=-1))

    if tx is not None:
        th = threading.Thread(target=producer, daemon=True)
        th.start()
        while True:
            msg = rx.recv(timeout=5)
            if msg is None or msg.timestamp == -1:
                break
            f = CANFrame(msg.timestamp, msg.arbitration_id, msg.dlc, bytes(msg.data))
            con.feed(f, eng.process(f))
        tx.shutdown()
        rx.shutdown()
    else:
        start = time.time()
        for f in frames:
            lag = f.timestamp / speed - (time.time() - start)
            if lag > 0.005:
                time.sleep(lag)
            con.feed(f, eng.process(f))
    con.flush(frames[-1].timestamp)
    n_atk = len(atk)
    caught = sum(1 for f, fl in zip(frames, Engine(prof).run(frames)[0]) if f.label and fl)
    print(con.c(BOLD, f"\nsummary: {con.frames:,} frames, {n_atk:,} injected, {caught:,} detected "
                      f"({(caught / n_atk * 100) if n_atk else 100:.1f}%), alert kinds {dict(con.total_alerts)}"))
