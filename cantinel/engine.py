"""CANtinel detection engine: trains a vehicle profile and fuses the layers.

Fusion rule
-----------
A frame is *flagged* if any layer raises an alert of severity high/medium.
Only frames that are **not** flagged are committed to the stateful layers
(timing anchor, last valid payload).  This cross-layer feedback is what keeps
injected frames from becoming the reference for the next decision and is the
main reason CANtinel has far fewer false positives on legitimate frames than
interval-only detectors.
"""
from __future__ import annotations

import json
import time
from collections import Counter, deque
from pathlib import Path
from typing import Iterable

import numpy as np

from .detectors.payload import PayloadLayer
from .detectors.spec import SpecLayer
from .detectors.timing import TimingLayer
from .frame import Alert, CANFrame

ALL_LAYERS = ("spec", "timing", "payload")


class Profile:
    """Everything CANtinel learns about one vehicle - stored as readable JSON."""

    def __init__(self, spec: SpecLayer, timing: TimingLayer, payload: PayloadLayer, meta: dict):
        self.spec, self.timing, self.payload, self.meta = spec, timing, payload, meta

    @classmethod
    def train(cls, frames: list[CANFrame], validation: list[CANFrame] | None = None,
              max_rounds: int = 3) -> "Profile":
        """Learn a profile from attack-free traffic.

        If ``validation`` (a second, attack-free capture of the same vehicle) is
        given, the profile is self-tuned: every byte rule that raises an alert
        on validation traffic is pruned and timing tolerances are widened to
        cover the observed legitimate frames.  This removes rules that only
        reflect one particular drive (fuel level, odometer, ...).
        """
        prof = cls._fit(frames)
        if validation:
            validation = [f for f in validation if not f.label]
            pruned = Counter()
            for _ in range(max_rounds):
                eng = Engine(prof)
                _, alerts = eng.run(validation)
                bad = [a for a in alerts if a.flags]
                if not bad:
                    break
                for a in bad:
                    if a.layer == "payload" and "byte" in a.extra:
                        prof.payload.relax(a.can_id, a.extra["byte"], a.extra.get("transition"))
                        pruned["payload"] += 1
                    elif a.kind == "TIMING_EARLY":
                        prof.timing.widen(a.can_id, a.extra.get("early_ms", 0) / 1e3)
                        pruned["timing"] += 1
                    elif a.kind == "DLC_MISMATCH":
                        prof.spec.dlc[a.can_id].add(int(a.message.split()[1]))
                        pruned["spec"] += 1
            prof.meta["validation_frames"] = len(validation)
            prof.meta["pruned_alerts"] = dict(pruned)
        return prof

    @classmethod
    def _fit(cls, frames: list[CANFrame]) -> "Profile":
        frames = [f for f in frames if not f.label]  # never learn from labelled attacks
        t0 = time.perf_counter()
        spec = SpecLayer().fit(frames)
        timing = TimingLayer().fit(frames)
        periods = {cid: p["period"] for cid, p in timing.prof.items() if p["periodic"]}
        payload = PayloadLayer().fit(frames, periods)
        dur = frames[-1].timestamp - frames[0].timestamp if frames else 0.0
        # bus-level statistics in 50 ms windows
        ts = np.array([f.timestamp for f in frames])
        counts = np.bincount(((ts - ts[0]) / 0.05).astype(int)) if len(ts) else np.array([0])
        meta = {
            "version": 1,
            "frames": len(frames),
            "duration_s": round(dur, 3),
            "ids": len(spec.dlc),
            "bus_max_frames_50ms": int(counts[:-1].max()) if len(counts) > 1 else int(counts.max()),
            "train_time_s": round(time.perf_counter() - t0, 3),
        }
        return cls(spec, timing, payload, meta)

    def save(self, path: str | Path) -> None:
        doc = {"meta": self.meta, "spec": self.spec.to_dict(), "timing": self.timing.to_dict(),
               "payload": self.payload.to_dict()}
        Path(path).write_text(json.dumps(doc, indent=1))

    @classmethod
    def load(cls, path: str | Path) -> "Profile":
        d = json.loads(Path(path).read_text())
        return cls(SpecLayer.from_dict(d["spec"]), TimingLayer.from_dict(d["timing"]),
                   PayloadLayer.from_dict(d["payload"]), d["meta"])

    def summary(self) -> str:
        per = sum(1 for p in self.timing.prof.values() if p["periodic"])
        roles = Counter(r["role"] for p in self.payload.prof.values() for r in p["bytes"])
        cks = Counter(r["algo"] for p in self.payload.prof.values() for r in p["bytes"] if r["role"] == "checksum")
        return (f"{self.meta['ids']} CAN IDs ({per} periodic) learned from {self.meta['frames']} frames "
                f"/ {self.meta['duration_s']} s\n"
                f"payload byte roles: {dict(roles)}\nchecksums discovered: {dict(cks)}")


class Engine:
    def __init__(self, profile: Profile, layers: Iterable[str] = ALL_LAYERS, auth=None,
                 bus_window: float = 0.05, flood_threshold: int = 10):
        self.profile = profile
        self.layers = set(layers)
        self.spec = profile.spec
        self.timing = profile.timing
        self.payload = profile.payload
        self.auth = auth            # optional SecOC verifier (see secoc.AuthLayer)
        self.bus_window = bus_window
        self.flood_threshold = flood_threshold
        self._win = deque()
        self._flood = deque()          # timestamps of UNKNOWN_ID frames
        self._early: dict[int, float] = {}  # last timing-only-early frame per ID
        self._bus_alarm_until = -1.0
        self.stats = Counter()
        self.reset()

    def reset(self):
        self.timing.reset()
        self.payload.reset()
        self._win.clear()
        self._flood.clear()
        self._early.clear()
        self.stats = Counter()

    def process(self, f: CANFrame) -> list[Alert]:
        alerts: list[Alert] = []
        self.stats["frames"] += 1
        # bus-level context (informational, does not flag individual frames)
        w = self._win
        w.append(f.timestamp)
        while w and w[0] < f.timestamp - self.bus_window:
            w.popleft()
        lim = self.profile.meta.get("bus_max_frames_50ms", 0) * 1.3
        if lim and len(w) > lim and f.timestamp > self._bus_alarm_until:
            alerts.append(Alert(f.timestamp, f.can_id, "BUS_LOAD_ANOMALY", "bus", "info",
                                f"{len(w)} frames in {self.bus_window * 1e3:.0f} ms (max {lim / 1.3:.0f} in training)"))
            self._bus_alarm_until = f.timestamp + 1.0

        if "spec" in self.layers:
            a = self.spec.check(f)
            if a:
                alerts += a
                if a[0].kind == "UNKNOWN_ID":
                    self.stats["flagged"] += 1
                    self._flood.append(f.timestamp)
                    return alerts
        # flood-aware timing: while the bus is saturated by foreign frames,
        # arbitration delays are unbounded and timing evidence is unreliable
        fl = self._flood
        while fl and fl[0] < f.timestamp - self.bus_window:
            fl.popleft()
        flooded = len(fl) >= self.flood_threshold
        if self.auth is not None:
            alerts += self.auth.check(f)
        if "timing" in self.layers:
            ta = self.timing.check(f)
            if ta and flooded:
                for x in ta:
                    x.severity = "info"
                    x.message += " [bus flooded - timing evidence downgraded]"
            alerts += ta
        if "payload" in self.layers:
            alerts += self.payload.check(f)

        bad = [a for a in alerts if a.flags]
        if bad and all(a.kind == "TIMING_EARLY" for a in bad):
            # Phase re-synchronisation: two payload-valid frames exactly one period apart,
            # both "early", with no accepted frame in between, mean the ECU's phase has
            # shifted (e.g. after a long arbitration delay) - not an injection, because an
            # injector running beside the real ECU always leaves accepted frames in between.
            p = self.timing.prof.get(f.can_id)
            prev = self._early.get(f.can_id)
            if p and prev is not None and abs(f.timestamp - prev - p["period"]) <= max(p["tol"], 0.02 * p["period"]):
                for a in bad:
                    a.severity = "info"
                    a.kind = "TIMING_RESYNC"
                    a.message = "ECU phase shift confirmed by two consecutive on-period frames - re-synchronised"
                self.timing.t_ref[f.can_id] = f.timestamp
                self._early.pop(f.can_id, None)
                bad = []
            else:
                self._early[f.can_id] = f.timestamp
        if bad:
            self.stats["flagged"] += 1
        else:
            self._early.pop(f.can_id, None)
            self.timing.commit(f)
            self.payload.commit(f)
            if self.auth is not None:
                self.auth.commit(f)
        return alerts

    def run(self, frames: list[CANFrame]) -> tuple[np.ndarray, list[Alert]]:
        """Process a capture; returns per-frame flags (0/1) and all alerts."""
        self.reset()
        flags = np.zeros(len(frames), dtype=np.int8)
        alerts: list[Alert] = []
        for i, f in enumerate(frames):
            a = self.process(f)
            if a:
                alerts += a
                if any(x.flags for x in a):
                    flags[i] = 1
        return flags, alerts
