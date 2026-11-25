"""Layer 3 - payload semantics learned without a DBC file.

For every CAN ID the training phase classifies each payload byte into one of
these roles (first match wins):

``checksum``  byte equals XOR / SUM / CRC-8 (SAE J1850) of the other bytes
``counter``   full byte or nibble that increments by one per frame
``enum``      at most 16 distinct values (gear selector, flags, constants)
``range``     numeric byte; learned [min, max] and max change per period
``free``      no usable constraint (e.g. low byte of a 16-bit signal)

At run time each frame is compared with the *last accepted* frame of the same
ID, so injected frames never become the reference (cross-layer fusion).
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..checksums import CHECKSUMS, compute
from ..frame import Alert, CANFrame

ENUM_MAX = 16
RATE_MAX = 48


def _nib(v: int, part: str) -> int:
    return v if part == "byte" else (v & 0x0F if part == "lo" else v >> 4)


class PayloadLayer:
    name = "payload"

    def __init__(self, min_frames: int = 100):
        self.min_frames = min_frames
        self.prof: dict[int, dict] = {}
        self.last: dict[int, tuple[float, bytes]] = {}

    # ---------------------------------------------------------------- train
    def fit(self, frames: list[CANFrame], periods: dict[int, float] | None = None) -> "PayloadLayer":
        by_id = defaultdict(list)
        for f in frames:
            by_id[f.can_id].append(f)
        for cid, fl in by_id.items():
            dlcs = {f.dlc for f in fl}
            if len(fl) < self.min_frames or len(dlcs) != 1 or fl[0].dlc == 0:
                continue
            dlc = fl[0].dlc
            arr = np.frombuffer(b"".join(f.data for f in fl), dtype=np.uint8).reshape(-1, dlc).astype(int)
            roles: list[dict] = [None] * dlc
            # 1) checksum
            sample = [f.data for f in fl[:3000]]
            for pos in (dlc - 1, 0):
                if roles[pos] is not None or len(set(arr[:, pos])) < 3:
                    continue
                for algo in CHECKSUMS:
                    ok = np.mean([compute(algo, d, pos) == d[pos] for d in sample])
                    if ok >= 0.999:
                        roles[pos] = {"role": "checksum", "algo": algo}
                        break
            # 2) counters
            for pos in range(dlc):
                if roles[pos] is not None:
                    continue
                col = arr[:, pos]
                for part, mod in (("byte", 256), ("lo", 16), ("hi", 16)):
                    v = col if part == "byte" else (col & 0x0F if part == "lo" else col >> 4)
                    if len(set(v)) < 4:
                        continue
                    diffs = (v[1:] - v[:-1]) % mod
                    if np.mean(diffs == 1) >= 0.95:
                        other = None
                        if part != "byte":
                            o = col >> 4 if part == "lo" else col & 0x0F
                            other = sorted(int(x) for x in set(o))
                        roles[pos] = {"role": "counter", "part": part, "mod": mod, "other": other}
                        break
            # 3) enum / range / free
            for pos in range(dlc):
                if roles[pos] is not None:
                    continue
                col = arr[:, pos]
                uniq = np.unique(col)
                prev_r = roles[pos - 1] if pos else None
                carry = prev_r is not None and prev_r["role"] in ("free", "range")
                if len(uniq) <= ENUM_MAX and carry:
                    # upper byte of a little-endian multi-byte signal (odometer, speed, ...):
                    # it only moves when the lower byte wraps -> numeric, allow neighbours
                    roles[pos] = {"role": "range", "min": int(uniq.min()), "max": int(uniq.max()),
                                  "signed": False, "max_delta": None, "soft": True}
                    continue
                if len(uniq) <= ENUM_MAX:
                    # a small but *contiguous* set (e.g. high byte of a signed 16-bit
                    # signal: 0xFF, 0x00, 0x01) is numeric, not categorical
                    sv = np.where(uniq >= 128, uniq - 256, uniq)
                    signed = (sv.max() - sv.min()) < (uniq.max() - uniq.min())
                    vals = np.sort(sv) if signed else uniq
                    span = int(vals.max() - vals.min())
                    if len(uniq) >= 2 and span <= 2 * len(uniq):
                        roles[pos] = {"role": "range", "min": int(vals.min()), "max": int(vals.max()),
                                      "signed": bool(signed), "max_delta": None, "soft": True}
                    else:
                        # learn the state machine of categorical bytes (e.g. P->R, D3->D4)
                        # (symmetrised: if a->b is legal, b->a is assumed legal too)
                        tr = {(int(a), int(b)) for a, b in zip(col[:-1], col[1:]) if a != b}
                        tr |= {(b, a) for a, b in tr}
                        roles[pos] = {"role": "enum", "values": [int(x) for x in uniq],
                                      "transitions": sorted([list(t) for t in tr])}
                    continue
                lo, hi = int(col.min()), int(col.max())
                if hi - lo >= 250:
                    roles[pos] = {"role": "free"}
                    continue
                md = int(np.max(np.abs(np.diff(col)))) if len(col) > 1 else 255
                roles[pos] = {"role": "range", "min": lo, "max": hi,
                              "max_delta": md if md <= RATE_MAX else None}
            self.prof[cid] = {"dlc": dlc, "bytes": roles,
                              "period": (periods or {}).get(cid)}
        return self

    def reset(self):
        self.last.clear()

    # --------------------------------------------------------------- detect
    def check(self, f: CANFrame) -> list[Alert]:
        p = self.prof.get(f.can_id)
        if p is None or f.dlc != p["dlc"]:
            return []
        out = []
        prev = self.last.get(f.can_id)
        steps = None
        if prev is not None:
            per = p.get("period")
            steps = max(1, round((f.timestamp - prev[0]) / per)) if per else None
        d = f.data
        for pos, r in enumerate(p["bytes"]):
            role = r["role"]
            v = d[pos]
            n0 = len(out)
            if role == "checksum":
                exp = compute(r["algo"], d, pos)
                if exp != v:
                    out.append(Alert(f.timestamp, f.can_id, "CHECKSUM_FAIL", self.name, "high",
                                     f"byte {pos}: {r['algo']} checksum 0x{v:02X} != expected 0x{exp:02X}"))
            elif role == "counter":
                part, mod = r["part"], r["mod"]
                if r["other"] is not None:
                    o = v >> 4 if part == "lo" else v & 0x0F
                    if o not in r["other"]:
                        out.append(Alert(f.timestamp, f.can_id, "ENUM_VIOLATION", self.name, "medium",
                                         f"byte {pos}: unexpected value 0x{v:02X} next to counter"))
                if prev is not None:
                    cur, old = _nib(v, part), _nib(prev[1][pos], part)
                    delta = (cur - old) % mod
                    if steps is None:
                        ok = 1 <= delta <= 3
                        exp_txt = "1..3"
                    else:
                        e = steps % mod
                        ok = min((delta - e) % mod, (e - delta) % mod) <= 1 and not (delta == 0 and steps < mod)
                        exp_txt = f"{e}"
                    if not ok:
                        out.append(Alert(f.timestamp, f.can_id, "COUNTER_BREAK", self.name, "high",
                                         f"byte {pos}: rolling counter advanced by {delta}, expected {exp_txt}"))
            elif role == "enum":
                if v not in r["values"]:
                    out.append(Alert(f.timestamp, f.can_id, "ENUM_VIOLATION", self.name, "medium",
                                     f"byte {pos}: value 0x{v:02X} never seen (learned {len(r['values'])} values)"))
                elif prev is not None and (steps or 1) == 1 and "transitions" in r:
                    pv = prev[1][pos]
                    if pv != v and [pv, v] not in r["transitions"]:
                        out.append(Alert(f.timestamp, f.can_id, "STATE_TRANSITION", self.name, "medium",
                                         f"byte {pos}: transition 0x{pv:02X} -> 0x{v:02X} never observed",
                                         {"transition": [pv, v]}))
            elif role == "range":
                span = r["max"] - r["min"]
                margin = max(2, int((0.25 if r.get("soft") else 0.05) * span))
                if r.get("signed") and v >= 128:
                    v = v - 256
                if v < r["min"] - margin or v > r["max"] + margin:
                    # slow numeric bytes (fuel level, odometer, ...) drift between drives:
                    # small excursions are reported but do not flag the frame
                    dist = max(r["min"] - margin - v, v - r["max"] - margin)
                    sev = "low" if r.get("soft") and dist <= 32 else "medium"
                    out.append(Alert(f.timestamp, f.can_id, "RANGE_VIOLATION", self.name, sev,
                                     f"byte {pos}: value {v} outside learned range [{r['min']}, {r['max']}]"))
                elif prev is not None and r["max_delta"] is not None:
                    allowed = r["max_delta"] * max(1, steps or 1) * 1.5 + 1
                    if abs(v - prev[1][pos]) > allowed:
                        out.append(Alert(f.timestamp, f.can_id, "RATE_VIOLATION", self.name, "medium",
                                         f"byte {pos}: jumped {abs(v - prev[1][pos])} (max {allowed:.0f} per "
                                         f"{steps or 1} period(s))"))
            for a in out[n0:]:
                a.extra["byte"] = pos
        return out

    def relax(self, can_id: int, pos: int, transition: list | None = None) -> None:
        """Validation pruning: drop (or extend) a byte rule that fired on clean traffic."""
        p = self.prof.get(can_id)
        if not p or pos >= len(p["bytes"]):
            return
        r = p["bytes"][pos]
        if transition is not None and r.get("role") == "enum":
            for t in (list(transition), list(transition)[::-1]):
                if t not in r["transitions"]:
                    r["transitions"].append(t)
            return
        p["bytes"][pos] = {"role": "free", "pruned": True}

    def commit(self, f: CANFrame) -> None:
        if f.can_id in self.prof:
            self.last[f.can_id] = (f.timestamp, f.data)

    def describe(self) -> dict:
        """Human readable summary: which roles were discovered per ID."""
        out = {}
        for cid, p in self.prof.items():
            out[f"0x{cid:03X}"] = [r["role"] + (f"({r['algo']})" if r["role"] == "checksum" else
                                                f"({r['part']})" if r["role"] == "counter" else "")
                                   for r in p["bytes"]]
        return out

    def to_dict(self) -> dict:
        return {f"{k:03X}": v for k, v in self.prof.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "PayloadLayer":
        p = cls()
        p.prof = {int(k, 16): v for k, v in d.items()}
        return p
