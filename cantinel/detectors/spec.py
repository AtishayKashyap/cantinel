"""Layer 1 - specification / whitelist checks (stateless, O(1) per frame)."""
from __future__ import annotations

from collections import Counter, defaultdict

from ..frame import Alert, CANFrame


class SpecLayer:
    name = "spec"

    def __init__(self):
        self.dlc: dict[int, set[int]] = {}
        self.min_id: int | None = None

    def fit(self, frames: list[CANFrame]) -> "SpecLayer":
        d = defaultdict(Counter)
        for f in frames:
            d[f.can_id][f.dlc] += 1
        # keep DLCs that make up >=0.1 % of an ID's frames (ignores rare noise)
        self.dlc = {cid: {k for k, v in c.items() if v >= 0.001 * sum(c.values())} for cid, c in d.items()}
        self.min_id = min(self.dlc) if self.dlc else None
        return self

    def check(self, f: CANFrame) -> list[Alert]:
        allowed = self.dlc.get(f.can_id)
        if allowed is None:
            msg = f"ID 0x{f.can_id:03X} is not part of this vehicle's communication matrix"
            if self.min_id is not None and f.can_id < self.min_id:
                msg += " (higher priority than any legitimate ID - typical of priority-flood DoS)"
            return [Alert(f.timestamp, f.can_id, "UNKNOWN_ID", self.name, "high", msg)]
        if f.dlc not in allowed:
            return [Alert(f.timestamp, f.can_id, "DLC_MISMATCH", self.name, "high",
                          f"DLC {f.dlc} not in learned set {sorted(allowed)}")]
        return []

    def to_dict(self) -> dict:
        return {f"{k:03X}": sorted(v) for k, v in self.dlc.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "SpecLayer":
        s = cls()
        s.dlc = {int(k, 16): set(v) for k, v in d.items()}
        s.min_id = min(s.dlc) if s.dlc else None
        return s
