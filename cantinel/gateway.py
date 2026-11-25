"""Intrusion *prevention*: a filtering security gateway between CAN domains.

The gateway sits between e.g. the infotainment / OBD domain and the
powertrain domain (the "Gateway ECU" of the AUTOSAR security architecture)
and forwards a frame only if

1. its ID is on the learned whitelist,
2. the CANtinel engine raised no high/medium alert for it, and
3. the per-ID token bucket (learned period x 1.2, burst 3) is not exhausted,
   which caps the damage of any flood even if it is not yet classified.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .engine import Engine
from .frame import CANFrame


@dataclass
class GatewayStats:
    forwarded: int = 0
    blocked: int = 0
    legit_forwarded: int = 0
    legit_blocked: int = 0
    attack_forwarded: int = 0
    attack_blocked: int = 0
    reasons: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        legit = self.legit_forwarded + self.legit_blocked
        atk = self.attack_forwarded + self.attack_blocked
        return {
            **{k: v for k, v in self.__dict__.items() if k != "reasons"},
            "legit_pass_rate": self.legit_forwarded / legit if legit else 1.0,
            "attack_block_rate": self.attack_blocked / atk if atk else 1.0,
            "reasons": self.reasons,
        }


class SecureGateway:
    def __init__(self, engine: Engine, rate_factor: float = 1.2, burst: float = 3.0):
        self.engine = engine
        self.rate = {cid: rate_factor / p["period"] for cid, p in engine.timing.prof.items() if p["periodic"]}
        self.burst = burst
        self.tokens: dict[int, tuple[float, float]] = {}

    def _bucket(self, f: CANFrame) -> bool:
        r = self.rate.get(f.can_id)
        if r is None:
            return True
        tok, last = self.tokens.get(f.can_id, (self.burst, f.timestamp))
        tok = min(self.burst, tok + (f.timestamp - last) * r)
        if tok < 1.0:
            self.tokens[f.can_id] = (tok, f.timestamp)
            return False
        self.tokens[f.can_id] = (tok - 1.0, f.timestamp)
        return True

    def run(self, frames: list[CANFrame]) -> tuple[list[CANFrame], GatewayStats]:
        self.engine.reset()
        self.tokens.clear()
        st = GatewayStats()
        out = []
        for f in frames:
            alerts = [a for a in self.engine.process(f) if a.flags]
            reason = alerts[0].kind if alerts else (None if self._bucket(f) else "RATE_LIMIT")
            if reason is None:
                out.append(f)
                st.forwarded += 1
                if f.label:
                    st.attack_forwarded += 1
                else:
                    st.legit_forwarded += 1
            else:
                st.blocked += 1
                st.reasons[reason] = st.reasons.get(reason, 0) + 1
                if f.label:
                    st.attack_blocked += 1
                else:
                    st.legit_blocked += 1
        return out, st
