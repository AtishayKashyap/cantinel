"""Layer 2 - phase-locked timing analysis.

Classic interval-based CAN IDSs compare the time since the *previous* frame
with the learned period.  During an injection both the injected frame and the
next legitimate one see a short interval, so legitimate frames are flagged as
well, and floods that merely delay frames trigger false alarms.

CANtinel instead locks onto the *transmission schedule* of every periodic ID
with an asymmetric first-order phase-locked loop (PLL):

* the anchor ``t_ref`` is the estimated ideal send time of the last accepted
  frame and is only advanced by frames accepted by **all** layers;
* a frame is *early* if ``t < t_ref + period - tol``.  Because a legitimate
  ECU can be delayed by arbitration but never transmits ahead of its
  schedule, ``tol`` only has to cover clock jitter (typically < 0.3 ms),
  which leaves an attacker a very small acceptance window;
* frames that are late by more than ``tol`` (bus floods, arbitration) are
  accepted without moving the phase estimate.
"""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..frame import Alert, CANFrame

# asymmetric loop gains: a legitimate ECU is never *ahead* of its schedule, so an
# early arrival pulls the phase estimate quickly while late (delayed) arrivals
# only nudge it - the anchor therefore follows the lower envelope of arrivals.
ALPHA_EARLY = 0.5
ALPHA_LATE = 0.02


def _pll_update(exp: float, err: float, tol: float) -> float:
    if err > tol:
        return exp
    return exp + (ALPHA_EARLY if err < 0 else ALPHA_LATE) * err


class TimingLayer:
    name = "timing"

    def __init__(self, min_samples: int = 20, cv_periodic: float = 0.2,
                 tol_floor: float = 0.0001, tol_scale: float = 1.25):
        self.min_samples = min_samples
        self.cv_periodic = cv_periodic
        self.tol_floor = tol_floor
        self.tol_scale = tol_scale
        self.prof: dict[int, dict] = {}
        self.t_ref: dict[int, float] = {}

    # ---------------------------------------------------------------- train
    def fit(self, frames: list[CANFrame]) -> "TimingLayer":
        ts = defaultdict(list)
        for f in frames:
            ts[f.can_id].append(f.timestamp)
        for cid, t in ts.items():
            if len(t) < self.min_samples:
                continue
            t = np.asarray(t)
            iv = np.diff(t)
            med = float(np.median(iv))
            mad = float(np.median(np.abs(iv - med))) * 1.4826
            cv = mad / med if med > 0 else 1.0
            prof = {"periodic": bool(cv < self.cv_periodic), "period": med, "jitter": mad,
                    "tol": 0.0, "min_iv": float(iv.min())}
            if prof["periodic"]:
                # run the PLL over the training data and measure how early frames can be
                prof["tol"] = 0.3 * med
                errs = self._pll_errors(t, prof)
                early = float(max(0.0, -errs.min())) if len(errs) else 0.0
                # floor of 1 % of the period: slow IDs see rare, long arbitration
                # delays that a short training drive may not contain
                prof["tol"] = float(min(max(self.tol_scale * early, 4 * float(np.std(errs)), self.tol_floor,
                                            0.01 * med), 0.45 * med))
            self.prof[cid] = prof
        return self

    @staticmethod
    def _pll_errors(t: np.ndarray, p: dict) -> np.ndarray:
        ref = t[0]
        out = []
        per, tol = p["period"], p["tol"]
        for x in t[1:]:
            d = x - ref
            n = max(1, math.floor((d + tol) / per))
            exp = ref + n * per
            err = x - exp
            out.append(err)
            ref = _pll_update(exp, err, tol)
        return np.asarray(out)

    def reset(self):
        self.t_ref.clear()

    # --------------------------------------------------------------- detect
    def check(self, f: CANFrame) -> list[Alert]:
        p = self.prof.get(f.can_id)
        ref = self.t_ref.get(f.can_id)
        if p is None or ref is None:
            return []
        d = f.timestamp - ref
        if p["periodic"]:
            if d < p["period"] - p["tol"]:
                return [Alert(f.timestamp, f.can_id, "TIMING_EARLY", self.name, "high",
                              f"arrived {(p['period'] - d) * 1e3:.2f} ms ahead of the ECU's schedule "
                              f"(period {p['period'] * 1e3:.2f} ms, tolerance {p['tol'] * 1e3:.2f} ms)",
                              {"early_ms": round((p["period"] - d) * 1e3, 3)})]
        elif d < 0.5 * p["min_iv"]:
            return [Alert(f.timestamp, f.can_id, "TIMING_BURST", self.name, "medium",
                          f"event-driven ID repeated after {d * 1e3:.2f} ms "
                          f"(never faster than {p['min_iv'] * 1e3:.1f} ms in training)")]
        return []

    def commit(self, f: CANFrame) -> None:
        p = self.prof.get(f.can_id)
        ref = self.t_ref.get(f.can_id)
        if p is None or ref is None or not p["periodic"]:
            self.t_ref[f.can_id] = f.timestamp
            return
        per, tol = p["period"], p["tol"]
        d = f.timestamp - ref
        n = max(1, math.floor((d + tol) / per))
        exp = ref + n * per
        err = f.timestamp - exp
        self.t_ref[f.can_id] = _pll_update(exp, err, tol)

    def widen(self, can_id: int, early_s: float) -> None:
        """Used by validation pruning: make tolerance cover an observed legit frame."""
        p = self.prof.get(can_id)
        if p and p["periodic"]:
            p["tol"] = float(min(max(p["tol"], self.tol_scale * early_s), 0.45 * p["period"]))

    def to_dict(self) -> dict:
        return {f"{k:03X}": v for k, v in self.prof.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "TimingLayer":
        t = cls()
        t.prof = {int(k, 16): v for k, v in d.items()}
        return t
