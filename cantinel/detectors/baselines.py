"""Re-implementations of two classic CAN IDS approaches used as baselines.

* ``IntervalBaseline`` - time-interval analysis in the spirit of
  Song, Kim & Kim (ICOIN 2016): a frame is malicious if it arrives less than
  half of the learned mean interval after the previous frame with the same ID.
* ``EntropyBaseline`` - ID-entropy analysis in the spirit of Mueter & Asaj
  (IEEE IV 2011): a window is malicious if its ID entropy deviates from the
  learned mean by more than k standard deviations.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from ..frame import CANFrame


class IntervalBaseline:
    def __init__(self, ratio: float = 0.5):
        self.ratio = ratio
        self.mean: dict[int, float] = {}

    def fit(self, frames: list[CANFrame]) -> "IntervalBaseline":
        ts = defaultdict(list)
        for f in frames:
            ts[f.can_id].append(f.timestamp)
        self.mean = {k: float(np.mean(np.diff(v))) for k, v in ts.items() if len(v) > 2}
        self.fastest = min(self.mean.values())
        return self

    def run(self, frames: list[CANFrame]) -> np.ndarray:
        last: dict[int, float] = {}
        out = np.zeros(len(frames), dtype=np.int8)
        for i, f in enumerate(frames):
            ref = self.mean.get(f.can_id, self.fastest)
            p = last.get(f.can_id)
            if p is not None and f.timestamp - p < self.ratio * ref:
                out[i] = 1
            last[f.can_id] = f.timestamp
        return out


def _entropy(ids) -> float:
    c = Counter(ids)
    n = sum(c.values())
    return -sum((v / n) * math.log2(v / n) for v in c.values()) if n else 0.0


class EntropyBaseline:
    def __init__(self, window: float = 0.1, k: float = 3.0):
        self.window = window
        self.k = k

    def _series(self, frames):
        t0 = frames[0].timestamp
        widx = np.array([int((f.timestamp - t0) / self.window) for f in frames])
        nwin = int(widx[-1]) + 1
        buckets = [[] for _ in range(nwin)]
        for f, w in zip(frames, widx):
            buckets[w].append(f.can_id)
        return np.array([_entropy(b) for b in buckets]), widx

    def fit(self, frames: list[CANFrame], validation: list[CANFrame] | None = None) -> "EntropyBaseline":
        h, _ = self._series(frames)
        h = h[1:-1]
        self.mu, self.sd = float(np.mean(h)), float(np.std(h)) or 1e-6
        self.thr = self.k * self.sd
        if validation:
            hv, _ = self._series(validation)
            self.thr = max(self.thr, float(np.max(np.abs(hv[1:-1] - self.mu))) * 1.02)
        return self

    def predict(self, frames: list[CANFrame]):
        h, widx = self._series(frames)
        return (np.abs(h - self.mu) > self.thr).astype(int), widx
