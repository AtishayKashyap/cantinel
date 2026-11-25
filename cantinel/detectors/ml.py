"""Layer 4 (optional) - unsupervised window-level anomaly detection.

A novelty detector is trained on 100 ms windows of *attack-free* traffic only,
so no labelled attack data is needed (unlike most supervised CAN IDS models).
It acts as a safety net for attack shapes the explainable rule layers do not
model, and reports window-level anomalies.

The default model is a k-nearest-neighbour novelty detector: the anomaly
score of a window is its mean standardised distance to the k closest
attack-free training windows.  An Isolation Forest is available (``model="iforest"``) but note a
known weakness: points far *outside* the training range land in the same leaf
as the most extreme training points, so the IF score saturates and floods
(DoS) can score no higher than unusual but benign windows.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..frame import CANFrame
from ..simulator import frame_bits

FEATURES = ["frames", "unique_ids", "id_entropy", "unknown_frac", "min_gap_ms",
            "max_rate_ratio", "hamming_mean", "below_min_id_frac", "bus_load"]


def window_features(frames: list[CANFrame], window: float, known: set[int],
                    periods: dict[int, float], bitrate: int = 500_000):
    """Returns (X, window_start_times, window_index_per_frame)."""
    if not frames:
        return np.zeros((0, len(FEATURES))), np.zeros(0), np.zeros(0, dtype=int)
    t0 = frames[0].timestamp
    widx = np.array([int((f.timestamp - t0) / window) for f in frames])
    nwin = int(widx[-1]) + 1
    min_id = min(known) if known else 0
    rows = []
    last_payload: dict[int, bytes] = {}
    start = 0
    for w in range(nwin):
        end = start
        while end < len(frames) and widx[end] == w:
            end += 1
        fr = frames[start:end]
        start = end
        n = len(fr)
        if n == 0:
            rows.append([0, 0, 0, 0, window * 1e3, 0, 0, 0, 0])
            continue
        ids = Counter(f.can_id for f in fr)
        ent = -sum((c / n) * math.log2(c / n) for c in ids.values())
        unknown = sum(c for i, c in ids.items() if i not in known) / n
        ts = np.array([f.timestamp for f in fr])
        min_gap = float(np.min(np.diff(ts))) * 1e3 if n > 1 else window * 1e3
        ratios = [c / max(window / periods[i], 1.0) for i, c in ids.items() if i in periods]
        max_ratio = max(ratios) if ratios else 0.0
        ham = []
        for f in fr:
            p = last_payload.get(f.can_id)
            if p is not None and len(p) == len(f.data) and f.dlc:
                ham.append(sum(bin(a ^ b).count("1") for a, b in zip(p, f.data)) / (8 * f.dlc))
            last_payload[f.can_id] = f.data
        below = sum(c for i, c in ids.items() if i < min_id) / n
        load = sum(frame_bits(f.dlc) for f in fr) / (window * bitrate)
        rows.append([n, len(ids), ent, unknown, min_gap, max_ratio,
                     float(np.mean(ham)) if ham else 0.0, below, load])
    return np.asarray(rows, dtype=float), t0 + np.arange(nwin) * window, widx


def window_labels(frames: list[CANFrame], widx: np.ndarray, nwin: int) -> np.ndarray:
    y = np.zeros(nwin, dtype=int)
    for f, w in zip(frames, widx):
        if f.label:
            y[w] = 1
    return y


class KNNNovelty:
    """Mean distance to the k nearest attack-free windows (higher = more anomalous)."""

    def __init__(self, k: int = 5):
        self.k = k
        self.scaler = StandardScaler()
        self.nn = NearestNeighbors(n_neighbors=k)

    def fit(self, X):
        Z = self.scaler.fit_transform(X)
        self.nn.fit(Z)
        return self

    def score_samples(self, X):  # sklearn convention: lower = more abnormal
        d, _ = self.nn.kneighbors(self.scaler.transform(X))
        return -d.mean(axis=1)


class MLLayer:
    name = "ml"

    def __init__(self, window: float = 0.1, model: str = "knn", seed: int = 0):
        self.window = window
        self.kind = model
        if model == "iforest":
            self.model = IsolationForest(n_estimators=200, random_state=seed, contamination="auto")
        else:
            self.model = KNNNovelty(k=5)
        self.known: set[int] = set()
        self.periods: dict[int, float] = {}
        self.threshold = 0.0

    def fit(self, frames: list[CANFrame], periods: dict[int, float],
            validation: list[CANFrame] | None = None) -> "MLLayer":
        self.known = {f.can_id for f in frames}
        self.periods = periods
        X, _, _ = window_features(frames, self.window, self.known, periods)
        X = X[1:-1]  # drop partial edge windows
        self.model.fit(X)
        ref = X
        if validation:
            Xv, _, _ = window_features(validation, self.window, self.known, periods)
            ref = Xv[1:-1]
        s = -self.model.score_samples(ref)
        # threshold: slightly above the most anomalous attack-free window
        self.threshold = float(np.max(s) * 1.02)
        return self

    def predict(self, frames: list[CANFrame]):
        X, starts, widx = window_features(frames, self.window, self.known, self.periods)
        s = -self.model.score_samples(X) if len(X) else np.zeros(0)
        return (s > self.threshold).astype(int), s, widx
