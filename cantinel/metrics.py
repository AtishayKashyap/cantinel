"""Evaluation metrics (frame level, window level, detection latency)."""
from __future__ import annotations

import numpy as np

from .frame import CANFrame


def prf(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true).astype(bool)
    y_pred = np.asarray(y_pred).astype(bool)
    tp = int(np.sum(y_true & y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    p = tp / (tp + fp) if tp + fp else (1.0 if not fn else 0.0)
    r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": p, "recall": r, "f1": f1, "fpr": fpr}


def frames_to_windows(flags: np.ndarray, widx: np.ndarray, nwin: int) -> np.ndarray:
    out = np.zeros(nwin, dtype=int)
    np.maximum.at(out, widx, flags.astype(int))
    return out


def detection_latency(frames: list[CANFrame], flags: np.ndarray, gap: float = 1.0) -> float | None:
    """Mean time (ms) from the first injected frame of each burst to the first
    flagged injected frame of that burst."""
    lat = []
    burst_start = None
    detected = False
    last_attack_t = -1e9
    for f, fl in zip(frames, flags):
        if not f.label:
            continue
        if f.timestamp - last_attack_t > gap:      # new burst
            burst_start, detected = f.timestamp, False
        last_attack_t = f.timestamp
        if fl and not detected:
            lat.append((f.timestamp - burst_start) * 1e3)
            detected = True
    return float(np.mean(lat)) if lat else None
