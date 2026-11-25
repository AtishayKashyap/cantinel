"""Figures for the paper / README (matplotlib, PDF + PNG)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

# validated categorical order (light surface): blue, orange, aqua, yellow
C = ["#257de0", "#f47230", "#00b894", "#f2a600"]
INK, INK2, GRID, NEUTRAL = "#191919", "#666666", "#e6e6e6", "#a8a8a8"

ATTACK_LABEL = {"dos": "DoS", "fuzzy": "Fuzzy", "spoof_rpm": "Spoof\nRPM", "spoof_gear": "Spoof\ngear",
                "replay": "Replay", "masquerade": "Masque-\nrade", "stealth_masquerade": "Stealth\nmasq."}


def _style(plt):
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8.5, "axes.edgecolor": INK2, "axes.labelcolor": INK,
        "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.6, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "legend.frameon": False, "pdf.fonttype": 42,
    })


def fig_window_f1(r: dict, out: Path):
    import matplotlib.pyplot as plt
    _style(plt)
    M = r["metrics"]
    attacks = r["config"]["attacks"]
    methods = [("cantinel", "CANtinel (ours)"), ("entropy", "Entropy [Müter]"),
               ("interval", "Interval [Song]"), ("knn", "k-NN novelty (ML layer)")]
    fig, ax = plt.subplots(figsize=(6.2, 2.1))
    x = np.arange(len(attacks))
    w = 0.2
    for i, (k, lab) in enumerate(methods):
        mean = [M[f"window|{k}|{a}|f1"]["mean"] for a in attacks]
        std = [M[f"window|{k}|{a}|f1"]["std"] for a in attacks]
        ax.bar(x + (i - 1.5) * w, mean, w, color=C[i], edgecolor="white", linewidth=1.0, label=lab,
               yerr=std, error_kw={"elinewidth": 0.6, "ecolor": INK2, "capsize": 0})
    ax.set_xticks(x, [ATTACK_LABEL[a] for a in attacks])
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Window-level F1")
    ax.yaxis.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.2), fontsize=7.5, handlelength=1.2)
    fig.tight_layout()
    fig.savefig(out / "fig_window_f1.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_window_f1.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_phase(out: Path, seed: int = 1, burst: int = 1):
    """Phase of every 0x43F frame relative to the gear ECU's schedule around an attack."""
    import matplotlib.pyplot as plt

    from .engine import Engine, Profile
    from .simulator import simulate
    _style(plt)
    vs = 7 + seed
    prof = Profile.train(simulate(300, None, seed=100 + seed, vehicle_seed=vs),
                         simulate(120, None, seed=150 + seed, vehicle_seed=vs))
    test = simulate(60, "spoof_gear", seed=200 + seed, vehicle_seed=vs)
    flags, _ = Engine(prof).run(test)
    cid = 0x43F
    per = prof.timing.prof[cid]["period"]
    tol = prof.timing.prof[cid]["tol"]
    starts, last = [], -1e9
    for f in test:
        if f.label:
            if f.timestamp - last > 1.0:
                starts.append(f.timestamp)
            last = f.timestamp
    t_atk = starts[min(burst, len(starts) - 1)]
    sel = [(f, fl) for f, fl in zip(test, flags) if f.can_id == cid and t_atk - 0.4 <= f.timestamp <= t_atk + 0.8]
    legit0 = [f.timestamp for f, _ in sel if not f.label][:20]
    t0 = float(np.median(np.asarray(legit0) - np.round((np.asarray(legit0) - legit0[0]) / per) * per))

    def phase(t):
        return (((t - t0) + per / 2) % per - per / 2) * 1e3

    groups = {
        "legitimate, accepted": ([], NEUTRAL, "o", 9),
        "injected, detected": ([], C[0], "x", 12),
        "injected, missed": ([], C[1], "^", 16),
        "legitimate, false alarm": ([], C[2], "s", 12),
    }
    for f, fl in sel:
        key = ("injected, detected" if fl else "injected, missed") if f.label else \
              ("legitimate, false alarm" if fl else "legitimate, accepted")
        groups[key][0].append(((f.timestamp - t_atk) * 1e3, phase(f.timestamp)))
    fig, ax = plt.subplots(figsize=(6.2, 2.0))
    ax.axhspan(-tol * 1e3, tol * 1e3, color="#cce3fd", alpha=0.7, linewidth=0, zorder=0)
    ax.text(-395, -tol * 1e3 - 0.35, f"ECU schedule \u00b1 tolerance ({tol * 1e3:.2f} ms)",
            color=INK2, fontsize=6.5, va="top")
    for lab, (pts, col, mk, sz) in groups.items():
        if not pts:
            continue
        a = np.asarray(pts)
        kw = {"facecolors": "none", "edgecolors": col} if mk in ("^", "s") else {"c": col}
        ax.scatter(a[:, 0], a[:, 1], s=sz, marker=mk, linewidths=0.8, label=f"{lab} ({len(a)})",
                   zorder=4 if lab.startswith("legitimate") else 3, **kw)
    ax.axvline(0, color=INK2, linewidth=0.6, linestyle="--")
    ax.text(-8, -5.2, "attack starts", color=INK2, fontsize=6.5, ha="right")
    ax.set_xlim(-400, 800)
    ax.set_ylim(-6, 6)
    ax.set_xlabel("time relative to attack start (ms)")
    ax.set_ylabel("phase vs. ECU schedule (ms)")
    ax.legend(ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.32), fontsize=7.5, handletextpad=0.3)
    fig.tight_layout()
    fig.savefig(out / "fig_phase.pdf", bbox_inches="tight")
    fig.savefig(out / "fig_phase.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def make_all(r: dict, out_dir: str | Path = "docs/figures"):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig_window_f1(r, out)
    fig_phase(out)
