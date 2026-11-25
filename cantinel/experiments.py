"""Reproducible benchmark used for the paper (``cantinel benchmark``).

For every seed a *different* vehicle instance is simulated; CANtinel is trained
on a 300 s attack-free drive, self-tuned on a second 120 s attack-free drive,
and tested on one 60 s capture per attack type (plus one clean capture).
"""
from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .detectors.baselines import EntropyBaseline, IntervalBaseline
from .detectors.ml import MLLayer, window_labels
from .engine import Engine, Profile
from .gateway import SecureGateway
from .metrics import detection_latency, frames_to_windows, prf
from .secoc import AuthLayer, KeyStore, SecOCConfig, SecOCReceiver, SecOCSender, benchmark
from .simulator import ATTACK_KINDS, simulate
from .vehicle import CATALOG_BY_ID, generate_trace

FRAME_METHODS = ["interval", "cantinel"]
WINDOW_METHODS = ["entropy", "interval", "iforest", "knn", "cantinel", "cantinel+ml"]
ABLATIONS = {"spec": ("spec",), "spec+timing": ("spec", "timing"),
             "spec+payload": ("spec", "payload"), "all": ("spec", "timing", "payload")}


def run_suite(seeds=(1, 2, 3, 4, 5), train_s=300, val_s=120, test_s=60,
              out_dir: str | Path = "results", log=print) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rec = defaultdict(list)          # (level, method, attack, metric) -> values
    latency = defaultdict(list)
    gateway = defaultdict(list)
    throughput = []
    profile_info = []
    for s in seeds:
        vs = 7 + s
        t0 = time.time()
        train = simulate(train_s, None, seed=100 + s, vehicle_seed=vs)
        val = simulate(val_s, None, seed=150 + s, vehicle_seed=vs)
        prof = Profile.train(train, val)
        profile_info.append({"seed": s, **prof.meta, "summary": prof.summary()})
        periods = {c: p["period"] for c, p in prof.timing.prof.items() if p["periodic"]}
        engines = {k: Engine(prof, layers=v) for k, v in ABLATIONS.items()}
        interval = IntervalBaseline().fit(train)
        entropy = EntropyBaseline().fit(train, val)
        ml = MLLayer(seed=s).fit(train, periods, val)
        iforest = MLLayer(model="iforest", seed=s).fit(train, periods, val)
        gw = SecureGateway(Engine(prof))
        for atk in ["clean"] + ATTACK_KINDS:
            test = simulate(test_s, None if atk == "clean" else atk, seed=200 + s, vehicle_seed=vs)
            y = np.array([f.label for f in test])
            fl = {}
            for k, e in engines.items():
                t1 = time.perf_counter()
                fl[k], _ = e.run(test)
                if k == "all":
                    throughput.append(len(test) / (time.perf_counter() - t1))
            fl["interval"] = interval.run(test)
            for k in ABLATIONS:
                for m, v in prf(y, fl[k]).items():
                    rec[("ablation", k, atk, m)].append(v)
            for name, key in (("interval", "interval"), ("cantinel", "all")):
                for m, v in prf(y, fl[key]).items():
                    rec[("frame", name, atk, m)].append(v)
                if atk != "clean":
                    lt = detection_latency(test, fl[key])
                    if lt is not None:
                        latency[(name, atk)].append(lt)
            # window level
            mlp, _, widx = ml.predict(test)
            nwin = len(mlp)
            yw = window_labels(test, widx, nwin)
            entp, _ = entropy.predict(test)
            wpred = {
                "entropy": entp[:nwin],
                "interval": frames_to_windows(fl["interval"], widx, nwin),
                "iforest": iforest.predict(test)[0],
                "knn": mlp,
                "cantinel": frames_to_windows(fl["all"], widx, nwin),
            }
            wpred["cantinel+ml"] = np.maximum(wpred["cantinel"], mlp)
            for k, p in wpred.items():
                for m, v in prf(yw, p).items():
                    rec[("window", k, atk, m)].append(v)
            # prevention
            _, st = gw.run(test)
            d = st.as_dict()
            gateway[atk].append({"legit_pass_rate": d["legit_pass_rate"], "attack_block_rate": d["attack_block_rate"]})
        log(f"seed {s} done in {time.time() - t0:.1f}s")

    def agg(vals):
        a = np.asarray(vals, dtype=float)
        return {"mean": float(a.mean()), "std": float(a.std())}

    results = {
        "config": {"seeds": list(seeds), "train_s": train_s, "val_s": val_s, "test_s": test_s,
                   "attacks": ATTACK_KINDS},
        "metrics": {"|".join(k): agg(v) for k, v in rec.items()},
        "latency_ms": {"|".join(k): agg(v) for k, v in latency.items()},
        "gateway": {k: {m: agg([x[m] for x in v]) for m in v[0]} for k, v in gateway.items()},
        "throughput_fps": agg(throughput),
        "profiles": profile_info,
        "secoc": secoc_experiment(),
    }
    (out_dir / "results.json").write_text(json.dumps(results, indent=1))
    return results


def secoc_experiment(seed: int = 3, duration: float = 60.0, cid: int = 0x4B0) -> dict:
    """Authenticated vehicle-speed PDUs (CAN FD, 8 B payload + 4 B SecOC trailer)."""
    rng = np.random.default_rng(seed)
    cfg = SecOCConfig()
    keys = KeyStore([cid], seed=b"cantinel-demo-ks")
    tx = SecOCSender(keys[cid], cid, cfg)
    trace = generate_trace(duration + 1, rng)
    m = CATALOG_BY_ID[cid]
    legit = []
    for k, t in enumerate(np.arange(0.0, duration, m.period)):
        legit.append((float(t), tx.protect(m.encode(trace.state(t), k)), m.encode(trace.state(t), k)))

    def accept_rate(pdus, rx=None, drop=0.0):
        rx = rx or SecOCReceiver(keys[cid], cid, cfg)
        ok = 0
        n = 0
        for p in pdus:
            if drop and rng.random() < drop:
                continue
            n += 1
            ok += rx.verify(p)[0]
        return ok, n

    res = {}
    ok, n = accept_rate([p for _, p, _ in legit])
    res["legitimate"] = {"accepted": ok, "sent": n}
    ok, n = accept_rate([p for _, p, _ in legit], drop=0.1)
    res["legitimate_10pct_loss"] = {"accepted": ok, "sent": n}

    # attacker sniffed the first 10 s and replays them after the legitimate stream
    rx = SecOCReceiver(keys[cid], cid, cfg)
    for _, p, _ in legit:
        rx.verify(p)
    sniffed = [p for t, p, _ in legit if t < 10]
    ok, n = accept_rate(sniffed, rx)
    res["replay"] = {"accepted": ok, "sent": n}

    # spoofing / stealth masquerade: correct inner payload (counter + CRC), no key
    def interleaved_attack(make_pdu):
        rx = SecOCReceiver(keys[cid], cid, cfg)
        acc = att = legit_ok = 0
        last_pdu = None
        for i, (t, p, inner) in enumerate(legit):
            legit_ok += rx.verify(p)[0]
            last_pdu = p
            if 1000 <= i < 2000:
                st = dict(trace.state(t))
                st["speed"] *= 0.6
                forged = m.encode(st, i)
                att += 1
                acc += rx.verify(make_pdu(forged, last_pdu))[0]
        return {"accepted": acc, "sent": att, "legit_accepted": legit_ok, "legit_sent": len(legit)}

    res["spoof_random_mac"] = interleaved_attack(lambda inner, last: inner + bytes(rng.integers(0, 256, 4, dtype=np.uint8)))
    res["stealth_masquerade_reuse_trailer"] = interleaved_attack(lambda inner, last: inner + last[-4:])
    res["stealth_masquerade_next_fv"] = interleaved_attack(
        lambda inner, last: inner + bytes(((last[-4] + 1) & 0xFF,)) + bytes(rng.integers(0, 256, 3, dtype=np.uint8)))

    # brute force on one PDU: probability of success per guess is 2^-24
    rx = SecOCReceiver(keys[cid], cid, cfg)
    target = legit[0][2]
    tries = 200_000
    hits = 0
    for _ in range(tries):
        g = target + bytes((1,)) + bytes(rng.integers(0, 256, 3, dtype=np.uint8))
        if rx.verify(g)[0]:
            hits += 1
            rx.last_fv = 0
    res["brute_force"] = {"tries": tries, "accepted": hits, "theoretical_p": 2 ** -cfg.mac_bits,
                          "expected_hours_at_1000fps": (2 ** cfg.mac_bits / 2) / 1000 / 3600}
    res["overhead_bytes"] = cfg.overhead
    res["timing_us_python"] = benchmark(5000)

    # IDS integration: the AUTH layer turns failures into alerts + brute-force detection
    auth = AuthLayer(keys, cfg)
    from .frame import CANFrame
    alerts = []
    for i, (t, p, _) in enumerate(legit[:1500]):
        alerts += auth.check(CANFrame(t, cid, len(p), p))
        if i >= 1000:
            g = legit[i][2] + bytes((1,)) + bytes(rng.integers(0, 256, 3, dtype=np.uint8))
            alerts += auth.check(CANFrame(t + 0.001, cid, len(g), g, 1))
    res["ids_alerts"] = {k: sum(1 for a in alerts if a.kind == k) for k in ("AUTH_FAIL", "MAC_BRUTE_FORCE")}
    return res


# ---------------------------------------------------------------- reporting
ATTACK_LABEL = {"dos": "DoS", "fuzzy": "Fuzzy", "spoof_rpm": "Spoof RPM", "spoof_gear": "Spoof gear",
                "replay": "Replay", "masquerade": "Masquerade", "stealth_masquerade": "Stealth masq."}


def markdown_tables(r: dict) -> str:
    M = r["metrics"]
    g = lambda *k: M["|".join(k)]["mean"]
    lines = ["### Frame level (mean of %d seeds)" % len(r["config"]["seeds"]), "",
             "| Attack | Interval P | Interval R | Interval F1 | CANtinel P | CANtinel R | CANtinel F1 |",
             "|---|---|---|---|---|---|---|"]
    for a in r["config"]["attacks"]:
        lines.append(f"| {ATTACK_LABEL[a]} | " + " | ".join(
            f"{g('frame', m, a, k):.3f}" for m in ("interval", "cantinel") for k in ("precision", "recall", "f1")) + " |")
    lines.append(f"| Clean FPR | {g('frame', 'interval', 'clean', 'fpr'):.2e} | | | {g('frame', 'cantinel', 'clean', 'fpr'):.2e} | | |")
    lines += ["", "### Window level F1 (100 ms windows)", "",
              "| Attack | " + " | ".join(WINDOW_METHODS) + " |", "|---" * (len(WINDOW_METHODS) + 1) + "|"]
    for a in r["config"]["attacks"]:
        lines.append(f"| {ATTACK_LABEL[a]} | " + " | ".join(f"{g('window', m, a, 'f1'):.3f}" for m in WINDOW_METHODS) + " |")
    lines.append("| Clean FPR | " + " | ".join(f"{g('window', m, 'clean', 'fpr'):.4f}" for m in WINDOW_METHODS) + " |")
    lines += ["", f"Throughput (pure Python, one core): {r['throughput_fps']['mean']:,.0f} frames/s", ""]
    return "\n".join(lines)


def latex_tables(r: dict, out_dir: str | Path) -> None:
    """Write the paper's LaTeX tables (booktabs) from results.json."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    M, L = r["metrics"], r["latency_ms"]
    g = lambda *k: M["|".join(k)]["mean"]
    lab = {k: v.replace("masq.", "masq.") for k, v in ATTACK_LABEL.items()}
    atk = r["config"]["attacks"]

    rows = []
    for a in atk:
        lat = L.get(f"cantinel|{a}", {}).get("mean")
        if g("frame", "cantinel", a, "recall") < 0.5:
            lat = None  # latency is meaningless when most bursts are missed
        cells = [f"{g('frame', m, a, k):.3f}" for m in ("interval", "cantinel") for k in ("precision", "recall", "f1")]
        rows.append(f"{lab[a]} & " + " & ".join(cells) + f" & {lat:.1f}" if lat is not None else
                    f"{lab[a]} & " + " & ".join(cells) + " & --")
    fpr_i, fpr_c = g("frame", "interval", "clean", "fpr"), g("frame", "cantinel", "clean", "fpr")
    (out / "tab_frame.tex").write_text(
        "\\begin{tabular}{@{}lccccccc@{}}\n\\toprule\n"
        " & \\multicolumn{3}{c}{Interval baseline~\\cite{song2016}} & \\multicolumn{3}{c}{CANtinel (ours)} & Latency\\\\\n"
        "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
        "Attack & P & R & F1 & P & R & F1 & (ms)\\\\\n\\midrule\n" + "\\\\\n".join(rows) + "\\\\\n\\midrule\n"
        f"Clean FPR & \\multicolumn{{3}}{{c}}{{{fpr_i:.1e}}} & \\multicolumn{{3}}{{c}}{{{fpr_c:.1e}}} & \\\\\n"
        "\\bottomrule\n\\end{tabular}\n")

    rows = []
    for a in atk:
        rows.append(f"{lab[a]} & " + " & ".join(f"{g('ablation', k, a, 'f1'):.3f}" for k in ABLATIONS))
    (out / "tab_ablation.tex").write_text(
        "\\begin{tabular}{@{}lcccc@{}}\n\\toprule\n"
        "Attack & L1 & +L2 & +L3 & All\\\\\n\\midrule\n" + "\\\\\n".join(rows) +
        "\\\\\n\\bottomrule\n\\end{tabular}\n")

    G = r["gateway"]
    rows = [f"{lab[a]} & {100 * G[a]['attack_block_rate']['mean']:.1f} & {100 * G[a]['legit_pass_rate']['mean']:.2f}"
            for a in atk]
    (out / "tab_gateway.tex").write_text(
        "\\begin{tabular}{@{}lcc@{}}\n\\toprule\n"
        "Attack & Blocked (\\%) & Legit passed (\\%)\\\\\n\\midrule\n" + "\\\\\n".join(rows) +
        "\\\\\n\\bottomrule\n\\end{tabular}\n")

    S = r["secoc"]
    sc = [("Legitimate PDUs", S["legitimate"]), ("Legitimate, 10\\,\\% frame loss", S["legitimate_10pct_loss"]),
          ("Replay of sniffed PDUs", S["replay"]), ("Spoofing, random MAC", S["spoof_random_mac"]),
          ("Stealth masq., re-used trailer", S["stealth_masquerade_reuse_trailer"]),
          ("Stealth masq., guessed FV", S["stealth_masquerade_next_fv"]),
          ("MAC brute force (one PDU)", {"accepted": S["brute_force"]["accepted"], "sent": S["brute_force"]["tries"]})]
    num = lambda x: f"{x:,}".replace(",", "{,}")
    rows = [f"{n} & {num(d['accepted'])} / {num(d['sent'])}" for n, d in sc]
    (out / "tab_secoc.tex").write_text(
        "\\begin{tabular}{@{}lr@{}}\n\\toprule\n"
        "Scenario & Accepted / sent\\\\\n\\midrule\n" + "\\\\\n".join(rows) +
        "\\\\\n\\bottomrule\n\\end{tabular}\n")
