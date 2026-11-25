"""Command line interface: ``cantinel <command> --help``."""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np


def _layers(s: str):
    return tuple(x.strip() for x in s.split(",") if x.strip())


def cmd_simulate(a):
    from .io import save_csv
    from .simulator import simulate
    frames = simulate(a.duration, a.attack, seed=a.seed, vehicle_seed=a.vehicle_seed)
    save_csv(frames, a.out)
    n_atk = sum(f.label for f in frames)
    print(f"wrote {len(frames):,} frames ({n_atk:,} injected) to {a.out}")


def cmd_train(a):
    from .engine import Profile
    from .io import load
    tr = load(a.input)
    va = load(a.validation) if a.validation else None
    prof = Profile.train(tr, va)
    prof.save(a.out)
    print(prof.summary())
    if va:
        print(f"self-tuning on validation capture pruned: {prof.meta.get('pruned_alerts')}")
    print(f"profile saved to {a.out}")


def cmd_detect(a):
    from .engine import Engine, Profile
    from .io import load
    from .metrics import detection_latency, prf
    prof = Profile.load(a.profile)
    frames = load(a.input)
    eng = Engine(prof, layers=_layers(a.layers))
    t = time.perf_counter()
    flags, alerts = eng.run(frames)
    dt = time.perf_counter() - t
    kinds = Counter(x.kind for x in alerts)
    print(f"{len(frames):,} frames in {dt:.2f}s ({len(frames) / dt:,.0f} frames/s); "
          f"{int(flags.sum()):,} flagged; alerts: {dict(kinds)}")
    for x in [x for x in alerts if x.flags][: a.show]:
        print(f"  {x.timestamp:10.4f}  0x{x.can_id:03X}  {x.kind:<16} {x.message}")
    y = np.array([f.label for f in frames])
    if y.any():
        m = prf(y, flags)
        lat = detection_latency(frames, flags)
        print(f"ground truth available -> precision {m['precision']:.4f}  recall {m['recall']:.4f}  "
              f"F1 {m['f1']:.4f}  FPR {m['fpr']:.2e}" + (f"  latency {lat:.2f} ms" if lat is not None else ""))
    if a.alerts:
        with open(a.alerts, "w") as fh:
            for x in alerts:
                fh.write(json.dumps(x.to_dict()) + "\n")
        print(f"alerts written to {a.alerts}")


def cmd_gateway(a):
    from .engine import Engine, Profile
    from .gateway import SecureGateway
    from .io import load, save_csv
    prof = Profile.load(a.profile)
    frames = load(a.input)
    out, st = SecureGateway(Engine(prof)).run(frames)
    d = st.as_dict()
    print(json.dumps(d, indent=1))
    if a.out:
        save_csv(out, a.out)
        print(f"forwarded traffic written to {a.out}")


def cmd_secoc(a):
    from .experiments import secoc_experiment
    print(json.dumps(secoc_experiment(), indent=1))


def cmd_benchmark(a):
    from .experiments import markdown_tables, run_suite
    r = run_suite(seeds=tuple(a.seeds), out_dir=a.out, train_s=a.train, val_s=a.val, test_s=a.test)
    md = markdown_tables(r)
    Path(a.out, "tables.md").write_text(md)
    print(md)
    try:
        from .figures import make_all
        make_all(r, a.figures)
        print(f"figures written to {a.figures}")
    except ImportError as e:  # matplotlib missing
        print(f"(figures skipped: {e})")


def cmd_demo(a):
    from .live import demo
    demo(a.attack, a.duration, a.speed, a.seed, color=not a.no_color, verbose=a.verbose)


def cmd_live(a):
    from .engine import Profile
    from .live import monitor_bus
    monitor_bus(Profile.load(a.profile), a.interface, a.channel, a.bitrate, a.alerts, color=not a.no_color,
                verbose=a.verbose)


def main(argv=None):
    from .simulator import ATTACK_KINDS
    p = argparse.ArgumentParser(prog="cantinel",
                                description="CANtinel - open, explainable intrusion detection & prevention for CAN")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("simulate", help="generate a labelled CAN capture")
    s.add_argument("--duration", type=float, default=60)
    s.add_argument("--attack", choices=ATTACK_KINDS, default=None)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--vehicle-seed", type=int, default=7)
    s.add_argument("-o", "--out", required=True)
    s.set_defaults(fn=cmd_simulate)

    s = sub.add_parser("train", help="learn a vehicle profile from attack-free traffic")
    s.add_argument("-i", "--input", required=True, help="CSV / HCRL / candump log")
    s.add_argument("-v", "--validation", help="second attack-free capture for self-tuning")
    s.add_argument("-o", "--out", default="profile.json")
    s.set_defaults(fn=cmd_train)

    s = sub.add_parser("detect", help="run the IDS on a capture")
    s.add_argument("-p", "--profile", required=True)
    s.add_argument("-i", "--input", required=True)
    s.add_argument("--layers", default="spec,timing,payload")
    s.add_argument("--alerts", help="write JSON-lines alerts here")
    s.add_argument("--show", type=int, default=10)
    s.set_defaults(fn=cmd_detect)

    s = sub.add_parser("gateway", help="filter a capture through the secure gateway (IPS)")
    s.add_argument("-p", "--profile", required=True)
    s.add_argument("-i", "--input", required=True)
    s.add_argument("-o", "--out")
    s.set_defaults(fn=cmd_gateway)

    s = sub.add_parser("secoc-demo", help="SecOC-lite authentication experiment")
    s.set_defaults(fn=cmd_secoc)

    s = sub.add_parser("benchmark", help="reproduce the paper's experiments")
    s.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    s.add_argument("--train", type=float, default=300)
    s.add_argument("--val", type=float, default=120)
    s.add_argument("--test", type=float, default=60)
    s.add_argument("--out", default="results")
    s.add_argument("--figures", default="docs/figures")
    s.set_defaults(fn=cmd_benchmark)

    s = sub.add_parser("demo", help="live classroom demo on a virtual CAN bus")
    s.add_argument("--attack", choices=ATTACK_KINDS + ["none"], default="masquerade")
    s.add_argument("--duration", type=float, default=40)
    s.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--no-color", action="store_true")
    s.add_argument("--verbose", action="store_true", help="also show low-severity notices")
    s.set_defaults(fn=cmd_demo)

    s = sub.add_parser("live", help="monitor a real CAN interface (python-can)")
    s.add_argument("-p", "--profile", required=True)
    s.add_argument("--interface", default="socketcan")
    s.add_argument("--channel", default="vcan0")
    s.add_argument("--bitrate", type=int)
    s.add_argument("--alerts")
    s.add_argument("--no-color", action="store_true")
    s.add_argument("--verbose", action="store_true", help="also show low-severity notices")
    s.set_defaults(fn=cmd_live)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
