"""Readers / writers for common CAN log formats.

Supported inputs
----------------
* CANtinel CSV       ``timestamp,can_id,dlc,data,label,attack``
* HCRL Car-Hacking   ``timestamp,ID,DLC,DATA0..DATA[dlc-1],Flag`` (Flag T/R)
* HCRL normal_run    ``Timestamp: 1479121434.850202  ID: 0350  000  DLC: 8  05 28 ...``
* candump -l         ``(1436509052.249713) vcan0 044#2A366C2BBA``

``load()`` sniffs the format automatically.
"""
from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Iterator

from .frame import ATTACK, NORMAL, CANFrame

_CANDUMP = re.compile(r"^\((\d+\.\d+)\)\s+\S+\s+([0-9A-Fa-f]+)#([0-9A-Fa-f]*)")
_HCRL_TXT = re.compile(r"Timestamp:\s*([\d.]+)\s+ID:\s*([0-9a-fA-F]+)\s+\d+\s+DLC:\s*(\d+)\s*(.*)")


def save_csv(frames: Iterable[CANFrame], path: str | Path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "can_id", "dlc", "data", "label", "attack"])
        for f in frames:
            w.writerow([f"{f.timestamp:.6f}", f"{f.can_id:03X}", f.dlc, f.data.hex().upper(), f.label, f.attack])


def _iter_native(path: Path) -> Iterator[CANFrame]:
    with open(path, newline="") as fh:
        r = csv.DictReader(fh)
        for row in r:
            yield CANFrame(float(row["timestamp"]), int(row["can_id"], 16), int(row["dlc"]),
                           bytes.fromhex(row["data"]), int(row.get("label") or 0), row.get("attack") or "")


def _iter_hcrl_csv(path: Path, attack_name: str) -> Iterator[CANFrame]:
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if not row or not row[0][:1].isdigit():
                continue
            dlc = int(row[2])
            data = bytes(int(x, 16) for x in row[3:3 + dlc])
            flag = row[3 + dlc].strip() if len(row) > 3 + dlc else "R"
            lab = ATTACK if flag == "T" else NORMAL
            yield CANFrame(float(row[0]), int(row[1], 16), dlc, data, lab, attack_name if lab else "")


def _iter_regex(path: Path) -> Iterator[CANFrame]:
    with open(path) as fh:
        for line in fh:
            m = _CANDUMP.match(line.strip())
            if m:
                data = bytes.fromhex(m.group(3))
                yield CANFrame(float(m.group(1)), int(m.group(2), 16), len(data), data)
                continue
            m = _HCRL_TXT.search(line)
            if m:
                dlc = int(m.group(3))
                data = bytes(int(x, 16) for x in m.group(4).split()[:dlc])
                yield CANFrame(float(m.group(1)), int(m.group(2), 16), dlc, data)


def sniff_format(path: str | Path) -> str:
    with open(path) as fh:
        head = "".join(fh.readline() for _ in range(3))
    if head.startswith("timestamp,can_id"):
        return "cantinel"
    if head.startswith("(") and "#" in head:
        return "candump"
    if "Timestamp:" in head:
        return "hcrl_txt"
    return "hcrl_csv"


def load(path: str | Path, fmt: str | None = None, limit: int | None = None) -> list[CANFrame]:
    path = Path(path)
    fmt = fmt or sniff_format(path)
    if fmt == "cantinel":
        it = _iter_native(path)
    elif fmt == "hcrl_csv":
        it = _iter_hcrl_csv(path, path.stem)
    elif fmt in ("candump", "hcrl_txt"):
        it = _iter_regex(path)
    else:
        raise ValueError(f"unknown format {fmt}")
    out = []
    for f in it:
        out.append(f)
        if limit and len(out) >= limit:
            break
    out.sort(key=lambda f: f.timestamp)
    return out
