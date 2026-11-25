"""SecOC-lite: AUTOSAR-SecOC-style message authentication for CAN / CAN FD.

Each protected PDU carries

    payload || truncated freshness value (8 bit) || truncated AES-128-CMAC (24 bit)

which corresponds to the widely used AUTOSAR SecOC profile "24Bit-CMAC-8Bit-FV".
The MAC is computed over ``DataID || payload || full 64-bit freshness value``;
the receiver reconstructs the full freshness value from the truncated one and
its last accepted value, so lost frames are tolerated while replayed frames
(stale freshness) and forged frames (no key) are rejected.

This closes the gap that no anomaly-based IDS can close: an attacker who has
reverse-engineered counters and checksums and transmits plausible values at
the correct period (``stealth_masquerade``).
"""
from __future__ import annotations

import hashlib
import hmac
import os
import time
from collections import deque
from dataclasses import dataclass

from cryptography.hazmat.primitives.ciphers import algorithms
from cryptography.hazmat.primitives.cmac import CMAC

from .frame import Alert, CANFrame


def cmac_aes128(key: bytes, msg: bytes) -> bytes:
    c = CMAC(algorithms.AES(key))
    c.update(msg)
    return c.finalize()


@dataclass
class SecOCConfig:
    mac_bits: int = 24
    fv_bits: int = 8
    acceptance_window: int = 64      # max freshness jump accepted (lost frames)

    @property
    def overhead(self) -> int:
        return (self.mac_bits + self.fv_bits) // 8


class KeyStore:
    """Toy key server: one 128-bit key per protected CAN ID (Data ID).

    In a vehicle the keys live in the ECU's HSM and are provisioned at the
    factory / by the OEM backend; here they are generated with os.urandom.
    """

    def __init__(self, ids, seed: bytes | None = None):
        # with a seed, keys are derived deterministically (demo / tests only):
        # K_id = CMAC(SHA-256(seed)[:16], "CANtinel-KDF" || id)
        master = hashlib.sha256(seed).digest()[:16] if seed is not None else None
        self.keys = {cid: (cmac_aes128(master, b"CANtinel-KDF" + cid.to_bytes(2, "big")) if master else os.urandom(16))
                     for cid in ids}

    def __getitem__(self, cid: int) -> bytes:
        return self.keys[cid]


class SecOCSender:
    def __init__(self, key: bytes, data_id: int, cfg: SecOCConfig | None = None):
        self.key, self.data_id = key, data_id
        self.cfg = cfg or SecOCConfig()
        self.fv = 0

    def protect(self, payload: bytes) -> bytes:
        self.fv += 1
        mac = cmac_aes128(self.key, self.data_id.to_bytes(2, "big") + payload + self.fv.to_bytes(8, "big"))
        tfv = self.fv & ((1 << self.cfg.fv_bits) - 1)
        return payload + tfv.to_bytes(self.cfg.fv_bits // 8, "big") + mac[: self.cfg.mac_bits // 8]


class SecOCReceiver:
    def __init__(self, key: bytes, data_id: int, cfg: SecOCConfig | None = None):
        self.key, self.data_id = key, data_id
        self.cfg = cfg or SecOCConfig()
        self.last_fv = 0

    def _reconstruct(self, tfv: int) -> int:
        bits = self.cfg.fv_bits
        mask = (1 << bits) - 1
        upper, lower = self.last_fv >> bits, self.last_fv & mask
        return (upper << bits) | tfv if tfv > lower else ((upper + 1) << bits) | tfv

    def verify(self, pdu: bytes) -> tuple[bool, bytes, str]:
        cfg = self.cfg
        nm, nf = cfg.mac_bits // 8, cfg.fv_bits // 8
        if len(pdu) <= nm + nf:
            return False, b"", "PDU_TOO_SHORT"
        payload = pdu[: -(nm + nf)]
        tfv = int.from_bytes(pdu[-(nm + nf): -nm], "big")
        tmac = pdu[-nm:]
        fv = self._reconstruct(tfv)
        if fv - self.last_fv > cfg.acceptance_window:
            return False, payload, "FRESHNESS_OUT_OF_WINDOW"
        mac = cmac_aes128(self.key, self.data_id.to_bytes(2, "big") + payload + fv.to_bytes(8, "big"))
        if not hmac.compare_digest(mac[:nm], tmac):
            return False, payload, "MAC_MISMATCH"
        self.last_fv = fv
        return True, payload, "OK"


class AuthLayer:
    """Engine layer: verifies SecOC PDUs of protected IDs and detects MAC brute forcing."""

    name = "auth"

    def __init__(self, keys: KeyStore, cfg: SecOCConfig | None = None, brute_force_per_s: int = 5):
        self.cfg = cfg or SecOCConfig()
        self.rx = {cid: SecOCReceiver(k, cid, self.cfg) for cid, k in keys.keys.items()}
        self.fails = deque()
        self.bf = brute_force_per_s
        self._bf_until = -1.0

    def check(self, f: CANFrame) -> list[Alert]:
        rx = self.rx.get(f.can_id)
        if rx is None:
            return []
        ok, _, reason = rx.verify(f.data)
        if ok:
            return []
        out = [Alert(f.timestamp, f.can_id, "AUTH_FAIL", self.name, "high", f"SecOC verification failed: {reason}")]
        self.fails.append(f.timestamp)
        while self.fails and self.fails[0] < f.timestamp - 1.0:
            self.fails.popleft()
        if len(self.fails) > self.bf and f.timestamp > self._bf_until:
            out.append(Alert(f.timestamp, f.can_id, "MAC_BRUTE_FORCE", self.name, "high",
                             f"{len(self.fails)} authentication failures in 1 s - possible MAC forging attempt"))
            self._bf_until = f.timestamp + 1.0
        return out

    def commit(self, f: CANFrame) -> None:
        pass


def benchmark(n: int = 20000) -> dict:
    key = os.urandom(16)
    tx, rx = SecOCSender(key, 0x4B0), SecOCReceiver(key, 0x4B0)
    payload = bytes(8)
    t0 = time.perf_counter()
    pdus = [tx.protect(payload) for _ in range(n)]
    t1 = time.perf_counter()
    for p in pdus:
        rx.verify(p)
    t2 = time.perf_counter()
    return {"protect_us": (t1 - t0) / n * 1e6, "verify_us": (t2 - t1) / n * 1e6}
