"""Checksum algorithms commonly found in automotive CAN payloads.

The payload layer of CANtinel *learns* which (if any) of these algorithms
protects each CAN ID, so the user does not need the OEM's DBC file.
"""
from __future__ import annotations


def xor8(data: bytes) -> int:
    r = 0
    for b in data:
        r ^= b
    return r


def sum8(data: bytes) -> int:
    return sum(data) & 0xFF


def _make_crc8_table(poly: int) -> list[int]:
    table = []
    for i in range(256):
        crc = i
        for _ in range(8):
            crc = ((crc << 1) ^ poly) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
        table.append(crc)
    return table


_J1850 = _make_crc8_table(0x1D)


def crc8_j1850(data: bytes) -> int:
    """CRC-8 SAE J1850 (poly 0x1D, init 0xFF, xorout 0xFF) - used by AUTOSAR E2E P01."""
    crc = 0xFF
    for b in data:
        crc = _J1850[crc ^ b]
    return crc ^ 0xFF


CHECKSUMS = {"xor": xor8, "sum": sum8, "crc8": crc8_j1850}


def compute(algo: str, data: bytes, position: int) -> int:
    """Checksum over every byte except the one at ``position``."""
    body = data[:position] + data[position + 1:]
    return CHECKSUMS[algo](body)
