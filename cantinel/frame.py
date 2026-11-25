"""Core data types shared by every CANtinel module."""
from __future__ import annotations

from dataclasses import dataclass, field

NORMAL = 0
ATTACK = 1


@dataclass(slots=True)
class CANFrame:
    """One classic-CAN (or CAN FD) data frame as seen on the bus.

    ``label``/``attack`` hold ground truth when it is known (simulated or
    labelled datasets) and are ignored by the detectors.
    """

    timestamp: float
    can_id: int
    dlc: int
    data: bytes
    label: int = NORMAL
    attack: str = ""

    def hex_id(self) -> str:
        return f"{self.can_id:03X}"

    def hex_data(self) -> str:
        return self.data.hex().upper()

    def __str__(self) -> str:  # candump-like
        return f"({self.timestamp:.6f}) {self.hex_id()}#{self.hex_data()}"


@dataclass(slots=True)
class Alert:
    """An explainable detection result attached to a single frame."""

    timestamp: float
    can_id: int
    kind: str          # e.g. UNKNOWN_ID, TIMING_EARLY, COUNTER_BREAK
    layer: str         # spec | timing | payload | auth | ml | bus
    severity: str      # high | medium (flag the frame) | low | info (reported only)
    message: str
    extra: dict = field(default_factory=dict)

    @property
    def flags(self) -> bool:
        """high / medium alerts flag the frame; low / info alerts are only reported."""
        return self.severity in ("high", "medium")

    def to_dict(self) -> dict:
        return {
            "timestamp": round(self.timestamp, 6),
            "can_id": f"0x{self.can_id:03X}",
            "kind": self.kind,
            "layer": self.layer,
            "severity": self.severity,
            "message": self.message,
            **({"extra": self.extra} if self.extra else {}),
        }
