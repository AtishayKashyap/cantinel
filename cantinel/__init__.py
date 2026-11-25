"""CANtinel - open, explainable intrusion detection & prevention for CAN networks."""
from .engine import Engine, Profile
from .frame import Alert, CANFrame
from .simulator import ATTACK_KINDS, simulate

__version__ = "1.0.0"
__all__ = ["Engine", "Profile", "Alert", "CANFrame", "simulate", "ATTACK_KINDS"]
