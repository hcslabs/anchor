"""Quality gates. See :mod:`anchor.quality.gates`."""
from anchor.quality.gates import (Gate, GateResult, GateCascade,
                                   DurationGate, ResolutionGate, FrameRateGate,
                                   default_open_gates)
__all__ = ["Gate", "GateResult", "GateCascade",
           "DurationGate", "ResolutionGate", "FrameRateGate", "default_open_gates"]
