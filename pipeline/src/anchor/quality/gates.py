"""The quality-gate cascade.

Every episode passes an ordered list of gates before it can be enriched.
Gates are cheap-first: a clip that's 0.4 s long shouldn't cost a GPU minute
to reject. Each gate returns pass/fail plus a score, and everything is logged
onto the Episode — rejections are kept for audit, never silently dropped.

The gates in this module are the *open* tier: container-level checks any
pipeline needs. The provenance and learned-quality gates (synthetic-content
screening, aesthetic/usability scoring, near-duplicate detection) are part of
Anchor's closed Trust stage — their interfaces live in :mod:`anchor.trust`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from anchor.ingest.intake import VideoMeta
from anchor.schema.episode import Episode


@dataclass(frozen=True)
class GateResult:
    passed: bool
    score: float          # 0..1, gate-specific meaning
    note: str = ""


class Gate(Protocol):
    name: str
    def check(self, meta: VideoMeta) -> GateResult: ...


@dataclass(frozen=True)
class DurationGate:
    name: str = "duration"
    min_s: float = 2.0
    max_s: float = 120.0

    def check(self, meta: VideoMeta) -> GateResult:
        ok = self.min_s <= meta.duration_s <= self.max_s
        return GateResult(ok, 1.0 if ok else 0.0,
                          f"{meta.duration_s:.1f}s not in [{self.min_s}, {self.max_s}]" if not ok else "")


@dataclass(frozen=True)
class ResolutionGate:
    name: str = "resolution"
    min_height: int = 480

    def check(self, meta: VideoMeta) -> GateResult:
        ok = meta.height >= self.min_height
        return GateResult(ok, min(1.0, meta.height / 1080),
                          f"{meta.height}p < {self.min_height}p" if not ok else "")


@dataclass(frozen=True)
class FrameRateGate:
    name: str = "fps"
    min_fps: float = 15.0

    def check(self, meta: VideoMeta) -> GateResult:
        ok = meta.fps >= self.min_fps
        return GateResult(ok, min(1.0, meta.fps / 30.0),
                          f"{meta.fps}fps < {self.min_fps}" if not ok else "")


def default_open_gates() -> list[Gate]:
    """The open-tier cascade, cheap-first. Thresholds here are θ_proc parameters —
    the closed-loop optimizer tunes them between rounds."""
    return [DurationGate(), ResolutionGate(), FrameRateGate()]


class GateCascade:
    def __init__(self, gates: list[Gate]):
        self.gates = gates

    def run(self, meta: VideoMeta, episode: Episode | None = None) -> bool:
        """Run gates in order; short-circuit on first failure. Logs to episode if given."""
        for gate in self.gates:
            r = gate.check(meta)
            if episode is not None:
                episode.log_gate(gate.name, r.passed, r.score, r.note)
            if not r.passed:
                return False
        return True
