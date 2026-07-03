"""The Episode record — Anchor's unit of data.

An *episode* is one contiguous clip of human motion, tracked from raw intake
through to a packaged training sample. The record accumulates fields as it
moves down the pipeline; every stage appends, none overwrite. That append-only
discipline is what makes an Anchor dataset auditable: for any training sample
you can walk back to the exact source bytes, the trust decision, and every
gate it passed.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Optional


class EpisodeStatus(str, Enum):
    INGESTED = "ingested"          # bytes on disk, content-addressed
    TRUSTED = "trusted"            # provenance scored (closed stage)
    GATED = "gated"                # passed the quality-gate cascade
    ENRICHED = "enriched"          # pose / depth / contact / caption attached
    RETARGETED = "retargeted"      # mapped onto a target robot (closed stage)
    PACKAGED = "packaged"          # serialized into a training shard
    REJECTED = "rejected"          # failed a gate; kept for audit, never trained on


class TrustTier(str, Enum):
    """Provenance grade. How the tier is *assigned* is closed; the ladder is public."""
    T1_VERIFIED_CAPTURE = "t1"     # we recorded it, chain of custody intact
    T2_PARTNER_LICENSED = "t2"     # licensed source, contract on file
    T3_PUBLIC_SCREENED = "t3"      # public source, passed synthetic + rights screens
    T4_SYNTHETIC_LABELED = "t4"    # generated content, labeled as such, never mixed silently


@dataclass
class Episode:
    content_id: str
    source_uri: str
    status: EpisodeStatus = EpisodeStatus.INGESTED
    trust_tier: Optional[TrustTier] = None
    target_robot: Optional[str] = None            # e.g. "unitree_g1"
    task_class: Optional[str] = None              # e.g. "bimanual_pick", "stair_ascent"
    duration_s: Optional[float] = None
    fps: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    gate_log: list[dict[str, Any]] = field(default_factory=list)
    enrichments: dict[str, str] = field(default_factory=dict)   # name -> artifact path
    created_at: float = field(default_factory=time.time)

    def log_gate(self, gate: str, passed: bool, score: float | None = None, note: str = "") -> None:
        self.gate_log.append({"gate": gate, "passed": passed, "score": score,
                              "note": note, "at": time.time()})
        if not passed:
            self.status = EpisodeStatus.REJECTED

    def to_json(self) -> str:
        d = asdict(self)
        d["status"] = self.status.value
        d["trust_tier"] = self.trust_tier.value if self.trust_tier else None
        return json.dumps(d, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, raw: str) -> "Episode":
        d = json.loads(raw)
        d["status"] = EpisodeStatus(d["status"])
        if d.get("trust_tier"):
            d["trust_tier"] = TrustTier(d["trust_tier"])
        return cls(**d)
