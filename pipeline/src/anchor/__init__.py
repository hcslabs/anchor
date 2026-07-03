"""Anchor — a trust-graded data refinery for humanoid robot learning.

Anchor turns ordinary human video into provenance-verified, feasibility-checked,
robot-ready training data. The pipeline runs seven stages:

    Intake -> Trust -> Gates -> Enrich -> Retarget -> Package -> Serve

This public SDK exposes the *open* stages — source connectors, media probing,
content addressing, the quality-gate scaffold, text-prompted person
segmentation, and the Noise-as-Signal residual sampler (under internal
validation). The Trust
scoring internals and the retargeting correspondence method remain closed; the
modules exist so downstream code can compile against their interfaces.

Quickstart::

    pip install -e ".[ingest,segment]"

    from anchor.ingest import probe, LocalFolderConnector
    from anchor.quality import GateCascade, default_open_gates

    meta = probe("clip.mp4")
    verdict = GateCascade(default_open_gates()).run(meta)

See ``examples/`` for runnable end-to-end scripts.
"""

from anchor.ids import content_id
from anchor.schema.episode import Episode, EpisodeStatus

__version__ = "0.3.0a0"
__all__ = ["content_id", "Episode", "EpisodeStatus", "__version__"]
