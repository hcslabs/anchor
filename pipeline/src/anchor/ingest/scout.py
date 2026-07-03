"""Scout: agentic, *scoped* data acquisition.

Anchor never bulk-crawls. A scout is given a scope — task class, conditions,
place — and returns candidate clips from sources we are allowed to use. The
scope object is part of the tunable parameter set θ_scope: the closed-loop
optimizer (see the project page) adjusts it between rounds based on the
measured sim-to-real gap.

Two connector families ship in the public SDK:

* :class:`LocalFolderConnector` — on-site footage, partner drops, mocap
  exports. The bread and butter of enterprise engagements.
* :class:`PexelsVideoConnector` — an example of a *licensed-API* connector.
  Pexels serves free-licensed stock video through a documented REST API;
  the connector shows the shape every remote connector follows: query →
  paginate → yield ``Candidate`` with the license recorded up front.

What is deliberately NOT here: scrapers for platforms whose terms prohibit it.
Provenance is a first-class field in Anchor, and it starts at acquisition —
a clip whose license you can't state is a clip the Trust stage will reject
anyway, so we don't spend bandwidth acquiring it.
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi"}


@dataclass(frozen=True)
class Scope:
    """What the scout is looking for. Every field is optimizer-tunable (θ_scope)."""
    task_class: str                      # e.g. "bimanual_pick", "stair_ascent"
    query_terms: tuple[str, ...]         # free-text expansion of the task
    min_duration_s: float = 2.0
    max_duration_s: float = 90.0
    min_height_px: int = 480
    max_candidates: int = 200


@dataclass(frozen=True)
class Candidate:
    """A clip the scout proposes for ingestion. License is recorded at birth."""
    uri: str
    source: str                          # connector name
    license: str                         # e.g. "pexels", "partner-contract", "onsite"
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    meta: dict = field(default_factory=dict)


class SourceConnector(ABC):
    """One acquisition source. Implementations must be polite and resumable."""

    name: str = "abstract"

    @abstractmethod
    def search(self, scope: Scope) -> Iterator[Candidate]:
        """Yield candidates matching *scope*, best-first, up to scope.max_candidates."""


class LocalFolderConnector(SourceConnector):
    """Recursively yield video files under a root — on-site footage, partner drops."""

    name = "local"

    def __init__(self, root: str | Path, license: str = "onsite"):
        self.root = Path(root)
        self.license = license

    def search(self, scope: Scope) -> Iterator[Candidate]:
        n = 0
        for p in sorted(self.root.rglob("*")):
            if p.suffix.lower() not in VIDEO_SUFFIXES:
                continue
            yield Candidate(uri=p.resolve().as_uri(), source=self.name, license=self.license)
            n += 1
            if n >= scope.max_candidates:
                return


class PexelsVideoConnector(SourceConnector):
    """Licensed stock-video search via the Pexels REST API.

    Requires a free API key: https://www.pexels.com/api/ — set ``PEXELS_API_KEY``.
    Rate limits are respected (the API returns X-Ratelimit headers; we also
    self-throttle). All results carry the Pexels license, recorded on the
    Candidate so the Trust stage can verify it later.
    """

    name = "pexels"
    _ENDPOINT = "https://api.pexels.com/videos/search"
    _PER_PAGE = 25
    _MIN_INTERVAL_S = 0.75          # self-imposed politeness floor

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.environ.get("PEXELS_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("PexelsVideoConnector needs PEXELS_API_KEY")
        self._last_call = 0.0

    def _throttle(self) -> None:
        wait = self._MIN_INTERVAL_S - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    def search(self, scope: Scope) -> Iterator[Candidate]:
        import requests  # optional dep: pip install "anchor-pipeline[ingest]"

        yielded = 0
        for term in scope.query_terms:
            page = 1
            while yielded < scope.max_candidates:
                self._throttle()
                resp = requests.get(
                    self._ENDPOINT,
                    headers={"Authorization": self.api_key},
                    params={"query": term, "per_page": self._PER_PAGE, "page": page},
                    timeout=30,
                )
                resp.raise_for_status()
                body = resp.json()
                videos = body.get("videos", [])
                if not videos:
                    break
                for v in videos:
                    dur = float(v.get("duration", 0))
                    if not (scope.min_duration_s <= dur <= scope.max_duration_s):
                        continue
                    files = sorted(v.get("video_files", []),
                                   key=lambda f: f.get("height") or 0, reverse=True)
                    best = next((f for f in files
                                 if (f.get("height") or 0) >= scope.min_height_px), None)
                    if best is None:
                        continue
                    yield Candidate(
                        uri=best["link"], source=self.name, license="pexels",
                        duration_s=dur, width=best.get("width"), height=best.get("height"),
                        meta={"pexels_id": v.get("id"), "query": term},
                    )
                    yielded += 1
                    if yielded >= scope.max_candidates:
                        return
                if body.get("next_page") is None:
                    break
                page += 1
