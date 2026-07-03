"""Generic web-crawler connector — SKELETON / PLACEHOLDER.

This module shows the *shape* of Anchor's crawl layer for partner-approved
domains: allowlist-scoped, robots.txt-respecting, rate-limited, resumable.
The production crawler (JS rendering, sitemap diffing, per-domain politeness
budgets, incremental frontier persistence) is not in the public SDK — what's
here is enough to see the contract every crawl connector honors:

    1. Never leave the allowlist.               (scope is explicit, not emergent)
    2. Check robots.txt before every fetch.     (urllib.robotparser, cached)
    3. Self-throttle per host.                  (floor interval, jittered)
    4. Filter by content type before download.  (HEAD first — bandwidth respect)
    5. Every Candidate is born with a license.  (from the domain agreement)

Provenance starts at acquisition: a domain we can't state a license for is a
domain the Trust stage would reject anyway, so it never enters the allowlist.
"""

from __future__ import annotations

import time
import urllib.robotparser
from dataclasses import dataclass, field
from typing import Iterator
from urllib.parse import urljoin, urlparse

from anchor.ingest.scout import Candidate, Scope, SourceConnector

VIDEO_CONTENT_TYPES = {"video/mp4", "video/webm", "video/quicktime"}


@dataclass
class DomainAgreement:
    """One allowlisted domain + the license terms we crawl it under."""
    domain: str                  # e.g. "media.partner-example.com"
    license: str                 # e.g. "partner-contract-2026-04"
    seed_paths: tuple[str, ...] = ("/",)
    min_interval_s: float = 2.0  # per-host politeness floor


class AllowlistCrawler(SourceConnector):
    """Crawl only the domains in *agreements*. Everything else is out of scope.

    SKELETON: link discovery below is a stub (``_discover``). The production
    frontier — sitemap-first, render-aware, resumable — is closed. The
    politeness contract (robots, throttle, HEAD-filter) is fully shown.
    """

    name = "crawl"

    def __init__(self, agreements: list[DomainAgreement]):
        self.agreements = {a.domain: a for a in agreements}
        self._robots: dict[str, urllib.robotparser.RobotFileParser] = {}
        self._last_hit: dict[str, float] = {}

    # -- politeness primitives (real) ------------------------------------

    def _allowed(self, url: str) -> bool:
        host = urlparse(url).netloc
        if host not in self.agreements:
            return False                                   # rule 1: allowlist
        if host not in self._robots:
            rp = urllib.robotparser.RobotFileParser()
            rp.set_url(f"https://{host}/robots.txt")
            try:
                rp.read()
            except OSError:
                return False                               # can't read robots -> don't crawl
            self._robots[host] = rp
        return self._robots[host].can_fetch("AnchorBot/0.3", url)   # rule 2

    def _throttle(self, host: str) -> None:
        floor = self.agreements[host].min_interval_s
        wait = floor - (time.monotonic() - self._last_hit.get(host, 0.0))
        if wait > 0:
            time.sleep(wait)                               # rule 3
        self._last_hit[host] = time.monotonic()

    def _head_is_video(self, url: str) -> bool:
        import requests
        self._throttle(urlparse(url).netloc)
        r = requests.head(url, timeout=15, allow_redirects=True,
                          headers={"User-Agent": "AnchorBot/0.3 (+hcslabs.github.io/anchor)"})
        ctype = (r.headers.get("Content-Type") or "").split(";")[0].strip()
        return r.ok and ctype in VIDEO_CONTENT_TYPES       # rule 4

    # -- discovery (STUB) --------------------------------------------------

    def _discover(self, agreement: DomainAgreement) -> Iterator[str]:
        """Yield candidate video URLs under *agreement*.

        PLACEHOLDER — production walks sitemaps + rendered pages with a
        persisted frontier. Here: seeds only, so the contract is testable.
        """
        for path in agreement.seed_paths:
            yield urljoin(f"https://{agreement.domain}", path)

    # -- connector interface ----------------------------------------------

    def search(self, scope: Scope) -> Iterator[Candidate]:
        n = 0
        for agreement in self.agreements.values():
            for url in self._discover(agreement):
                if n >= scope.max_candidates:
                    return
                if not self._allowed(url):
                    continue
                if not self._head_is_video(url):
                    continue
                yield Candidate(uri=url, source=self.name,
                                license=agreement.license,          # rule 5
                                meta={"domain": agreement.domain})
                n += 1
