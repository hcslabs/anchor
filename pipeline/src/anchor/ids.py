"""Content addressing.

Every asset that enters Anchor gets an identity derived from its bytes, not its
filename or URL. This is what makes dedup, provenance, and re-ingestion
idempotent: the same clip arriving twice — from two scouts, two mirrors, or two
years apart — resolves to the same ``content_id`` and is processed once.

IDs are of the form ``anc1-<sha256-hex>`` (the prefix versions the scheme).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import BinaryIO, Union

_SCHEME = "anc1"
_CHUNK = 1 << 20  # 1 MiB


def content_id(source: Union[str, Path, bytes, BinaryIO]) -> str:
    """Return the stable content ID for a file path, raw bytes, or stream.

    >>> content_id(b"hello")
    'anc1-2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824'
    """
    h = hashlib.sha256()
    if isinstance(source, bytes):
        h.update(source)
    elif isinstance(source, (str, Path)):
        with open(source, "rb") as f:
            for chunk in iter(lambda: f.read(_CHUNK), b""):
                h.update(chunk)
    else:  # file-like
        for chunk in iter(lambda: source.read(_CHUNK), b""):
            h.update(chunk)
    return f"{_SCHEME}-{h.hexdigest()}"


def short_id(cid: str, n: int = 12) -> str:
    """Human-friendly truncation for logs and filenames: ``anc1-2cf24dba5fb0``."""
    scheme, digest = cid.split("-", 1)
    return f"{scheme}-{digest[:n]}"
