"""Closed stage — interface stub. See :mod:`anchor.trust` and pipeline/README.md."""

def score(*args, **kwargs):
    raise NotImplementedError(
        "anchor.trust.synthetic is part of Anchor's closed Trust stage. "
        "The tier ladder is public (anchor.schema.episode.TrustTier); "
        "the scoring method is not. Contact HCS Labs for partner access."
    )
