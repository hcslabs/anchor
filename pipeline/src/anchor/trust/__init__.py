"""Trust: provenance scoring, synthetic-content screening, dedup. CLOSED.

The trust tier ladder (T1 verified-capture → T4 labeled-synthetic) is public —
see :class:`anchor.schema.episode.TrustTier` and the project page. How a tier
is *assigned* — the provenance classifier, the synthetic-content screens, the
near-duplicate index — is Anchor's core IP and is not in the public SDK.

These modules exist so downstream code can compile against the interfaces.
"""
