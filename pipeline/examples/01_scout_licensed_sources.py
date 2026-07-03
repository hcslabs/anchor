"""Scout a scoped set of candidate clips from a licensed source.

Requires: pip install "anchor-pipeline[ingest]"  and  PEXELS_API_KEY in env.
"""
from anchor.ingest.scout import Scope, PexelsVideoConnector

scope = Scope(
    task_class="box_carry",
    query_terms=("person carrying box warehouse", "worker lifting package"),
    min_duration_s=3.0,
    max_candidates=25,
)

scout = PexelsVideoConnector()
for cand in scout.search(scope):
    print(f"{cand.duration_s:5.1f}s  {cand.width}x{cand.height}  "
          f"[{cand.license}]  {cand.uri[:80]}")
