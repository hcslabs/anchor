"""Ingest a local folder: probe every clip, run the open gate cascade, print verdicts."""
import sys
from pathlib import Path

from anchor.ingest import probe, ingest
from anchor.ingest.scout import Scope, LocalFolderConnector
from anchor.quality import GateCascade, default_open_gates

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
scope = Scope(task_class="unlabeled", query_terms=())
cascade = GateCascade(default_open_gates())

for cand in LocalFolderConnector(root).search(scope):
    path = cand.uri.removeprefix("file://")
    ep = ingest(path, source_uri=cand.uri)
    ok = cascade.run(probe(path), ep)
    print(f"{'PASS' if ok else 'REJECT':7s} {ep.duration_s:6.1f}s "
          f"{ep.width}x{ep.height}  {path}")
