"""``anchor`` command-line entry point.

    anchor probe clip.mp4          probe a file, print its metadata + content ID
    anchor gates clip.mp4          run the open gate cascade, print the verdict
    anchor stages                  list the seven stages and their open/closed status
"""

from __future__ import annotations

import argparse
import json
import sys


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="anchor", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("probe", help="probe a video file")
    sp.add_argument("path")
    sg = sub.add_parser("gates", help="run the open quality-gate cascade")
    sg.add_argument("path")
    sub.add_parser("stages", help="list pipeline stages")

    args = p.parse_args(argv)

    if args.cmd == "probe":
        from anchor.ingest import ingest
        ep = ingest(args.path)
        print(ep.to_json())
        return 0

    if args.cmd == "gates":
        from anchor.ingest import probe, ingest
        from anchor.quality import GateCascade, default_open_gates
        ep = ingest(args.path)
        ok = GateCascade(default_open_gates()).run(probe(args.path), ep)
        print(ep.to_json())
        return 0 if ok else 1

    if args.cmd == "stages":
        from anchor.pipeline.stages import STAGE_ORDER
        closed = {"trust", "retarget"}
        for i, name in enumerate(STAGE_ORDER, 1):
            mark = "closed" if name in closed else "open"
            print(f"  {i:02d}  {name:<10s} [{mark}]")
        return 0

    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except BrokenPipeError:            # piping to head/grep is normal usage
        sys.stderr.close()
        sys.exit(0)
