"""Small dependency-free CLI for simulator preparation steps."""

from __future__ import annotations

import argparse
import json

from anchor.sim.urdf_surrogate import build_collision_surrogate


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    surrogate = subparsers.add_parser("build-surrogate")
    surrogate.add_argument("--source", required=True)
    surrogate.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "build-surrogate":
        print(json.dumps(build_collision_surrogate(args.source, args.output), indent=2))
        return 0
    raise RuntimeError("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())

