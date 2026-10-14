#!/usr/bin/env python3
"""Train the Anchor phase policy from a provenance-bound trajectory."""

from __future__ import annotations

import argparse
import json

from anchor.policy.phase_policy import train_phase_policy


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trajectory", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    receipt = train_phase_policy(
        args.trajectory,
        args.output,
        epochs=args.epochs,
        device=args.device,
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

