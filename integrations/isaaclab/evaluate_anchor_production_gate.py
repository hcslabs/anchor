#!/usr/bin/env python3
"""Combine Anchor lineage, policy, physics and task evidence into one gate receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from anchor.agent.production_loop import evaluate_production_candidate


def load(path: str | None) -> dict | None:
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lineage", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--physics", required=True)
    parser.add_argument("--task")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    receipt = evaluate_production_candidate(
        load(args.lineage) or {},
        load(args.policy),
        load(args.physics),
        load(args.task),
    )
    receipt["evidence"] = {
        "lineage": str(Path(args.lineage).resolve()),
        "policy": str(Path(args.policy).resolve()),
        "physics": str(Path(args.physics).resolve()),
        "task": str(Path(args.task).resolve()) if args.task else None,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
