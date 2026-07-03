"""Stage registry: the pipeline's seven stages as a declared, ordered graph.

Stages register themselves with :func:`register`; the runner resolves them by
name. Keeping the graph declarative means the closed-loop optimizer can treat
"which stages run, with which θ" as data — a round of the data-tuning loop is
literally a different registry configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

STAGE_ORDER = ("intake", "trust", "gates", "enrich", "retarget", "package", "serve")


@dataclass(frozen=True)
class StageSpec:
    name: str
    fn: Callable
    open_source: bool     # is the implementation public in this SDK?
    doc: str = ""


_REGISTRY: dict[str, StageSpec] = {}


def register(name: str, *, open_source: bool, doc: str = ""):
    if name not in STAGE_ORDER:
        raise ValueError(f"unknown stage {name!r}; must be one of {STAGE_ORDER}")
    def deco(fn: Callable) -> Callable:
        _REGISTRY[name] = StageSpec(name=name, fn=fn, open_source=open_source, doc=doc)
        return fn
    return deco


def get(name: str) -> StageSpec:
    return _REGISTRY[name]


def registered() -> list[StageSpec]:
    return [_REGISTRY[n] for n in STAGE_ORDER if n in _REGISTRY]
