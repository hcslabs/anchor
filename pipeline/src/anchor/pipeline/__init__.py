"""Stage registry + orchestration. See :mod:`anchor.pipeline.stages`."""
from anchor.pipeline.stages import STAGE_ORDER, StageSpec, register, get, registered
__all__ = ["STAGE_ORDER", "StageSpec", "register", "get", "registered"]
