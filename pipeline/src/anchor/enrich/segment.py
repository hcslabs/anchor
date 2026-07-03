"""Text-prompted person segmentation (CLIP-based).

The Enrich stage's first job is isolating the human from everything else in
frame, because every downstream signal — pose, contact, hands — conditions on
that mask. In production we run a video-native tracker for temporal
consistency; the public SDK ships the *per-frame* segmenter, built on CLIPSeg
(Lüddecke & Ecker, CVPR 2022, ``CIDAS/clipseg-rd64-refined``), because it is
small, permissively licensed, and prompt-driven — the same property our
production stack relies on: the segmentation target is *text*, so the same
code isolates "a person carrying a box" today and "a person on a ladder"
tomorrow with zero retraining.

Requires the ``segment`` extra: ``pip install "anchor-pipeline[segment]"``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass
class SegmentResult:
    mask: np.ndarray            # (H, W) float32 in [0, 1]
    prompt: str
    coverage: float             # fraction of pixels above threshold — a cheap gate signal


class PersonSegmenter:
    """Prompt-driven segmenter. One instance holds the model; call per frame.

    >>> seg = PersonSegmenter()                       # doctest: +SKIP
    >>> r = seg(frame_rgb, "a person walking")        # doctest: +SKIP
    >>> r.mask.shape, r.coverage                      # doctest: +SKIP
    """

    DEFAULT_PROMPT = "a person"
    _MODEL = "CIDAS/clipseg-rd64-refined"

    def __init__(self, device: str | None = None, threshold: float = 0.4):
        try:
            import torch
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                'segmentation needs the "segment" extra: '
                'pip install "anchor-pipeline[segment]"'
            ) from e
        self._torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.threshold = threshold
        self.processor = CLIPSegProcessor.from_pretrained(self._MODEL)
        self.model = CLIPSegForImageSegmentation.from_pretrained(self._MODEL).to(self.device).eval()

    @staticmethod
    def _to_pil(frame_rgb: np.ndarray):
        from PIL import Image
        return Image.fromarray(frame_rgb.astype(np.uint8))

    def __call__(self, frame_rgb: np.ndarray, prompt: str = DEFAULT_PROMPT) -> SegmentResult:
        torch = self._torch
        h, w = frame_rgb.shape[:2]
        inputs = self.processor(
            text=[prompt], images=[self._to_pil(frame_rgb)], return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits          # (1, 352, 352)
        prob = torch.sigmoid(logits)[0]
        prob = torch.nn.functional.interpolate(
            prob[None, None], size=(h, w), mode="bilinear", align_corners=False
        )[0, 0]
        mask = prob.float().cpu().numpy()
        coverage = float((mask > self.threshold).mean())
        return SegmentResult(mask=mask, prompt=prompt, coverage=coverage)

    def batch(self, frames_rgb: Sequence[np.ndarray],
              prompt: str = DEFAULT_PROMPT) -> list[SegmentResult]:
        """Convenience loop. Production batches on-GPU; this stays simple on purpose."""
        return [self(f, prompt) for f in frames_rgb]
