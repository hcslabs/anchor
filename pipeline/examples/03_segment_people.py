"""Prompt-driven person segmentation on one frame of a clip.

Requires: pip install "anchor-pipeline[segment]"
"""
import sys

import cv2  # comes with the segment extra

from anchor.enrich.segment import PersonSegmenter

path = sys.argv[1]
cap = cv2.VideoCapture(path)
ok, bgr = cap.read()
assert ok, f"could not read a frame from {path}"
frame_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

seg = PersonSegmenter()
res = seg(frame_rgb, prompt="a person")
print(f"prompt={res.prompt!r}  coverage={res.coverage:.3f}  mask={res.mask.shape}")

overlay = frame_rgb.copy()
overlay[res.mask > seg.threshold] = (overlay[res.mask > seg.threshold] * 0.4
                                     + (255 * 0.6, 0, 0))
cv2.imwrite("segmented.png", cv2.cvtColor(overlay.astype("uint8"), cv2.COLOR_RGB2BGR))
print("wrote segmented.png")
