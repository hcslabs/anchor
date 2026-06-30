# Anchor

**Anchor** is HCS Labs' data refinery for humanoid robot learning: a pipeline
that turns ordinary human video into provenance-verified, feasibility-checked,
robot-ready training data.

This repository hosts the public project page and credits for the published
research Anchor's reconstruction stage builds on. The pipeline implementation
itself is closed for now — see [`pipeline/`](pipeline) for what that means and
what's planned.

## What's here

```
index.html      the project page (no build step — just open it)
assets/         diagrams, validation imagery, and demo video clips referenced
                by the page
third_party/    credits for the published / open-source work this pipeline
                builds on or references (VideoMimic real2sim, Video-LLaMA)
pipeline/       placeholder for the pipeline source — closed for now
```

## Viewing the page locally

```bash
open index.html       # macOS
xdg-open index.html   # Linux
```

Or serve it if your browser is picky about `file://` and relative paths:

```bash
python3 -m http.server 8000
# then visit http://localhost:8000
```

## Status

Internal validation in progress. The page covers:

- The acquisition, reconstruction, and retargeting pipeline overview
- Validation against the published VideoMimic benchmark (SLOPER4D subset),
  re-run by us — see the page for what's our measurement vs. cited
- How this pipeline's intake also feeds Aegis, HCS Labs' predictive safety
  system
- Anchor's broader trust-tier design (Intake → Trust → Gates → Enrich →
  Retarget → Package → Serve), with an honest accounting of what's validated,
  partially validated, and still in development

What's deliberately closed as of now from any diagram on this page: the Trust + Gates
classification logic and the retargeting correspondence/shape-fitting method (coming soon).
See `assets/anchor_acquisition_figure.png` for how that's represented visually
(dashed borders, lock indicator).

## Ownership

© HCS Labs — Humanoid Control Systems株式会社.
