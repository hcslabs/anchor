# Anchor

**Anchor** is HCS Labs' data refinery for humanoid robot learning: a pipeline
that turns ordinary human video into provenance-verified, feasibility-checked,
robot-ready training data.

**Project page:** https://hcslabs.github.io/anchor/

```
Intake → Trust → Gates → Enrich → Retarget → Package → Serve
```

## What's in this repo

```
index.html      the project page (no build step — just open it)
assets/         diagrams, validation imagery, demo video clips
pipeline/       the pipeline SDK — open stages as working code, closed
                stages as interface stubs. Start at pipeline/README.md
third_party/    credits for the published / open-source work the
                reconstruction and intake stages build on
```

## The SDK, in one breath

```bash
cd pipeline && pip install -e ".[ingest]"
anchor probe clip.mp4      # probed + content-addressed episode record
anchor gates clip.mp4      # open quality-gate cascade verdict
anchor stages              # the seven stages, open vs closed
```

Working code in the SDK today: scoped **source connectors** (licensed-API
acquisition, local/partner drops), **ffprobe intake** with content addressing,
the **quality-gate cascade** (open tier), **CLIP-based prompt-driven person
segmentation**, and the **Noise-as-Signal residual sampler** —
plus four runnable [examples](pipeline/examples). See
[`pipeline/README.md`](pipeline/README.md) for the full module map and the
honest open/closed table.

## Noise as Signal

New on the page (**§04.5**): our research direction treating the residual
between video-based pose estimation and mocap as a *structured, task-
conditioned noise distribution* — and injecting it during policy training as
a regularizer for sim-to-real transfer. Results are being validated
internally; a public sampler skeleton lives in `anchor.noise`.

## Validation

The reconstruction stage builds on [VideoMimic](https://arxiv.org/abs/2505.03729)
(CoRL 2025); we re-ran its own evaluation protocol (SLOPER4D filtered subset)
and report our measured numbers on the page — clearly separated from cited
baselines. Everything downstream of reconstruction — acquisition, multi-robot
retargeting, feature extraction — is our own work.

## What's deliberately withheld

The Trust + Gates classification logic and the retargeting correspondence /
shape-fitting method. Diagrams mark these with dashed borders and a lock;
code marks them with interface stubs that raise `NotImplementedError`.

## Ownership

© HCS Labs — Humanoid Control Systems株式会社. All rights reserved.
