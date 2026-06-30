# Pipeline source — closed for now

This folder mirrors the structure of Anchor's pipeline implementation. The
files here are placeholders: each one exists to show the shape of the system,
not the system itself. Every file currently contains a single `# coming soon`
line.

## What's public today

- The [project page](../index.html) — architecture, validated results, and an
  honest account of what's built vs. in development.
- [`third_party/`](../third_party) — credit and citations for the published
  and open-source work this pipeline's reconstruction and intake stages build
  on or reference.
- This folder's structure — the module layout, not the implementation.

## Layout

```
src/anchor/
  schema/       episode + taxonomy data model
  ingest/       intake: probing, content-addressing
  trust/        provenance, synthetic-content screening, dedup
  quality/      the quality-gate cascade
  enrich/       depth, hands, body, segmentation, tracks, contact, captions
  retarget/     human reconstruction → robot-specific motion mapping
  latents/      cached encoder feature store
  package/      dataset packaging + datasheets
  pipeline/     stage registry and orchestration
  serve/        catalog + API surface
```

## What's closed today

- The Trust + Gates classification logic (provenance scoring, quality
  cascade, what gets rejected and why).
- The retargeting correspondence and shape-fitting method (how a human
  reconstruction maps onto a specific robot's joints and dynamics).
- All production infrastructure, configs, and orchestration code.

This isn't a permanent state. As stages mature past internal validation we
expect to open more of this up — starting with the stages already marked
"validated" on the project page.

**Status:** coming soon.
