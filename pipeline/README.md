# Anchor pipeline SDK

The pipeline behind [the Anchor project page](../index.html): video in,
provenance-verified, feasibility-checked, robot-ready training data out.

```
Intake → Trust → Gates → Enrich → Retarget → Package → Serve
  open   closed   open     open     closed     open     open
```

This SDK ships the **open stages as working code** and the closed stages as
interface stubs. That split is deliberate: what's open is everything a team
needs to *plug into* Anchor — connectors, schemas, gates, segmentation, the
Noise-as-Signal sampler. What's closed is the part that makes
Anchor's output trustworthy and deployable — the Trust scoring method and the
retargeting correspondence method.

## Install

```bash
cd pipeline
pip install -e .                       # core: numpy only
pip install -e ".[ingest]"             # + source connectors (requests)
pip install -e ".[segment]"            # + CLIP-based segmentation (torch, transformers)
pip install -e ".[noise]"              # + Noise-as-Signal extras (scipy)
```

## 60-second tour

```python
from anchor.ingest import ingest, probe
from anchor.quality import GateCascade, default_open_gates

ep = ingest("clip.mp4")                          # probed + content-addressed Episode
ok = GateCascade(default_open_gates()).run(probe("clip.mp4"), ep)
print(ep.to_json())                              # full audit trail, gates included
```

```bash
anchor probe clip.mp4        # same thing from the shell
anchor gates clip.mp4
anchor stages                # the seven stages and their open/closed status
```

Runnable end-to-end scripts live in [`examples/`](examples):

| example | shows |
|---|---|
| `01_scout_licensed_sources.py` | scoped acquisition from a licensed API (Pexels) |
| `02_probe_and_gate.py` | folder → probe → gate cascade → verdicts |
| `03_segment_people.py` | prompt-driven person segmentation (CLIPSeg) |
| `04_noise_as_signal.py` | fit D̂ from paired data, sweep λ — the §04.5 research direction |

## Module map

```
src/anchor/
  ids.py          content addressing (anc1-sha256) — dedup + idempotent re-ingestion   [open]
  schema/         Episode record, trust-tier ladder, task taxonomy                     [open]
  ingest/         intake (ffprobe) + scout connectors (θ_scope lives here)             [open]
                  crawler.py — allowlist, robots-respecting crawl skeleton              [open, placeholder]
  quality/        gate cascade — open tier real, learned tier closed (θ_proc)          [open/closed]
  enrich/         segmentation open (CLIPSeg); pose/depth/hands/contact stubs          [open/stubs]
  noise/          Noise-as-Signal empirical residual sampler                           [open, in validation]
  trust/          provenance scoring, synthetic screening, dedup                       [CLOSED]
  retarget/       reconstruction → robot joint mapping + feasibility                   [CLOSED]
  latents/        cached encoder feature store                                         [stub]
  package/        dataset shards + auto-datasheets (LeRobot-v3 export planned)         [stub]
  pipeline/       stage registry + runner                                              [open/stub]
  serve/          catalog + query API                                                  [stub]
  cli.py          `anchor` command                                                     [open]
```

## Design notes worth stealing

- **Content addressing first.** Every asset's identity is its bytes
  (`anc1-<sha256>`), so re-ingestion, dedup, and provenance are idempotent by
  construction. Filenames and URLs are metadata, never identity.
- **Append-only episodes.** Stages append fields; nothing overwrites. Any
  training sample can be walked back to its source bytes and every gate it
  passed. Rejections are kept for audit, never silently dropped.
- **Cheap gates first.** A 0.4-second clip should never cost a GPU minute to
  reject. The cascade is ordered by cost, and every threshold in it is a
  θ_proc parameter the closed-loop optimizer tunes between rounds.
- **License at acquisition, not after.** A `Candidate` is born with its
  license recorded. Anchor doesn't bulk-crawl platforms whose terms prohibit
  it — a clip whose license you can't state is one the Trust stage rejects
  anyway, so we don't spend the bandwidth.
- **Prompt-driven segmentation.** The segmentation target is *text*, so the
  same code isolates "a person carrying a box" today and "a person on a
  ladder" tomorrow, zero retraining.

## Noise as Signal

The residual between video-based pose estimation and mocap, measured on paired
recordings, is a *structured* noise distribution D̂ — per-joint scale,
temporally correlated. `anchor.noise` fits D̂ and samples it by block
bootstrap, so a training loop can perturb clean trajectories with
**realistic** estimator noise:

```python
from anchor.noise import fit, perturb
bank = fit(pairs)                        # pairs: [(q_mocap, q_video), ...]
q_tilde = perturb(q_mocap, bank, scale=0.5)   # q̃ = q_m + λ·ε′,  ε′ ~ D̂
```

The approach, status, and the figure explaining why this beats Gaussian
augmentation: project page, **§04.5 — Noise as signal**. Results are being
validated internally.

## What's closed, exactly

| stage | public | closed |
|---|---|---|
| Trust | the tier ladder (`TrustTier`), the interfaces | provenance classifier, synthetic-content screens, near-dup index |
| Retarget | stage shape (reconstruction in → joints + feasibility out), validation results | correspondence + shape-fitting method |

Closed modules raise `NotImplementedError` with a pointer here. As stages
mature past internal validation we expect to open more — starting with the
stages already marked "validated" on the project page. Partner access:
contact HCS Labs.

© HCS Labs — Humanoid Control Systems株式会社. All rights reserved.
