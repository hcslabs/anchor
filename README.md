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
an HCS-owned **morphology-aware Unitree G1 retargeter**, a KHI reference
exporter, and a hash-bound proof verifier — plus four runnable
[examples](pipeline/examples). See
[`pipeline/README.md`](pipeline/README.md) for the full module map and the
honest open/closed table.

## KHI human-video proof

[`integrations/khi_g1/README.md`](integrations/khi_g1/README.md) contains the
reproducible HCS human-video → G1 → KHI integration. The proof hashes the
original video and every derived artifact without copying the large KHI
handoff. Released GRAIL/GRAB robot trajectories are not used as motion input.
The current receipt is a screening-lineage pass, not a physics or real-robot
success claim.

## Portable embodiment and policy loop

The retargeting optimizer is robot-agnostic. Robot joint names, semantic
marker links and active full-body or upper-body subsets live in validated
embodiment profiles under [`integrations/embodiments`](integrations/embodiments).
The full reproducible flow—including the video-derived phase policy, Isaac Lab
physics receipt and production release gates—is documented in
[`docs/ANCHOR_VIDEO_TO_PRODUCTION.md`](docs/ANCHOR_VIDEO_TO_PRODUCTION.md).

```text
video -> 3-D human motion -> morphology retargeting -> motion-prior policy
      -> physics/task learning -> feedback receipt -> bounded Anchor revision
```

The supplied KHI physics path deliberately remains non-actuating. Its public
collision surrogate preserves the KHI URDF joints, inertials, axes and limits,
but not confidential vendor mesh geometry; it therefore cannot by itself
qualify grasp contacts or a production release.

The 2026-10-14 GPU evidence is checked in under
[`evidence/vast_20261014/isaaclab_policy_physics`](evidence/vast_20261014/isaaclab_policy_physics):
the video-derived policy fit reached 0.00459 rad joint RMSE, and the Isaac Lab
fixed-base upper-body playback reached 0.04880 rad active-joint RMSE with zero
joint-limit violations over 2,120 physics steps. The combined release receipt
still blocks production until exact collision/contact assets and task-level
grasp/hold evidence exist.

## Noise as Signal

New on the page (**§04.5**): our research direction treating the residual
between video-based pose estimation and mocap as a *structured, task-
conditioned noise distribution* — and injecting it during policy training as
a regularizer for sim-to-real transfer. Results are being validated
internally; a public sampler skeleton lives in `anchor.noise`.

## Validation

The production reconstruction design builds on
[VideoMimic](https://arxiv.org/abs/2505.03729) (CoRL 2025). VideoMimic also
publishes a baseline G1 real-to-sim retargeting path; Anchor does not claim
that published work as HCS IP. HCS's contribution in this repository is the
provenance/quality layer, the independently implemented robot-proportioned
DLS retargeter, task-specific KHI profile, export adapter, and proof system.
Historical benchmark numbers on the project page are not part of the new KHI
proof receipt unless their raw run bundle is separately attached and hashed.

## What's deliberately withheld

The learned Trust classifier and advanced contact-/object-aware retargeting
remain closed or incomplete. The dependency-light HCS G1 body retargeter is
working code in `anchor.retarget.hcs_g1`; unrelated advanced method interfaces
remain explicit stubs.

## Contact

Questions about Anchor, research collaborations, licensing, or commercial partnerships are welcome.

**Email:** contact@hcslabs.ai

## Ownership

© HCS Labs — Humanoid Control Systems株式会社. All rights reserved.
