# Anchor → KHI Unitree G1 integration

For the profile-driven, multi-embodiment implementation and the bounded
feedback agent, see [`CLOSED_LOOP.md`](CLOSED_LOOP.md).  The G1 joint/link map
is data in `unitree_g1_29dof.embodiment.json`; it is not embedded in the generic
solver.  `integrations/embodiments/unitree_h2.template.json` documents the H2
adapter contract but intentionally refuses execution until an exact H2 URDF
and mapping are validated.

This integration proves a narrow but important claim:

> A KHI-compatible Unitree G1 joint reference was computed from a selected
> human RGB video by HCS code. It was not copied from a released robot-motion
> dataset.

The proof is intentionally small. Large source videos and the 6+ GB KHI
handoff stay where they are; the capsule stores paths, sizes, and SHA-256
hashes rather than duplicating them.

## Data flow

```text
approved human video
  → content hash + quality record
  → 3-D human landmarks (screening extractor or production reconstruction)
  → body-centred coordinates (x forward, y left, z up)
  → HCS robot-proportioned landmark targets
  → damped least-squares IK against the KHI G1 URDF
  → joint-limit and temporal checks
  → KHI joblib reference (N × 29 DoF)
  → key-matched free-cube initialization companion
  → existing KHI Isaac Lab loader / PPO environment
```

The local screening extractor uses MediaPipe to prove byte-level lineage from
video. Production training data must replace that screening pose with a
metric VideoMimic/SMPL-X reconstruction or an equivalent validated 4-D human
reconstruction. The HCS retargeting and KHI export stages do not change.

## What is HCS-owned here

- the body-frame normalization;
- robot-proportioned target construction from human bone directions;
- the weighted, joint-limited, temporally regularized DLS solver;
- the KHI G1 profile (stationary lower body, waist + right-arm motion);
- the no-copy lineage manifest and verifier;
- the KHI reference exporter and round-trip validator.

GMR and GRAIL are formulation and interchange references. Their released
robot trajectories are **not** inputs to this proof. VideoMimic is the planned
production reconstruction stage, not the retargeting implementation shipped
here.

## Solver

For marker residual `e(q)` and weighted point Jacobian `J`, each iteration is
the damped and regularized least-squares update

```text
Δq = (Jᵀ W J + λ²I + αI + βI)⁻¹
     (Jᵀ W e − αq + β(q_previous − q))
```

The update is clipped per iteration and the resulting joints are clipped to
the limits read from the KHI URDF. `α` keeps unused freedom near the neutral
pose; `β` discourages frame-to-frame jitter. Human segment lengths are not
copied. Human bone *directions* are preserved while target segment lengths are
reconstructed from G1 geometry.

## Exact local proof commands

PowerShell:

```powershell
cd C:\Users\salio\anchor_page
$env:PYTHONPATH = (Resolve-Path pipeline\src)
$PY = 'C:\Users\salio\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$KHI = 'C:\Users\salio\Downloads\G1_COMPLETE_HANDOFF_20260915_01\source\g1-aprilcube-autonomy'
$VIDEO = 'C:\Users\salio\Downloads\KHI_HumanMotion_Video_Sources\KIT_CrackerBox_FullBody_Direct_Grasp.mp4'
$MODEL = 'C:\Users\salio\.cache\anchor-models\pose_landmarker_lite.task'

& $PY -m anchor.cli extract-human-pose `
  --video $VIDEO --model $MODEL `
  --output evidence\khi_g1\kit_crackerbox_human_motion_v1.json `
  --sample-fps 5

& $PY -m anchor.cli retarget-g1 `
  --human-motion evidence\khi_g1\kit_crackerbox_human_motion_v1.json `
  --urdf "$KHI\g1_29dof_pc2.urdf" `
  --profile khi_tabletop_right `
  --output evidence\khi_g1\kit_crackerbox_hcs_g1_retarget_v1.json

& $PY -m anchor.cli export-khi-g1 `
  --trajectory evidence\khi_g1\kit_crackerbox_hcs_g1_retarget_v1.json `
  --output evidence\khi_g1\kit_crackerbox_khi_g1_reference.pkl

& $PY -m anchor.cli validate-khi-g1 `
  evidence\khi_g1\kit_crackerbox_khi_g1_reference.pkl

& $PY -m anchor.cli export-khi-object `
  --robot-reference evidence\khi_g1\kit_crackerbox_khi_g1_reference.pkl `
  --output evidence\khi_g1\kit_crackerbox_khi_object_reference.pkl `
  --relative-position-m 0.487 -0.180 -0.089

& $PY -m anchor.cli validate-khi-pair `
  --robot-reference evidence\khi_g1\kit_crackerbox_khi_g1_reference.pkl `
  --object-reference evidence\khi_g1\kit_crackerbox_khi_object_reference.pkl

& $PY -m anchor.cli proof-khi `
  --khi-root $KHI --human-video $VIDEO `
  --human-motion evidence\khi_g1\kit_crackerbox_human_motion_v1.json `
  --g1-trajectory evidence\khi_g1\kit_crackerbox_hcs_g1_retarget_v1.json `
  --khi-reference evidence\khi_g1\kit_crackerbox_khi_g1_reference.pkl `
  --khi-object-reference evidence\khi_g1\kit_crackerbox_khi_object_reference.pkl `
  --output evidence\khi_g1\anchor_khi_proof_manifest_v1.json

& $PY -m anchor.cli proof-verify `
  evidence\khi_g1\anchor_khi_proof_manifest_v1.json `
  --receipt evidence\khi_g1\anchor_khi_proof_receipt_v1.json

& $PY -m unittest discover -s pipeline\tests -v
```

## Current verified result

- Six unit/integration tests pass, including a regression test that prevents
  missed detections from compressing the motion clock.
- The actual video hash is bound through every derived artifact.
- All three generated `N × 29` references round-trip through `joblib.load`.
- Every referenced KHI and Anchor source file matches its manifest hash.
- The strongest screening case is the lateral GRAB clip: 53/53 detected
  frames, 8.7 mm mean marker residual, 22.8 mm maximum residual, and all URDF
  joint/velocity limits satisfied.
- The lateral KIT clip emits all 40 source-time samples (27 detected plus 13
  internally interpolated). Preserving its clock reduced the prior false
  velocity spike from 6.86 to 2.69 rad/s.
- All generated evidence for three cases totals about 0.34 MB. The source
  videos and 6+ GB KHI repository are not copied.

The current receipt is `TECHNICAL_LINEAGE_PASS_SCREENING_ONLY`. It is not a
claim of Isaac physics success, robot success, production metric-pose quality,
or dataset-rights clearance.

## Gates before training or a customer-facing claim

1. Replace the screening pose artifact with a VideoMimic/SMPL-X metric output
   generated from an HCS-owned or contractually approved video.
2. Add object pose/contact timing (the current proof covers body motion).
3. Run the existing KHI free-object replay in Isaac Lab; require no attachment
   or object-pose writes and archive `result.json` + `state_log.csv`.
4. Run the KHI feasibility/clearance checks before any physical execution.
5. Sign the source-rights attestation and rebuild the proof manifest.

## Disposable-GPU physics check (next gate)

Do not clone the simulator or weights onto this nearly full Windows disk. On
the existing disposable Isaac Lab machine, upload only the ~1.5 KB generated
reference and use it as `--robot-reference` with the KHI replay tool. The
object reference is used only to initialize the free cube; the KHI replay does
not write object pose after reset.

```bash
export HCS_PROJECT=/workspace/g1-aprilcube-autonomy
export UNITREE_SIM_ROOT="$HCS_PROJECT/simulation/unitree_sim_isaaclab"
export ISAACLAB_ROOT=/workspace/IsaacLab
export ANCHOR_ROBOT=/workspace/anchor_khi/kit_crackerbox_khi_g1_reference.pkl
export ANCHOR_OBJECT=/workspace/anchor_khi/kit_crackerbox_khi_object_reference.pkl
export ANCHOR_RESULTS=/workspace/results/anchor_khi_screening

cd "$UNITREE_SIM_ROOT"
"$ISAACLAB_ROOT/isaaclab.sh" -p tools/aprilcube_inspire_grail_replay.py \
  --robot-reference "$ANCHOR_ROBOT" \
  --object-reference "$ANCHOR_OBJECT" \
  --output-dir "$ANCHOR_RESULTS" \
  --cube-size 0.040 --cube-mass 0.060 --headless
```

This run is expected to test physical compatibility, not necessarily to lift:
the lightweight pose proof does not yet reconstruct object contact timing or
generate hand closure. A valid result still requires a free cube, zero
attachments, zero post-reset object-pose writes, and archived output hashes.

## Research references

- [VideoMimic](https://github.com/hongsukchoi/VideoMimic): production human + scene reconstruction reference.
- [GMR](https://github.com/YanjieZe/GMR): published general motion-retargeting reference.
- [GRAIL](https://github.com/NVlabs/GRAIL): 4-D HOI and G1 interchange reference. Its license and released data are not inherited by this HCS implementation.

