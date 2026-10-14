# Anchor policy and Isaac physics evidence — 2026-10-14

This directory is the reproducible, non-actuating evidence bundle for the
selected KHI G1 motion candidate.

| Artifact | Purpose |
|---|---|
| `policy/anchor_motion_prior_policy_receipt_v1.json` | video/trajectory/profile lineage and policy-fit metrics |
| `policy/anchor_phase_policy.pt` | TorchScript imitation warm-start policy |
| `g1_29dof_collision_surrogate.receipt.json` | source/surrogate URDF hashes and allowed-use boundary |
| `g1_29dof_collision_surrogate.urdf` | exact joint tree/inertials with primitive collision geometry |
| `anchor_isaaclab_physics_playback_receipt_v1.json` | 10.6 s, 2,120-step Isaac Lab tracking result |
| `anchor_production_gate_receipt_v1.json` | combined pass/fail release decision |

Measured active-joint physics tracking RMSE is **0.04880 rad** with **zero
joint-limit violations**.  The receipt says `physics_pass: true` for the
fixed-base joint-dynamics scope, but `production_release_ready: false` because
exact production collision geometry, contact-task success and hold evidence
are not present.  No physical robot commands were sent.
