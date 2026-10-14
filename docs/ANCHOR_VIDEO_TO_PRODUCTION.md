# Anchor: human video to production candidate

## Reproducible recipe

```text
approved human RGB video
  -> fixed-rate 5 Hz decoding
  -> 3-D landmarks in x-forward / y-left / z-up
  -> pelvis- or base-centred motion
  -> robot-proportioned semantic marker targets
  -> URDF FK + weighted damped least-squares IK
  -> joint/velocity/continuity quality gates
  -> phase-conditioned motion-prior policy
  -> physics playback and task RL with domain randomization
  -> simulation receipt and failure classification
  -> dataset/reward/profile revision by the bounded Anchor agent
  -> supervised hardware candidate only after every release gate passes
```

The solver is portable; the embodiment profile is not.  Porting to another
humanoid requires that robot's URDF, semantic link map, joint order, active
joint subset, joint/velocity limits and output adapter.  A semi-humanoid uses
an upper-body profile and a fixed base.  A mobile full humanoid additionally
needs its own balance/locomotion controller and whole-body validation.

## Modules and contracts

| Stage | Implementation | Reusable part | Robot-specific part |
|---|---|---|---|
| 5 Hz video pose | `anchor.enrich.human_pose` | decoding, landmark confidence, coordinate convention | camera calibration when available |
| Morphology retargeting | `anchor.retarget.embodiment` | robot-proportioned landmarks and weighted DLS IK | embodiment JSON + URDF |
| Motion-prior policy | `anchor.policy.phase_policy` | phase features, imitation loss, TorchScript export | active joint list |
| Physics screening | `integrations/isaaclab/g1_reference_playback.py` | policy playback and receipt schema | asset, gains, stance/base controller |
| Closed-loop gate | `anchor.agent.production_loop` | evidence checks and next-action classification | task thresholds |

## The retargeting contribution

For each human bone, Anchor preserves its observed direction but replaces its
length with the neutral marker-to-marker distance measured from the target
robot URDF.  The target points are therefore human-directed and robot-sized.
Weighted damped least-squares IK then solves the target robot joint angles.
Wrist and ankle markers can be weighted more heavily, while posture and
continuity terms suppress singular or discontinuous solutions.  Every frame
is initialized from the preceding solution and clamped to URDF limits.

This is why the implementation can target a full humanoid, a different
humanoid, or a fixed semi-humanoid upper body without changing the solver.
Only the validated embodiment profile and output adapter change.

## Policy learning

The retargeted trajectory first trains a compact phase-conditioned imitation
policy.  This provides a video-derived motion prior and a reproducible initial
policy.  It is not the final grasp controller.  Contact, balance and robustness
are learned in physics as a residual policy with randomized mass, friction,
object pose, table height, actuator response, sensor delay and observation
noise.  PPO is one supported optimizer; its reward contract should reward
containment, lift and stable hold, and penalize lateral error, excessive speed,
over-lift and abrupt commands.  Table collisions and joint limits remain hard
termination constraints rather than tunable reward shortcuts.

The included trainer is the **imitation warm-start**, not a claim that PPO
contact training has already completed.  It turns the motion seen in the
human video into a compact policy prior.  The next physics stage should add a
task observation (robot state, object pose and contact state) and train a
residual action on top of that prior.  This preserves human motion intent
while allowing physics feedback to correct balance, contact and timing.

## Validated Vast.ai run — 2026-10-14

The selected GRAB clip was decoded into 53 frames at 5 Hz and retargeted to
the KHI 29-DoF G1 URDF.  The results below come from the checked-in JSON
receipts, not from presentation estimates.

| Gate | Measured result | Verdict |
|---|---:|---|
| Human pose coverage | 53 / 53 frames | pass |
| Retargeting mean marker RMSE | 9.49 mm | pass |
| Policy fit joint RMSE | 0.00459 rad | pass |
| Isaac active-joint tracking RMSE | 0.04880 rad | pass (`<= 0.12`) |
| Active maximum tracking error | 0.28374 rad | diagnostic |
| Joint-limit violations | 0 over 2,120 steps | pass |
| Simulated duration | 10.6 s at 0.005 s physics step | complete |
| Exact production collisions/contact task | unavailable | **blocked** |

The physics run used Isaac Lab on an NVIDIA Quadro RTX 6000.  It used the
exact KHI joint names, axes, limits and inertials, with primitive collision
surrogates in place of non-redistributable vendor meshes.  The fixed-base
upper-body policy controls waist and right arm; a separate stance controller
holds inactive leg joints.  Active-joint tracking passed.  The all-joint RMSE
was 0.23552 rad because this screening setup does not include the production
whole-body balance controller, so it must not be presented as a mobile
full-body validation.

## Closed feedback

Anchor reads machine-verifiable receipts instead of editing code freely.  A
failed run is classified into lineage, pose coverage, retarget fit, actuator
tracking, balance, collision, grasp or hold failure.  The agent may select a
bounded recipe, request a different human-motion prior, adjust an allowed
reward profile, or request more domain randomization.  It may not authorize a
physical robot run.

## Release boundary

The included Isaac Lab playback uses the exact KHI URDF joints, inertials,
axes and limits, but primitive collision geometry because the vendor meshes
are not redistributable here.  It is valid for joint-dynamics plumbing and
policy tracking only.  Production contact qualification requires the vendor
collision meshes, calibrated actuator model, randomized tabletop task,
multi-seed success statistics, and supervised hardware validation.

The resulting production-gate receipt therefore reports lineage, policy fit,
physics tracking, uprightness and joint limits as passing, while production
asset mapping, grasp success and five-second hold remain failing/unmeasured.
Anchor's next action is to install the exact collision/actuator asset before
contact policy training; it never authorizes physical execution itself.

