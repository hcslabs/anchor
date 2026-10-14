# Anchor scalable retargeting and KHI closed loop

## What is proven

The pipeline decodes approved human RGB videos on a fixed 5 Hz clock, derives
pelvis-centred 3-D landmarks, reconstructs target bones from the target robot's
own URDF link lengths, and solves a joint-limited damped least-squares problem.
It does not consume a pre-retargeted G1 motion.

Robot-specific information is stored in a versioned embodiment profile.  The
solver itself does not contain G1 joint or link names.  Porting to another
humanoid therefore means supplying and validating:

1. the exact controller URDF;
2. controller joint order;
3. semantic human-marker to robot-link mappings;
4. active joint and marker subsets;
5. simulator and hardware output adapters.

The H2 file in `integrations/embodiments` is deliberately an unvalidated
template.  The loader refuses to execute it until an exact H2 URDF and verified
mapping are installed.  This prevents a portability claim from being confused
with H2 validation.

## Closed-loop agent boundary

`anchor closed-loop-khi` is a bounded orchestration agent.  It:

1. extracts each candidate human motion once;
2. evaluates named IK recipes;
3. rejects candidates that fail visibility, timing, residual, joint or velocity gates;
4. selects the lowest-scoring passing candidate;
5. emits KHI-compatible robot/object references only after those gates pass;
6. reads optional KHI physics feedback and recommends the next dataset/reward action;
7. writes an auditable receipt with every decision and hash.

It cannot send physical robot commands.  With no physics feedback, its receipt
must remain `physics_pass=false`.  Table collisions request a lateral-clearance
human-motion candidate.  Low grasp/hold performance requests a reward-priority
revision around containment, lift and stable hold, followed by resimulation.

## Vast.ai command

```bash
cd /workspace/anchor-khi-closed-loop-20261014
bash integrations/khi_g1/run_anchor_closed_loop.sh
```

Optional KHI feedback:

```bash
ANCHOR_PHYSICS_FEEDBACK=/path/to/anchor_khi_physics_feedback_v1.json \
  bash integrations/khi_g1/run_anchor_closed_loop.sh
```
