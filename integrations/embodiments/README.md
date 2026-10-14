# Porting Anchor retargeting to another embodiment

The retargeter is morphology-driven rather than G1-specific.  Its Python solver
contains no G1 joint or link names.  A target robot supplies a versioned
`anchor_embodiment_profile_v1` containing:

1. the exact controller joint order;
2. semantic marker-to-link mappings;
3. parent-before-child skeleton edges;
4. marker weights;
5. named motion profiles such as `full_body` or `upper_body`;
6. a verified URDF containing axes, limits and link geometry.

For a semi-humanoid or fixed upper body, the profile exposes only the torso,
shoulder, elbow and wrist joints.  The same DLS solver runs unchanged.  The
base marker may be pelvis, torso or a fixed mounting link, but the coordinate
contract must remain explicit.

An unvalidated profile is rejected by the loader.  Porting is complete only
after FK/Jacobian tests, joint-limit tests, simulator playback and the target
controller adapter all pass.  The H2 template intentionally remains disabled
until the exact vendor model is installed.

