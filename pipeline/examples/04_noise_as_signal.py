"""Noise-as-Signal in 20 lines: fit D̂ from paired data, perturb a clean trajectory.

This is the research direction from the project page (§04.5).
Here the "paired data" is synthetic so the example runs anywhere; in practice
the pairs are real (mocap, video-pipeline) recordings of the same motions.
"""
import numpy as np

from anchor.noise import fit, perturb

rng = np.random.default_rng(7)
T, J, D = 600, 23, 3

# stand-in for paired recordings of the same motion
q_mocap = rng.standard_normal((T, J, D)).cumsum(0) * 0.01
drift = 0.015 * np.sin(np.linspace(0, 8 * np.pi, T))[:, None, None]   # structured, not white
q_video = q_mocap + drift + 0.01 * rng.standard_normal((T, J, D))

bank = fit([(q_mocap, q_video)], block_len=32)
print("per-joint residual RMS (first 5):", np.round(bank.joint_scale[:5], 4))

for lam in (0.0, 0.5, 1.0, 2.0):
    q_tilde = perturb(q_mocap, bank, scale=lam, rng=rng)
    rms = float(np.sqrt(((q_tilde - q_mocap) ** 2).mean()))
    print(f"λ={lam:3.1f}  injected-noise RMS={rms:.4f}")
# sweep λ against your policy's held-out sim-to-real transfer to find λ*.
