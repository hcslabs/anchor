# VideoMimic (real2sim)

This project's scene-and-human reconstruction stage builds on the **real2sim**
pipeline from VideoMimic — the vision pipeline that reconstructs 3D
environments and human motion from single-camera video, then retargets that
motion to humanoid robots.

We use this pipeline for reconstruction only, and have customized it for
Anchor's specific requirements. We did not develop the underlying
reconstruction algorithm. Anchor's own contributions sit around it: the data
acquisition pipeline that gets arbitrary video into the system reliably, the
multi-robot retargeting layer downstream of reconstruction, the
feature-extraction stage that produces structured, queryable output, and the
production validation that makes the whole thing deployable.

We also independently re-ran VideoMimic's own published evaluation protocol — a
filtered six-sequence subset of the SLOPER4D dataset, comparing reconstructed
human trajectories and scene geometry against motion-capture-grade ground truth
— to validate our customized pipeline against the same benchmark the original
paper reports against. Those re-run results are reported on the project page,
clearly marked as our own measurements, not the paper's published figures.

**Source**

> [github.com/hongsukchoi/VideoMimic/tree/main/real2sim](https://github.com/hongsukchoi/VideoMimic/tree/main/real2sim)

**Citation**

> Allshire, A., Choi, H., Zhang, J., McAllister, D., Zhang, A., Kim, C. M.,
> Darrell, T., Abbeel, P., Malik, J., Kanazawa, A.
> *Visual Imitation Enables Contextual Humanoid Control.* CoRL 2025
> (Best Student Paper Award). arXiv:[2505.03729](https://arxiv.org/abs/2505.03729)

No VideoMimic source code is redistributed in this repository. This notice
exists to credit the method this pipeline's reconstruction stage is built on.
To clone the upstream repo locally for reference:

```bash
git clone https://github.com/hongsukchoi/VideoMimic.git
# the relevant subfolder is VideoMimic/real2sim
```
