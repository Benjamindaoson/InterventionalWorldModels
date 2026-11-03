# Model implementations

- `RSSM`: observation encoder, GRU state, Normal prior/posterior, action transition, state decoder, KL and reconstruction loss, free rollout. Reward and termination heads exist but have no current labels.
- `ActionPredictor`: deterministic residual state transition, teacher-forced and free modes.
- `VisualPredictor`: scratch visual/action encoders, GRU fusion, future representation and reconstruction losses. **Not V-JEPA 2-AC.**
- `VideoFlow`: 32×32 latent encoder/decoder, causal temporal transformer with position/time embeddings, flow-matching objective, Euler conditional generation. Prototype only.
- `JointWorldAction`: shared visual tokens and GRU, world and inverse-action heads with weighted joint loss. Independent and ordering comparisons remain pending.
- `geometry`: SE(3), inverse pose, projection/unprojection, reprojection and temporal consistency losses. Geometry-constrained training remains pending.
