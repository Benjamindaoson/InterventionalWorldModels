# Architecture

`iwm.paired_replay` preserves exact restore contracts. `iwm.mujoco_paired_backend` records measured trajectories. `iwm.data` writes JSON provenance and compressed RGB, checks hashes and frozen splits. `iwm.models` supplies state dynamics; `iwm.visual_models` supplies compact RGB baselines. `iwm.training` saves resumable state-model checkpoints and `iwm.evaluation` measures frozen rollouts independently. `iwm.geometry` contains differentiable camera operators; `iwm.rollout` exposes re-anchoring.

State: `[batch,time+1,3]` meters. Action: `[batch,time,3]` newtons. RGB: `[batch,time+1,3,32,32]`. Current scene is a freely moving cube on a plane, not robot manipulation.
