# WorldDynamics — Reliable Action-Conditioned World Models for Long-Horizon Physical Prediction

世界模型：面向长时序物理预测的可靠动作条件世界模型。

WorldDynamics retains the historical `iwm` package, threshold experiments and paired-replay research. Earlier simplified RGB threshold experiments did **not** establish an advantage for paired-branch supervision. No novel algorithm or published benchmark claim is made.

## Implemented CPU pilot

- MuJoCo same-state action branches with per-step state, velocity, contact and 32×32 RGB, hash receipts and frozen scene splits.
- Trainable RSSM and deterministic action-conditioned state predictor with saved checkpoints and independent rollout evaluation.
- From-scratch visual representation, latent video flow matching and joint world/action baselines; one-epoch smoke runs only.
- Differentiable geometry and a state re-anchoring interface.

Use Python 3.12. Install `numpy mujoco torch pyyaml ruff` and `pip install -e ".[test]"`. Then run:

```bash
python scripts/generate_data.py --output artifacts/mujoco-rgb-v2 --scenes 12 --horizon 16
python scripts/validate_data.py artifacts/mujoco-rgb-v2
python scripts/train.py --config configs/models/rssm.yaml --dataset artifacts/mujoco-rgb-v2 --run-dir artifacts/rssm-v2
python scripts/train.py --config configs/models/action_predictor.yaml --dataset artifacts/mujoco-rgb-v2 --run-dir artifacts/action-v2
python scripts/evaluate.py --checkpoint artifacts/rssm-v2/last.pt --dataset artifacts/mujoco-rgb-v2 --output results/rssm-v2-test.json
python scripts/train.py --config configs/models/video_flow.yaml --dataset artifacts/mujoco-rgb-v2 --run-dir artifacts/video-v2
python scripts/generate_video.py --checkpoint artifacts/video-v2/last.pt --dataset artifacts/mujoco-rgb-v2 --output artifacts/video-v2-sample.npz
python -m pytest -q
```

Outputs must use new paths. The local pilot used 12 scenes (8 train, 2 validation, 2 test), 36 action branches and 16 steps. On six test branches, RSSM one-step/free-running MSE was 0.009637/0.001731; deterministic predictor MSE was 0.002513/0.169207. The video sample had 17 frames and 14.33 dB PSNR. These are single-seed pilot measurements, not benchmark conclusions. `pytest`: 43 passed. See `docs/IMPLEMENTATION_STATUS.md` for limits.

## Historical project statement

Research code for **action-conditioned visual world models from sparse physical interventions**.

**Status:** Early-stage research. Existing simplified RGB threshold experiments did **not** establish that paired-branch supervision beats independent supervision. No claim of CVPR-level novelty, real physics validation, or published-model reproduction is made.

## Research question

Can a model infer the *differences between futures produced by alternative actions from the same physical state*, especially near contact or mode transitions, with fewer interventions and reliable uncertainty?

## Initial contribution to this repository

- A mathematically auditable threshold-intervention baseline (not a novel algorithm).
- Explicit abstention on actions for which observations do not determine the outcome.
- Tests including randomized finite-case verification and an indistinguishable-world counterexample.
- A research audit outlining how to compare against **Video Prediction Policy** (ICML 2025) and **OpenWAM** (October 2026) without misrepresenting their methods.

## Local setup

```bash
python -m pip install -e ".[test]"
pytest -q
python -m iwm.thresholds
```

## Research gates

1. Verify same-state physical replay and event labels using MuJoCo, then independently PhysX.
2. Validate the exact OpenWAM/FDM evaluation protocol and checkpoints before comparing scores.
3. Train equal-data, equal-capacity, equal-compute independent and paired conditional predictors.
4. Report shuffled-pair, target-only, auxiliary event loss, and capacity-matched ablations.
5. Freeze final held-out scenes, use independent seeds, confidence intervals, calibration and closed-loop outcomes.

## Prior research

- VPP (ICML 2025): https://proceedings.mlr.press/v267/hu25g.html
- OpenWAM: https://github.com/OpenWAM/OpenWAM

All research claims should identify the experiment, dataset, checkpoint, seed and statistical uncertainty that support them.
