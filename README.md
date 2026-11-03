# WorldDynamics

**Reliable Action-Conditioned World Models for Long-Horizon Physical Prediction**

WorldDynamics is an auditable research codebase for predicting physical state and video under interventions. It asks whether a model can distinguish futures caused by different actions from the **same** initial physical state, remain stable over long rollouts, and improve closed-loop control. The historical Python import path is `iwm` (formerly InterventionalWorldModels); its earlier research and negative results remain in this repository.

## Implemented research baseline

- MuJoCo cube scenes with exact state snapshot/restore, seven paired force branches, position, velocity, contact, calibrated RGB camera, artifact hashes, and scene/physics-disjoint evaluation families.
- Trainable deterministic action-blind, action-conditioned, paired-difference and shuffled-pair controls, plus an RSSM with a causal prior/posterior boundary. V1's future-observation posterior leakage was fixed in V1.1 and is covered by regression tests.
- 64-step teacher-forced and free-running state evaluation, same-state counterfactual error, OOD physics/geometry tests, contact proxy, and horizon curves.
- Compact from-scratch latent video **flow-matching** model with reconstruction, teacher-forced prediction, autoregressive generation, action-blind control, checkpointing, PSNR/SSIM, and shared-noise counterfactual evaluation. Its codec is deterministic; this is not a Video VAE or a large Video DiT.
- Lightweight visual world/action heads trained independently and jointly. The inverse-action head never receives the target action. These are exploratory WAM baselines, not a validated WAM contribution.
- Real MuJoCo closed-loop random, physics-based greedy Oracle, learned MPC, and learned CEM pilots. Model planners rank actions from current observation and model predictions only.

## Actual results and limits

The V2 dataset contains **72 scenes, 504 paired branches, 64 steps**. Eight state-model arms were trained for **three seeds each**, 10 epochs and 160 optimizer steps per run. All checkpoint and raw per-scene metrics are retained locally. On the independently sealed eight-scene confirmation family, the predeclared primary improvement statistic (blind minus conditioned final-step counterfactual-delta MSE) was **−5.405 m²**, scene-bootstrap 95% CI **[−6.862, −3.926] m²**. The action-conditioned baseline failed this test. The RSSM had lower long-run position error in this pilot but did not deliver planning gains. Earlier V1.1 action-blind free-running MSE was **0.000199**, versus **0.002660** for action-conditioned and **0.002076** for paired supervision.

On eight new closed-loop scenes, the greedy physics Oracle succeeded in **5/8**, random in **0/8**, and conditioned-model MPC in **1/8, 0/8, 0/8** across seeds. This planning task has unresolved feasibility limits and no learned-control success claim. Single-seed V2 video and joint world/action experiments are also negative or inconclusive; video pixel PSNR alone does not establish action fidelity. See the [baseline and confirmation report](docs/V2_BASELINE_REPORT.md), [long-horizon report](docs/V2_LONG_HORIZON_REPORT.md), [video report](docs/V2_VIDEO_REPORT.md), [WAM report](docs/V2_WAM_REPORT.md), and [planning report](docs/V2_PLANNING_REPORT.md). Historical [V1.1 audit](docs/V11_AUDIT_REPORT.md) and [negative results](docs/V11_BASELINE_RESULTS.md) remain available.

**Acceptance:** L1 research engineering is locally demonstrated for state-model data, training, checkpoint loading, independent evaluation, and prediction. L2 strong research engineering is **partial**: three-seed and OOD state experiments exist, but action causality and planning failed; video/WAM are single-seed. L3 research science is **not passed**. No novel algorithm, SOTA claim, real-robot result, official V-JEPA 2-AC run, or large-GPU scaling result is claimed.

## Install and reproduce

Python 3.11+ is supported; the verified Windows environment uses Python 3.12, CPU PyTorch, and MuJoCo. Create a local virtual environment, then:

```bash
python -m pip install -e ".[test,research,visualization]"
python -m pytest -q
ruff check src/iwm/data.py src/iwm/evaluation.py src/iwm/models.py src/iwm/visual_models.py
```

Use **new output directories** on each run. The full V2 reproduction protocol and frozen split definitions are in [the preregistration](docs/CONFIRMATORY_PROTOCOL.md). Example commands:

```bash
python scripts/generate_v2_data.py --output artifacts/my-v2-data --train-scenes 32 --eval-scenes 8 --horizon 64
python scripts/validate_data.py artifacts/my-v2-data
python scripts/run_v2_baselines.py --dataset artifacts/my-v2-data --output artifacts/my-v2-baselines --seeds 17 29 43 --epochs 10
python scripts/evaluate_v2_matrix.py --dataset artifacts/my-v2-data --runs artifacts/my-v2-baselines --output results/my-v2-exploratory --families iid_test physics_ood geometry_ood
```

Run the `confirmatory` family **once after model and metric choices are frozen**. Video, visual, planning, and plotting commands are documented in [training](docs/TRAINING_GUIDE.md), [evaluation](docs/EVALUATION_GUIDE.md), and the V2 reports. Generated data, checkpoints, full raw results, GIFs, and virtual environments are ignored by Git and must be generated locally. The repository publishes only code, protocols, and small evidence summaries. The original threshold-intervention and paired-replay prototypes remain in history; their toy results are never labeled MuJoCo evidence.
