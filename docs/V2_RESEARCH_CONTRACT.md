# V2 research contract: Action-conditioned visuo-tactile world models

Status: **DESIGN ONLY**. No v2 neural training, tactile model or generative results are asserted.

## Target
One research flagship in the existing repository, with scientific question:
> Under contact-rich interaction and restricted data, which representations and action-conditioned temporal mechanisms improve out-of-distribution physical-event and future-state prediction, and when can tactile observations reduce visual ambiguity?

The main target is shared world-model algorithm competence for representation/dynamics/generalization roles and visuo-tactile/video world-model roles. Matching a job description is not a scientific result or a guarantee of job eligibility.

## Core hypotheses and falsification
H1 Representation: JEPA-style training with appropriate anti-collapse regularization yields nondegenerate embeddings containing probe-accessible position, pose, motion and contact information, compared with capacity/data/computation-controlled alternatives. Check variance/effective rank, probe error and future prediction jointly. A regularizer increasing embedding spread without improving prediction is not evidence of better dynamics.

H2 Actions: Correct action-conditioned predictors outperform no-action, zero-action and shuffled-action controls on genuinely held-out same-start branches. Each model must have parameter/compute matched baselines where feasible; report latency and train budget. Failure to beat these controls invalidates the action-use claim.

H3 Physical events: Event-aware latent prediction improves contact transition timing and calibrated near-boundary prediction beyond standard latent prediction at fixed data and compute. Evaluate separately by contact event, contact-rich vs free-motion, and scene.

H4 Tactile: Synchronized tactile observation improves contact-event/physical-state prediction specifically under visual ambiguity; compare tactile-only, visual-only, fusion with and without action, matched splits and missing-modality stress tests. Simulated tactile is not real-sensor evidence.

H5 Generative: A **trained** conditional generative decoder/video model produces action-consistent future frames; decoding observed features or passing images through a deterministic head does not qualify as trained video generation.

H6 Planning: A frozen trained world model can score candidate actions without querying environment futures at decision time; compare against simple planning, random and model-free candidates under locked evaluation. Predictive improvement without control improvement must be reported.

## Data and protocol
- Sequence record: `episode_id, parent_source_id, scene_id, embodiment_id, simulator_name/version, model_version, observation_time, action_time, dt, rgb/depth availability, robot_state, action, tactile availability, contact/event labels, hash`.
- Preserve action units, reference frames, actuator limits, observation synchronization, physical randomization values and whether observations are real, synthetic or simulated.
- Train/val/test separated by **source episode/scene/physical variation**; future target, contact oracle and heldout labels never enter the predictor at inference.
- Same-state branches share a parent group; sample splitting and bootstrap must be group-safe.
- No raw video, private weights, copyrighted mesh/assets or API credentials uploaded without distribution audit.
- Freeze testing protocol, seeds, data hashes, model/preprocessing configs, metrics and acceptance thresholds **before** confirmatory evaluation.

## Baseline and ablation matrix
G0: faithful upstream LeWM at pinned commit on Push-T; real training and downstream evaluation. Store an exact up/downstream dependency lock and upstream attribution; expected score is unknown until execution.
G1: JEPA + SIGReg / no-SIGReg + frozen pretrained visual encoder where resource permits. Physical linear probe and covariance/effective-rank diagnostics.
G2: no action vs concatenation vs AdaLN/FiLM action injection; one-step and multi-step latent dynamics; same-state replay + true/shuffled action; event auxiliary supervision.
G3: vision vs tactile vs fused, with a synchronized tactile source and clear simulated-sensor label; validate provenance and simulator adapter separately.
G4: small conditional diffusion/flow matching on short future frames with heldout physical consistency; do not imply Cosmos-scale model training.
G5: heldout geometry/friction/object/initial-state/time horizon; candidate planning in simulator with prediction-versus-control analysis.
G6: independent freeze, confidence intervals, repeated seeds where feasible, reproducibility and limitations.

## Evaluation ownership
- World-model metrics: latent future-prediction loss (with explicit target encode policy), physical displacement/pose error, event F1 and timing, risk calibration, OOD split, multi-step drift.
- Representation metrics: per-dimension variance, effective rank/covariance, physical state probe on independently separated data. Never interpret arbitrary embedding scatter as physics mastery.
- Video-generation metrics: quantitative image/trajectory/contact-action agreement and paired qualitative failure sets; do not use cherry-picked videos as proof.
- Control metrics: task success and safety/rate/latency, actual rollout denominators; no simulated omniscient cost at test time.

## Evidence and statements
For every result record `environment + exact repo/branch/commit + dataset/hash + model/checkpoint/hash + hardware + command + wall/GPU time + seed + numerator/denominator + statistic + known limitations`. Separate status vocabulary: PLANNED, IMPLEMENTED, EXECUTED, VERIFIED, FAILED, BLOCKED.

A completed engineering gate does not imply increased model accuracy; a real simulation is not a real robot; end-to-end training is not foundation-scale pretraining; auxiliary pairwise supervision is not presumed novel. Preserve negative results.

## Resource and permission rules
CPU preparation and isolated software unit tests may proceed where technically available. Before chargeable GPU work, cloud creation, large external downloads, purchase, sensitive uploads or destructive changes, report exact costs and await approval. If machine access is unavailable, stop that step and provide explicit commands to the environment owner.

## Upstream
LeWorldModel official source: https://github.com/lucas-maes/le-wm (provisional source pin `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`, verify integration). Preserve MIT license and check stable-worldmodel/stable-pretraining and dataset licenses separately. This repository's existing theory/mathematical toy models remain credited baselines.
