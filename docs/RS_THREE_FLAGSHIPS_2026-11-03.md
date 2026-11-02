# Three flagship research programs — evidence contract (2026-11-03)

**The following are future acceptance gates, not completed results.** Starting point: existing InterventionalWorldModels, VLA-Forge, StartShift-VLA, ModelOps-Platform and reward-modeling-lab results. Reuse evidence without extending claims beyond original experiments.

| Level | Acceptance criterion |
| --- | --- |
| L1 / Research Engineer | real training, saved loadable weights, frozen held-out evaluation, reproducible code and traceable runtime |
| L2 / Strong Research Engineer | matched data and compute, strong baselines, >=3 independent seeds when feasible, appropriate confidence intervals, OOD and downstream outcomes |
| L3 / Research Scientist candidate | clear falsifiable research gap, full-text prior-art audit, genuine new mechanism, controlled ablations, independent confirmation and paper-quality open artifacts |

### P1: Reliable Action-Conditioned World Models
Primary: [InterventionalWorldModels](https://github.com/Benjamindaoson/InterventionalWorldModels)

Hypothesis: short-horizon fit is insufficient for physically reliable, action-faithful long rollouts. It is **unknown** whether paired difference supervision or future methods help.

- **W0 (in progress):** exact same-physics-state branch replay, action and result receipts with hashes, non-overlapping scene splits. CPU toy fixture tests are not physical evidence.
- **W1:** actually execute MuJoCo: vary mass/friction/contact/action, save real rollouts, engine and source hashes, freeze scene split and independent labels. MuJoCo adapter has not been validated in this environment.
- **W2:** capacity-, data- and compute-matched action-blind, action-conditioned and paired-difference predictors. Add shuffled-pair controls. Event F1, displacement error, Brier/ECE and near-contact boundaries; scene-level uncertainty.
- **W3:** RSSM / compact action-conditioned video prediction; teacher-forced vs free-running horizons, drift, zero/inverse action, OOD mass/friction/geometry, and action fidelity.
- **W4:** independent strong baseline reproduction and novelty gate against CoCo, OneWorld, OpenWAM and MiraBench, then a newly qualified intervention if justified.
- **W5:** downstream policy/physical planning improvement or a rigorously reported negative result.

### P2: World-Model-Guided Robot Learning
Primary: [VLA-Forge](https://github.com/Benjamindaoson/VLA-Forge); historical robustness evidence: [StartShift-VLA](https://github.com/Benjamindaoson/StartShift-VLA).

- **R0:** audit actual VLA-Forge M04 outcomes and *uncompleted M05 fine-tuning*; separately verify legacy ACT/Diffusion training receipts.
- **R1:** actually train ACT and Diffusion Policy, plus a frozen SmolVLA baseline; log weights/checkpoints, seeds, observations, action units and episode-level outcomes.
- **R2:** paired ID/OOD benchmarks by identical tasks/initial states. Do not causally interpret earlier nonpaired StartShift differences.
- **R3:** fixed real data and GPU budgets: plain SFT, equal random demonstrations, corrective data, world-model MPC, selective model-based RL, policy distillation. Independently evaluate in physics simulator.
- **R4:** real SO-101 only after independent servo, supply, calibration, rate limit, E-stop and supervised operation gates; preserve original success/failure video.

### P3: Scalable World Model Training & Real-Time Inference
Primary: [ModelOps-Platform](https://github.com/Benjamindaoson/ModelOps-Platform), reusing VLA-Forge sampler benchmark with careful scope.

- **S0:** freeze model/weights/device/precision/history/batch/quality targets, and define short prediction, long streaming video and whole-loop workloads.
- **S1:** compact world model actual GPU training plus pause/resume correctness; no CPU-only or dry-run claim.
- **S2:** real BF16, activation checkpoint, FSDP/DeepSpeed comparisons as appropriate. Throughput per processed sequence/frame, peak VRAM, GPU hours, multi-GPU scale efficiency at fixed quality.
- **S3:** CUDA Graph, async chunks and KV cache where applicable. Report P50/P95/P99, startup cost, end-to-end task timing, deadline misses, action equivalence, numerical tolerance and closed-loop success; existing 6.95x is only a warm FP32 sampler microbenchmark.
- **S4:** metadata, immutable checkpoint IDs, reproducibility, quality-based release gate and recovery tests.

## Four shared portfolio outputs

1. A genuine new world-model algorithm result **or** a technically valuable, rigorously verified negative study.
2. Physical robot-policy improvement under matched resources, verified on held-out tasks.
3. Real GPU training and inference system evidence.
4. Reproducible code, configs, frozen data manifests, model weights where licensable, raw results, docs and demo.

## Hard no-go

- A script, config, passing CI or prototype is not evidence of training or policy quality.
- A synthetic toy receipt is not real MuJoCo, real-world physics or a video model.
- A nonpaired comparison is not a causal shift effect.
- No publication status beyond verified actual submission/acceptance.
- No SO-101 autonomous control before physical safety/communication checks.

## Current kickoff result

The G0 paired replay module and **16 CPU unit tests** passed in an isolated project environment on 2026-11-03. The tests cover deterministic identical-state action replay, receipt integrity, invalid actions, corrupt restoration and split leakage. MuJoCo was not installed: the MuJoCo backend is provided as an **unexecuted adapter**, not a completed physics run. No world-model training, robot policy update or downstream benefit has been observed in this kickoff.
