# V2 baseline audit — 2026-10-15

## Scope and authority
Source: GitHub `Benjamindaoson/InterventionalWorldModels`, default branch `main` at `c49cb7ca215762adaa1d30b381a58c2f8be7fecd`. This audit inspected public repository source via GitHub. It did NOT inspect the user's Windows drive, run pytest, connect to AutoDL, download model weights, or perform GPU training. All conclusions below are source-level unless explicitly documented otherwise.

## Actual v1 tree and evidence
| File | Observable implementation | Limitation |
|---|---|---|
| `src/iwm/learning.py` | `VisualBranchModel`: 3-layer convolutional RGB encoder, action concatenation, 2-output head (event logit and displacement). Masked independent and paired-difference loss, one-epoch train function, event accuracy/Brier/near-zero-proxy evaluator. | No learned latent state transition `z(t+1)`, no image/video decoding, no JEPA, no SIGReg, no published checkpoint or executed GPU run. Existing `test_learning.py` verifies forward/backward on random tensors, not physical learning. Docstring says "RGB history" but actual input is `[B,3,H,W]` single frame. |
| `src/iwm/mujoco_data.py` | Optional MuJoCo box-and-plane generator varying force and friction; output displacement/threshold label. | Source reports generator unexecuted; no pixel frames, robot action trajectory, tactile readings, event labels, replay receipts, or scene-disjoint train/val/test manifests. |
| `src/iwm/experiment_protocol.py` | Run fingerprint dataclass and scene-level overlap rejection. | No dataset registry, source-origin hashes, policy/checkpoint identity or test-set immutable seal. |
| `src/iwm/graph_theory.py` | Deterministic graph error bound. | Not neural physical-generalization evidence. |
| `src/iwm/thresholds.py` | Classical monotone 1D bisection and abstention baseline. | Not an original ML method, real-world simulation or world-model training. |
| `tests/` | Threshold, graph, split, manifest and light tensor-gradient tests. | No simulator integration, LeWM training, checkpoint/resume, held-out evaluation or GPU acceptance tests. |
| `.github/workflows/tests.yml` | GitHub Actions CPU test workflow. | Current source-only audit did not execute CI and does not infer current run status from YAML. |
| `docs/RESEARCH_PROPOSAL.md` + `docs/research_audit.md` | Research proposal and explicit prior-work review. | Pairwise auxiliary-loss superiority unproven; no physical benchmark/reproduction demonstrated. |

## Existing negative and boundary evidence
- Prior toy RGB pilots did **not** reliably demonstrate that paired-branch supervision improves over independent action supervision. Keep this negative result in reports and comparisons.
- Original branch threshold and graph certificates are restricted to defined mathematical assumptions. Do not describe them as a novel physical-world identifiability theorem.
- OpenWAM and related work already use counterfactual/same-start supervision; do not claim to be first.
- Existing `docs/implementation_status.md` says optional MuJoCo source is not executed in that environment; no GPU LeWM reproduction or cross-engine evaluation confirmed.

## Upgrade decision
Retain all v1 implementation, tests, and research notes. Do not overwrite or delete `src/iwm/learning.py`, `mujoco_data.py`, `thresholds.py` or `graph_theory.py` before a tested compatibility migration. Create v2 packages only when needed for running features, not empty facade directories.

G0 aims to reproduce upstream LeWorldModel on Push-T, with an explicit external source hash and independent execution receipts. The upstream `lucas-maes/le-wm` main commit observed on 2026-10-15 is `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (provisional pin until training integration verifies dependencies). The upstream source is MIT, copyright Lucas Maes (2026); transitive dependencies/dataset redistribution licenses require separate audits.

## Development environments (not yet verified)
- Proposed local working tree: `D:\robotics\InterventionalWorldModels`. User previously cloned from `C:\Users\Admin（无密码）` and reported that D: was missing at the time; current actual location is UNKNOWN until checked locally. Never assume it moved.
- GitHub is the source of truth, branch `feat/world-model-v2`. This branch's initial files are preliminary planning artifacts, not source or training implementation.
- Proposed AutoDL code/data/runs: `/root/autodl-tmp/InterventionalWorldModels`, `/root/autodl-tmp/IWM-data`, `/root/autodl-tmp/IWM-runs`. Connection, disk persistence, GPU and billing are UNKNOWN.

## Gate state at authoring
- Remote repository read audit: COMPLETE.
- V2 GitHub preparatory documentation: IN PROGRESS.
- Windows working tree status: NOT VERIFIED.
- Local CPU checks: NOT RUN in user's Windows environment.
- Upstream LeWM integration: NOT STARTED.
- GPU smoke/full reproduction: NOT RUN.
- G1–G6: NOT STARTED.

Evidence provenance: repository source paths listed above and upstream https://github.com/lucas-maes/le-wm . This document is an audit, not an experimental results report.
