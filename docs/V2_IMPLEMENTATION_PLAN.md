# V2 implementation plan — gated execution

Last revised: 2026-10-15. **No G0–G6 training has been performed by this document.** Primary source repository: `Benjamindaoson/InterventionalWorldModels`.

## Execution locations (verify before operating)
| Layer | Proposed path | Responsibility |
|---|---|---|
| Windows developer | `D:\robotics\InterventionalWorldModels` | Git checkout, edits/tests, authoring configs and docs. **Actual path not verified** from GitHub access. |
| WSL optional | `/mnt/d/robotics/InterventionalWorldModels` | Only if Windows path verified; not a second authoritative checkout. |
| GitHub | `feat/world-model-v2` | Branch, source-of-truth commit history, small numeric reports, CI. No model weights/videos. |
| AutoDL GPU (pending access) | `/root/autodl-tmp/InterventionalWorldModels` | Runs at pinned GitHub commit, no speculative local changes. |
| AutoDL data/runs (pending) | `/root/autodl-tmp/IWM-data` and `/root/autodl-tmp/IWM-runs` | Large data/checkpoints/logs with persistence/storage policy verified first. |

### G0 mandatory preflight
1. Windows PowerShell: `Get-Location`, `Test-Path 'D:\robotics\InterventionalWorldModels\.git'`, `git -C 'D:\robotics\InterventionalWorldModels' remote -v` **only if directory exists**. If absent, check `$HOME\InterventionalWorldModels\.git` and move/clone safely only after checking both destinations and uncommitted work.
2. Fetch origin and inspect both local/remote status. NEVER hard reset, force push or overwrite. Checkout `feat/world-model-v2` only after confirming working tree changes.
3. Inventory current sources and tests. Retain legacy code; review `docs/V2_BASELINE_AUDIT.md` and `docs/V2_RESEARCH_CONTRACT.md`.
4. Pin LeWM to commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` **provisionally**. Review exact upstream `README.md`, `jepa.py`, `module.py`, `train.py`, Hydra configs and package runtime dependencies. Do not transplant upstream code without MIT attribution and dependency audit.
5. Add only needed config/runner/tests for **faithful** Push-T baseline. First CPU import/shape/forward-backward smoke; then cloud GPU preflight (not full training). Use upstream `STABLEWM_HOME` for dataset location after checking actual storage.
6. Check CUDA/PyTorch/GPU VRAM/scratch disk/instance rate/data license/download size. Produce proposed hours and max spend; **pause for user approval** before paid training.
7. After approval and actual run: save training receipt, effective batch, optimizer steps, checkpoint SHA, exact model configuration, independent eval, per-run time/cost and discrepancies. Put compact evidence in `reports/G0/`, link only authorized large assets.
8. G0 acceptance requires (i) source pinned and license preserved, (ii) data hashed/disjoint where applicable, (iii) real GPU optimizer updates, (iv) checkpoint saved and independently reloaded, (v) non-Mock evaluation, (vi) CPU/CI tests and artifact manifest, (vii) code pushed back from local developer. A model imported or checkpoint downloaded does **not** pass.

### Gates after G0
| Gate | Priority | Required deliverable | Fail-closed condition |
|---|---|---|---|
| G1 Representation | P0 | JEPA/SIGReg experiments, frozen-vs-trainable encoder, covariance/collapse measures and physical probes. | Missing raw runs, contaminated probe split, or skipped loss ablation. |
| G2 Dynamics | P0 | MuJoCo pixel/action/source dataset; matched same-state replay; true/zero/shuffled-action comparison; contact/time predictions. | Model cannot outperform controls or no real physics execution: report negative/blocked. |
| G3 Visuo-tactile | P1 | Reviewed SAPIEN-IPC/ManiSkill tactile data adapter, synchronized observations, modality experiments. | No licensed tactile data, no sensor synchronization or only placeholder channel. |
| G4 Generative | P1 | Train lightweight action-conditioned flow/diffusion future-frame model; real heldout predictions and consistency. | Only decoder reconstruction, no generative training or incompatible 24 GB GPU. |
| G5 OOD & planning | P1 | Heldout physics/geometry plus learned model-only action scoring and rollout comparisons. | Future simulator information leaked into planner. |
| G6 Research closeout | P1 | Immutable experiment matrix, metrics/confidence intervals, cost, failure cases, reproduction docs. | Incomplete provenance or unsupported improvement claim. |

### Code modules (create on demand)
- `src/iwm/data/`: episode schema, source hashes, split checks, loader.
- `src/iwm/models/`: explicit encoder/predictor/action conditioning/optional tactile/video parts.
- `src/iwm/training/`: trainer, checks, resume and immutable logging.
- `src/iwm/evaluation/`: physical probes, latent dynamics, contact events, generalization.
- `src/iwm/simulation/`: MuJoCo snapshot and optional separate SAPIEN tactile interface.
- `configs/`, `scripts/`, `tests/`, `reports/`: only implemented, runnable workloads. Preserve legacy files and do not create empty facade modules.

### Git publication and CI
Use stage-specific commits on `feat/world-model-v2`. For each stage, report Windows/WSL/AutoDL path, branch, source SHA, exact changed files, tests/execution (explicitly NOT RUN when unavailable), costs and artifact locations. PR/merge to main only after review; no force push. GitHub Actions YAML existing tests are not equivalent to confirming passing CI; inspect actual run status.

### Stage state when authored
GitHub source audit: DONE. Research contract and plan: DOCUMENTED. Windows local clone state: UNKNOWN. AutoDL access: UNKNOWN. G0 model integration and CPU tests: NOT RUN. GPU: NOT AUTHORIZED / NOT RUN. G1–G6: PLANNED only.
