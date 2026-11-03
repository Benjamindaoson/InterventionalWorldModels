# Public branch release audit — 2026-11-03

Target: `feat/worlddynamics-v2-complete`, with common ancestor `c49cb7ca215762adaa1d30b381a58c2f8be7fecd` on `origin/main`. This release adds code and research reports on an independent branch; it does not rewrite or merge `main` or PR #2 (`research/rs-g0-paired-replay-20261103`).

## Scope and sensitive data

`scripts/release_audit.py` scanned **211 reachable Git blobs across all local refs** at preflight. It found **zero** high-confidence private key, GitHub/OpenAI/AWS credential, password assignment, sensitive filename, model-weight/data/video artifact, or blob over 10 MB. The scan reports object IDs and rules only, never secret values. It is a heuristic audit, not a formal proof of absence. `.gitignore` excludes `.venv/`, cache directories, `artifacts/`, and `results/`; `git ls-files` confirms that no `.pt`, `.pth`, `.ckpt`, `.npz`, or `.mp4` is tracked. No `.env` or private key is tracked. Historical author names and public research URLs are part of the existing Git history.

Only source, tests, configs, README, preregistration, negative-result reports, and small Markdown evidence summaries are intended for this branch. Local MuJoCo RGB datasets, 24 state-model checkpoints, WAM/video checkpoints, raw per-scene JSON, GIFs, and plots remain in ignored `artifacts/` and `results/` and are **not** uploaded. The local V2 dataset manifest SHA-256 is `8c7815f8015ce075d20d6b44f4caecd875e6e1b3cd73e4a75c8802a0aa6730ea`; source revision `3f97ef9351443adce84613d77a3c8e508cbdf626`. Reproduction uses the documented generator and frozen protocol.

## Local release gate

- `python -m pytest -q`: **57 passed**.
- Ruff on new/revised release-critical modules: **passed**. A whole-repository Ruff scan still reports 20 pre-existing style issues in older historical files; these are not suppressed or described as green.
- Existing training-resume, MuJoCo branch-contract, future-observation leakage, paired-noise, planner-information-boundary, and checkpoint-load tests are included in the 57 passing tests.
- Direct load of `artifacts/v2-baselines-3f97ef9/conditioned_vel-s17/last.pt` into the 10,854-parameter model: **passed**.
- CUDA/GPU training: **NOT_RUN**; the installed local PyTorch wheel is CPU-only. No GPU result is claimed.

The negative V2 confirmatory statistic, video action-fidelity failure, and weak closed-loop planner outcomes are explicitly retained in the public reports. Documentation is not a substitute for raw data release; external researchers must regenerate the dataset or obtain a separately published artifact package to independently inspect every raw trajectory.
