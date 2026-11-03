# V1.1 frozen pilot protocol

**IMPLEMENTED / TESTED:** `scripts/generate_v11_data.py` creates 20 train, 6 validation and 6 test scenes, each with seven exact-snapshot branches and 32 steps. Train mass/friction pairs: (0.5,0.3), (1.0,0.5), (1.5,0.4), (0.8,0.7); validation: (1.2,0.2), (1.8,0.6); test: (2.0,0.8), (0.6,0.9). Initial xyz and height vary; height 0.4 m gives non-contact trajectories, 0.05/0.055 m produce contact or near-contact states. Forces include zero, ±2/±5 N in x and ±5 N in y. Each scene's branches start from identical full physics state and parameters.

**EXPERIMENTALLY VERIFIED:** Five arms use seeds 17/29/43, 10 epochs, batch 14, AdamW at 0.001, 100 optimizer steps and 140 train branches each. Arms: action-blind (zero action input), conditioned, conditioned plus same-scene final-state difference loss (weight 0.1), shuffled-pair target control (same auxiliary weight with shuffled action-branch targets), and causal RSSM. Deterministic width 96 gives 10,275 parameters; RSSM width 32 gives 10,117. `last.pt` is the predeclared evaluation checkpoint. Validation is recorded but not used to drop seeds. The shuffled control alters pair supervision while preserving the ordinary action-conditioned reconstruction target.

**TESTED:** Test metrics average over all 42 held-out branches and 32 time steps. Horizon MSE uses exactly steps 1/4/8/16/32. Contact F1 applies a fixed cube-center rule `z <= 0.052 m` to predicted state and compares with logged MuJoCo contact; it is not a trained contact head. Scene bootstrap resamples six independent test scenes (2,000 draws, RNG 123) after averaging over all three model seeds. Raw per-scene metrics, checkpoints, loss logs, CSV and SVG remain under local ignored `artifacts/` and `results/`.

**BLOCKED:** equal-FLOP comparison and larger video scaling require additional compute accounting/hardware. **NOT RUN:** independent confirmatory split beyond this pilot and cross-engine/real-robot validation. Test results were inspected during implementation, so this is exploratory evidence despite frozen files; no confirmatory superiority claim is made.

## Reproduction commands

Use Python 3.12 and `python -m pip install -e ".[test,research]"`. Output directories below must be new; use a different suffix if they already exist.

```bash
python scripts/generate_v11_data.py --output artifacts/repro-v11-data --train-scenes 20 --val-scenes 6 --test-scenes 6 --horizon 32
python scripts/validate_data.py artifacts/repro-v11-data
python scripts/run_v11_baselines.py --dataset artifacts/repro-v11-data --output artifacts/repro-v11-runs --seeds 17 29 43 --epochs 10
python scripts/summarize_v11.py --runs artifacts/repro-v11-runs --dataset artifacts/repro-v11-data --output results/repro-v11-analysis
python scripts/train_v11_video.py --dataset artifacts/repro-v11-data --output artifacts/repro-v11-video --epochs 3 --batch-size 2
python scripts/evaluate_v11_video.py --checkpoint artifacts/repro-v11-video/best.pt --dataset artifacts/repro-v11-data --output results/repro-v11-video --steps 4
python scripts/evaluate_v11_planning.py --checkpoint artifacts/repro-v11-runs/conditioned-s17/last.pt --output results/repro-v11-planning-s17.json --scenes 6
python scripts/build_v11_ledger.py --dataset artifacts/repro-v11-data --runs artifacts/repro-v11-runs --analysis results/repro-v11-analysis --video-run artifacts/repro-v11-video --video-test results/repro-v11-video --planning results/repro-v11-planning-s17.json --output results/repro-v11-ledger.json
python -m pytest -q
```
