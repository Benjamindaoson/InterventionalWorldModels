# Local pilot experiment report — 2026-11-03

## Audit and provenance

The local directory was empty. Full remote history was cloned from `research/rs-g0-paired-replay-20261103`; local branch `feat/v1-core-20261103` was created. Draft PR #2 remains untouched. Earlier negative toy/RGB findings are preserved in historical documents. Final v3 MuJoCo data was generated from commit `6fb2b8273820965bb21c8c7d7850bcf8d48b44b9`; dataset manifest SHA-256: `D526847F486AD767C7306892D663685AC2A348354D336EEA5DE404AAF5332FD8`. Earlier v1/v2 artifacts remain local but have weaker provenance or split evidence and are excluded here.

## Runtime and data

Windows, Python 3.12, torch 2.15.0+cpu, MuJoCo 3.17.0. GTX 1650 (4 GiB) detected, but CUDA is unavailable in the installed wheel. About 296 GB disk was free at first audit. Twelve MuJoCo cube scenes, 36 same-state action branches, 16 steps and 32×32 RGB; 8/2/2 scenes for train/validation/test. Branches restore the same full initial state within each scene. Identical initial states across frozen splits are rejected.

## Actual CPU runs

| Model or control | Training | Measured result |
| --- | --- | --- |
| RSSM | 5 epochs, seed 17 | test one-step MSE 0.003509; free-running MSE 0.001731 |
| Deterministic action predictor | 5 epochs, seed 17 | test one-step MSE 0.002513; free-running MSE 0.169207 |
| Four-arm matched-step control | 5 epochs, seed 17 | conditioned 0.147516; shuffled 0.144570; blind 0.140938; paired 0.135443 free-running MSE |
| Visual representation predictor | 1 epoch | final train loss 1.406392; no validation metric |
| Video flow | 1 epoch | final train loss 1.237128; one 17-frame sample PSNR 14.3326 dB |
| Joint world-action | 1 epoch | final train loss 5.075450; no validation metric |

State results are saved under `results/rssm-v3-deterministic-test.json` and `results/action-v3-test.json`; generated RGB is `artifacts/video-v3-sample.npz`. State checkpoint resume was exercised, and automated tests cover it. RSSM deterministic posterior evaluation was fixed after data generation; the data-generation source commit remains accurate. Scores are single-seed, six-branch smoke measurements; they establish no superiority or generalization.

## Verification and limits

`pytest`: 44 passed. Ruff passed for new code. MuJoCo rendering, generation, forward/backward, checkpoint loading/resume, future-state prediction, video generation, frozen evaluation and four-arm ablation ran on CPU. No CUDA test, official V-JEPA adapter, reward/termination training, full joint comparison, contact F1, OOD matrix, independent seeds, closed-loop planning or real-robot test was completed. L1 is partial; L2/L3 are not passed. Scaling needs a CUDA-enabled PyTorch build and larger compute; no cloud resources or large weights were acquired.

Offline fixed-candidate action selection on two test scenes: RSSM mean x-regret 0.000000 m; deterministic predictor 0.009511 m. These are two choices from three recorded candidate actions and do not demonstrate general planning utility or closed-loop success. Raw decisions are in `results/rssm-v3-action-selection.json` and `results/action-v3-action-selection.json`.
