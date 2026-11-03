# V1.1 audit report — 2026-11-03

## Evidence status

| Item | Status | Finding |
| --- | --- | --- |
| V1 source and raw artifacts | TESTED | Historical `artifacts/` and `results/` were read; none were overwritten. V1 code tip was `5145536`. |
| RSSM causal prediction | EXPERIMENTALLY VERIFIED | V1 decoded the posterior conditioned on `observation[t+1]` into the prediction of that same observation. This was future observation leakage. Fixed at `f6cb467`; a perturbation test verifies first-step invariance. V1 RSSM prediction metrics are superseded, not deleted. |
| One-step denominator | TESTED | V1 `one_step_mse` measured only the first sequence step. V1.1 averages all teacher-forced one-step predictions over branches, time and xyz; `first_step_mse` is separate. No masks are used because every V1.1 trajectory has 32 valid steps. State units are raw meters; actions are raw newtons. |
| Budget comparison | EXPERIMENTALLY VERIFIED | All five V1.1 conditions use 140 training branches, batch 14 and 100 optimizer steps per seed. Deterministic models have 10,275 parameters, RSSM 10,117. Actual CPU wall time differs and is reported; equal steps are not equal FLOPs. |
| MuJoCo replay | TESTED | `mjSTATE_FULLPHYSICS` is restored byte-for-byte before each branch. Mass and friction are fixed in each backend model and recorded per scene. Seven constant-force action sequences are logged. |
| Data split | TESTED | V1.1 validates scene-disjoint groups, equal branch initial bytes/physics, artifact/RGB hashes and disjoint mass-friction combinations across train/val/test. V1 had repeated physical configurations across splits and cannot support held-out physics claims. |
| V1 video PSNR | TESTED | V1 reported one 17-frame sample, including the known initial frame in the aggregate, and was not a dataset-level prediction benchmark. V1.1 excludes frame zero, aligns prediction `t` to RGB `t`, separates reconstruction/teacher forcing/free generation, and saves raw RGB comparisons. |
| Planner future access | TESTED | V1 offline selection ranked candidates from model predictions and used recorded futures only afterward for regret. V1.1 learned MPC/CEM call only model predictions at selection; physics rollouts are restricted to the oracle and environment feedback. |

## Limits

**IMPLEMENTED:** causal RSSM, strict V1.1 data split, corrected metrics, independent video evaluation and closed-loop pilot. **TESTED:** 47 automated tests before final reporting. **EXPERIMENTALLY VERIFIED:** fresh multi-seed runs and MuJoCo planning outputs are linked in the other V1.1 reports. **BLOCKED:** large CUDA model scaling on the installed CPU-only PyTorch wheel. **NOT RUN:** real robot and external official-model reproduction.
