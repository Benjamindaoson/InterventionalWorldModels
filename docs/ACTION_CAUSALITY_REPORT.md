# Action causality diagnosis — V1.2 exploratory audit

Source: six held-out V1.1 MuJoCo physics scenes, seven exact-snapshot branches each, seed-17 checkpoints in `artifacts/v11-baselines-815387b/`. Raw scene/branch rows and mass/friction strata: `results/v12-action-causality-complete/`. Metric code: `src/iwm/causality.py` and `scripts/diagnose_action_causality.py`.

| Predictor | True mean branch delta m | Predicted delta m | Counterfactual delta MSE m² | Direction accuracy | Mean d(final x)/d(Fx) m/N |
| --- | ---: | ---: | ---: | ---: | ---: |
| Identity | 0.01953 | 0 | 0.000468 | 0 | 0 |
| Constant initial velocity | 0.01953 | 0 | 0.000468 | 0 | 0 |
| Action blind | 0.01953 | ~0 | 0.000468 | 0.139* | 0 |
| Conditioned | 0.01953 | 0.15929 | 0.014926 | 0.528 | -0.03686 |
| Paired | 0.01953 | 0.14610 | 0.013120 | 0.639 | -0.02119 |
| Shuffled pair | 0.01953 | 0.23585 | 0.034841 | 0.028 | -0.05269 |
| RSSM | 0.01953 | 0.03640 | 0.000832 | 0.833 | 0.00450 |

*The tiny nonzero action-blind direction score is floating-point noise from separately evaluated identical branch inputs; its practical action sensitivity is zero. Constant velocity equals identity because V1.1 initial qvel is zero. These baselines use only the allowed initial state.

For seed 17, conditioned trajectory MSE is 0.00136 on 28 contact branches and 0.00721 on 14 non-contact branches. Identity obtains 0.00000621 on contact branches but 0.00130 on non-contact branches. This mix explains why aggregate position MSE favors persistence, while action-conditioned rollouts overreact. The conditioned model's reverse-sign local gradient is a separate action-response failure. The data show both weak on-ground action effects and an under-observed moving/free-fall regime; causal attribution between data scale, optimization and missing velocity still requires a controlled new dataset.

**Status:** IMPLEMENTED and CPU_VERIFIED. This is exploratory diagnosis, not a confirmed general result. Next dependency: vary initial velocity and geometry, expose velocity to a capacity-matched model, and preregister an untouched confirmatory split before training.
