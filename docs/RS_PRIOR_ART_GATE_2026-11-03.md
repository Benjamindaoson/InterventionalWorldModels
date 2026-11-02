# Prior-art / novelty gate (2026-11-03)

**Verdict: NO VALIDATED ALGORITHMIC NOVELTY YET.** All candidate contributions are hypotheses.

| Closest work | Overlap with our ideas | Why it matters |
| --- | --- | --- |
| [MiraBench](https://arxiv.org/abs/2605.29360), May 2026 | action following, physical validity, optimism bias | A failure-oriented benchmark is not by itself novel |
| [CoCo](https://arxiv.org/abs/2608.04653), Aug 2026 | zero/inverse-action consistency and counterfactual controls | "Action consistency" alone is not novel |
| [OneWorld](https://arxiv.org/abs/2609.30946), Sep 2026 | shared physical mechanism across intervention branches | "Consistent physics across branches" alone is not novel |
| [OpenWAM](https://arxiv.org/abs/2610.07922), Oct 2026 | coupled video/action generation and counterfactual FDM/IDM | Same-state counterfactual branches themselves already exist |
| [Video Prediction Policy](https://proceedings.mlr.press/v267/hu25g.html), ICML 2025 | future visual predictions for robot control | Must compare against established video-aided planning |

The original InterventionalWorldModels compact RGB paired-loss pilot **did not** establish a robust benefit. Preserve the negative result and all limitations.

## Before calling anything original

1. Conduct full-text comparison rather than relying on abstracts.
2. Verify and freeze benchmark implementations, data provenance, compute and action conditions.
3. Reproduce the closest strong baselines, including null/shuffled-pair and capacity-matched controls.
4. Pre-register an independent physical reliability outcome and an independent downstream planning outcome.
5. Keep scene/task-level holdouts, >=3 seeds where feasible, valid confidence intervals and negative results.
6. Only then decide whether a new mechanism meaningfully differs and improves on prior art.

**Exploratory idea, not an algorithmic novelty claim:** adaptive imagination budgeting using cross-action disagreement and physics-calibrated confidence. Must first audit prior work on uncertainty-guided MPC, selective rollouts, model exploitation and adaptive lookahead.
