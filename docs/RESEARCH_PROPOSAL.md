# Student-led independent research — faculty review draft

**Working title:** Interventional World Models: Learning Action-Outcome Differences from Sparse Visual Evidence

**Status:** No faculty advisor or laboratory affiliation at present. Target: CVPR main conference; oral is an aspiration, not an established result.

## Scientific question
Can sparse, same-state physical interventions teach action-conditioned visual models to predict differences between unexecuted futures, especially near physical mode transitions?

## Closest research
Video Prediction Policy (ICML 2025) uses predictive video representations for robot action generation. OpenWAM (2026) already trains on same-state counterfactual trajectories, with forward/inverse dynamics and extensive evaluations. We do not claim that collecting counterfactual branches is new. The candidate extension is explicit cross-branch difference supervision under matched data, capacity and compute.

## Method hypothesis
Encode pre-action RGB, condition a shared predictor on each candidate action, predict event probability and physical displacement, and optionally supervise observed same-state outcome differences. Compare against independent action supervision and shuffled-branch controls. The current prototype is a compact RGB model, not a full generative video foundation model.

## Research design
E0 replay and physical event label audit; E1 indistinguishable-world test; E2 independent visual baseline; E3 paired-branch objective; E4 sparse observation budgets; E5 ablations including shuffled pairs, matched extra capacity, and pair loss removal; E6 held-out physics, geometry and appearance; E7 cross-engine; E8 closed-loop manipulation; E9 independent confirmatory evaluation.

Metrics: event F1, near-boundary error, physical displacement, event timing, Brier calibration, inference cost, and closed-loop success. Freeze scene-level splits and report seeds and uncertainty intervals.

## Theory and limitations
See THEORY_COMPENDIUM.md and theory.md. The proved results are conditional and largely based on established mathematics. A new general-RGB identifiability theorem, verified physical-simulation dataset and state-of-the-art reproduction have not been completed. Prior simplified paired-supervision pilot did not establish an advantage.
