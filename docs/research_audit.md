# Research audit — 2026-10-11

## Verified conceptual distinction

**VPP (ICML 2025):** Video diffusion predictive representations used to condition robot action generation. Source: https://proceedings.mlr.press/v267/hu25g.html

**OpenWAM (2026):** A composable world/action modelling framework with counterfactual same-state trajectories and separate forward/inverse dynamics training. Source: https://github.com/OpenWAM/OpenWAM

A comparison must be based on the full training/evaluation protocol. OpenWAM already collects same-state counterfactual trajectories; the research claim cannot be 'first counterfactual data for world models'.

## Prior pilot (outside this repository)

Prior small analytic RGB experiments compared:
- Independent action-event prediction
- Direct paired branch event-difference auxiliary loss
- Shuffled-pair control

Results were inconclusive or negative for paired supervision. These are **not** MuJoCo measurements nor reproductions of OpenWAM/VPP. They must not be cited as superiority evidence.

## Mathematical baseline

For an unknown monotone event threshold in [0,1], each noiseless bisection query halves the feasible interval. After n queries its width is 2^-n, and midpoint estimation has worst-case error 2^-(n+1). A binary-query decision tree has at most 2^n outcome leaves; some leaf's feasible threshold interval has length at least 2^-n. Hence bisection is minimax for this restricted model. This is a classical result, not a new research theorem.

## Falsification requirements

A proposed learned change operator must outperform independent conditional world modelling **with exactly matched data, compute and trainable parameter count** on held-out scenes. Use calibration, event-time error, physical trajectory errors, boundary subset error, and closed-loop decisions; audit for leakage, physical replay drift, and contact solver sensitivity.

Before claiming a theorem about RGB features, specify the observation model, identifiable quantities, support assumptions, and what is proved for a finite sample versus assumed about network optimization.
