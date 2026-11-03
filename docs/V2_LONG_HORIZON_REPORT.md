# V2 long-horizon rollout: 64-step MuJoCo cube trajectories

Status: **EXPERIMENTALLY VERIFIED** for state models on eight IID scenes and two sets of eight OOD scenes. Video and visual-token rollout are separate experiments.

## Information boundary and metric

Teacher forcing predicts state `t+1` after seeing the real state at `t`. Free running sees only initial state `0` and the specified action sequence; all later state inputs are its own predictions. RSSM decodes its prior before updating a posterior with `t+1`; free running never reads a future posterior. Position MSE averages the three xyz coordinates, branches, and aligned time indices. The full 64-element error vector is stored in each run's evaluation JSON. Horizon 1, 4, 8, 16, 32, and 64 can therefore be inspected without interpolation. The final-step counterfactual statistic compares action branches within an identical MuJoCo snapshot, never across unrelated scenes.

The figure `results/v2-horizon-plots-ea981e6/horizon_mse.png` plots mean free-running xyz MSE over all three seeds on a log scale. Its `horizon_mse_raw.csv` has all 3,456 per-seed/per-horizon values and source JSON paths. Reproduce with `python scripts/plot_v2_horizons.py --evaluations results/v2-exploratory-geometryfix-ea981e6 --output <new-output-directory>` after installing the optional Matplotlib plotting dependency.

| IID test model | H1 | H4 | H8 | H16 | H32 | H64 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Blind position | 0.00000 | 0.00005 | 0.00021 | 0.00096 | 0.00759 | 0.10534 |
| Conditioned position | 0.00003 | 0.00041 | 0.00149 | 0.00501 | 0.01751 | 0.15611 |
| Blind position + velocity | 0.00001 | 0.00021 | 0.00101 | 0.00613 | 0.07018 | 20.09177 |
| Conditioned position + velocity | 0.00003 | 0.00049 | 0.00217 | 0.01152 | 0.11599 | 5.15384 |
| RSSM position + velocity | 0.01049 | 0.01151 | 0.01185 | 0.01128 | 0.00678 | 0.03175 |

Values are m² and means of seeds 17, 29, 43. The near-zero H1 entry rounds a positive value to five decimal places; raw files retain full precision. The dramatic blind-velocity H64 mean contains an unstable seed and is intentionally retained. The RSSM curve is comparatively flat through H32 but has poor H1 accuracy; lower long-run error alone does not establish useful physical dynamics. The physics OOD and geometry OOD curves are in the same figure and raw CSV. The confirmation set was evaluated once under the preregistered protocol, with separate raw files in `results/v2-confirmatory-first-3f97ef9/`.

## Open questions

Action-conditional force response has the wrong sign in the earlier V1.1 causal probe and remains poor in V2 confirmation. A position-only identity forecast can outperform learned models for these slowly moving scenes. These failures suggest that one-step reconstruction, long-horizon stability, and counterfactual correctness must be optimized and reported separately. No geometry loss, state re-anchoring, or uncertainty-calibrated rollout improvement has yet been demonstrated on this V2 dataset.
