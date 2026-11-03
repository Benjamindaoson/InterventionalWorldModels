# V2 state-model baseline and independent confirmation report

Status: **EXPERIMENTALLY VERIFIED (negative action-conditioning result)** on a small MuJoCo cube task. No SOTA, robot-transfer, or novel-method claim follows from this experiment.

## Frozen protocol and provenance

The protocol in [`CONFIRMATORY_PROTOCOL.md`](CONFIRMATORY_PROTOCOL.md) was committed before formal data generation. Dataset `artifacts/v2-3f97ef9-32x8-h64/` was generated from commit `3f97ef9351443adce84613d77a3c8e508cbdf626`; its manifest SHA-256 is `8c7815f8015ce075d20d6b44f4caecd875e6e1b3cd73e4a75c8802a0aa6730ea`. It contains 72 distinct scenes, 504 exact-snapshot action branches, 64 action steps and 65 observed states/frames per branch. There are 32 train scenes and eight each for IID validation, IID test, physics OOD, geometry OOD, and independent confirmation. Physics OOD and geometry OOD have distinct configurations. Confirmation labels were first used after all 24 prespecified model runs and three exploratory evaluations had completed.

The eight learned arms are `blind_pos`, `conditioned_pos`, `paired_pos`, `blind_vel`, `conditioned_vel`, `paired_vel`, `shuffled_vel`, and `rssm_vel`. Each ran seeds 17, 29, and 43 for 10 epochs and 160 AdamW steps with batch 14 and 224 training branches. All used the last checkpoint. The position-only and velocity-aware groups have different input dimensions, so comparisons across those groups are diagnostic rather than capacity matched. Action-blind and action-conditioned models within one group have the same architecture and parameter count; paired supervision adds objective computation and therefore is not FLOP matched. Every run manifest records wall time and checkpoint. Analytic identity and constant-initial-velocity baselines have no optimizer steps.

## Exploratory held-out results

Values below are xyz free-running trajectory MSE and same-scene final-step counterfactual-delta MSE, both in m²; learned rows average all three seeds. Full per-seed, per-scene results and contact metrics are in `results/v2-exploratory-geometryfix-ea981e6/`. The first `results/v2-exploratory-3f97ef9/` directory is an incomplete, superseded evaluation that used a fixed contact threshold; it is retained as an audit trail. The corrected evaluator uses each scene's actual box half-size plus 0.002 m for the approximate floor-contact proxy.

| Model | IID free | IID delta | Physics OOD free | Physics OOD delta | Geometry OOD free | Geometry OOD delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Identity | 0.01486 | 0.06389 | 0.01382 | 0.05926 | 0.00744 | 0.02137 |
| Blind position | 0.02264 | 0.06389 | 0.02185 | 0.05926 | 0.01500 | 0.02137 |
| Conditioned position | 0.03567 | 0.19806 | 0.03674 | 0.20023 | 0.03371 | 0.18026 |
| Paired position | 0.03383 | 0.18337 | 0.03531 | 0.18808 | 0.03238 | 0.16816 |
| Blind position + velocity | 1.76673 | 0.06389 | 1.76492 | 0.05926 | 1.75776 | 0.02137 |
| Conditioned position + velocity | 0.77369 | 5.55453 | 0.77398 | 5.55228 | 0.76688 | 5.51223 |
| RSSM position + velocity | 0.01163 | 0.03395 | 0.01226 | 0.03494 | 0.00813 | 0.00578 |

The shuffled-pair velocity arm also ran all seeds; its IID free MSE was 2.19693 m² and delta MSE 16.80085 m². It is included in the raw matrix. The large velocity-arm means are genuine divergence, including `blind_vel` seed 43, and no seed was excluded.

## One-time independent confirmation

`results/v2-confirmatory-first-3f97ef9/matrix.json` contains all 24 learned runs, two analytic baselines, 8 scene-level rows per arm/seed, and the prespecified scene bootstrap. The primary statistic, **blind_vel minus conditioned_vel** final-step counterfactual-delta MSE, was **−5.40476 m²**, with 2,000-draw scene bootstrap 95% interval **[−6.86249, −3.92626] m²**. A positive value would have supported the primary action-conditioning hypothesis. All eight scene differences are negative.

| Model | Confirmatory free xyz MSE (m²) | Confirmatory counterfactual-delta MSE (m²) |
| --- | ---: | ---: |
| Identity | 0.00831 | 0.02725 |
| Blind position | 0.01578 | 0.02725 |
| Conditioned position | 0.03463 | 0.18416 |
| Paired position | 0.03365 | 0.17465 |
| Blind position + velocity | 1.75757 | 0.02725 |
| Conditioned position + velocity | 0.75703 | 5.43201 |
| RSSM position + velocity | 0.00916 | 0.01244 |

For the primary velocity arms, all individual seed outcomes are preserved: blind delta MSE was 0.02725, 0.02725, and 0.02725 m²; conditioned delta MSE was 0.91775, 1.48453, and 13.89376 m² for seeds 17, 29, and 43. The confirmation set is only eight scenes and is a pilot. Bootstrap interval describes scene variability conditional on this small protocol, not broad physical-world generalization.

## Interpretation and limitations

The simple position/velocity residual predictor minimizes one-step loss but is unstable when its own predictions feed the next step. Pairwise supervision did not solve this. RSSM has lower 64-step trajectory error on IID and both OOD families, yet its first-step error is relatively high and its action causal effect still requires planning validation. The low-error identity baseline exposes the weak-motion structure of this small cube task. Contact F1 is a geometric floor-contact proxy, not a learned contact classifier. No model was selected or retrained after viewing the confirmation set. The next model revision requires a new independent confirmation family.
