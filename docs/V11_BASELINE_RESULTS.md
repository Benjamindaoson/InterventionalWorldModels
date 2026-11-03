# V1.1 baseline results — exploratory held-out physics pilot

**EXPERIMENTALLY VERIFIED:** 32-scene MuJoCo dataset (20 train, 6 validation, 6 test), seven paired branches per scene, 32 steps. Dataset manifest SHA-256 `83576fe2cf07df02f7df318f4d9fb5089e2f8f17270d0b50e121d16ba01b874d`; data generator commit `f6cb467715ced5caa40617a220b998209cb06f21`. Training/evaluation commit `815387b5cbb0fcc39a258cafd39550d42b777f79`. Raw run directories: `artifacts/v11-baselines-815387b/`; independently recomputed test files, per-scene rows and curve: `results/v11-815387b-analysis/`.

All arms used seeds 17, 29, 43, 140 train branches and 100 optimizer steps per seed, batch 14. The deterministic arms have 10,275 parameters; RSSM has 10,117. Test includes 42 branches in six physics-disjoint scenes. MSE is in squared meters over xyz. Confidence intervals resample the six scenes after averaging three seeds; they quantify only this small test set and do not establish a population-wide effect.

| Model | One-step MSE mean | Free MSE mean ± seed SD | 95% scene bootstrap CI | Contact F1 mean | CPU train seconds/run |
| --- | ---: | ---: | ---: | ---: | ---: |
| Action-blind | 0.00000176 | 0.000199 ± 0.000109 | [0.0000685, 0.000367] | 0.925 | 1.47 |
| Action-conditioned | 0.00000902 | 0.002660 ± 0.000566 | [0.001701, 0.003800] | 0.707 | 1.48 |
| Conditioned + paired differences | 0.00000693 | 0.002076 ± 0.000996 | [0.001170, 0.003201] | 0.788 | 1.48 |
| Shuffled-pair control | 0.00001149 | 0.004318 ± 0.003144 | [0.002207, 0.006546] | 0.747 | 1.50 |
| Causal RSSM | 0.007683 | 0.008090 ± 0.000136 | [0.004173, 0.012164] | 0.000 | 6.62 |

Contact F1 uses a fixed threshold on predicted cube-center height and logged MuJoCo contact. It is **not** a learned contact head. Action-branch difference error and full precision/recall are in the raw JSON. Blind prediction wins on overall state MSE; the seven forces and 0.16-second horizon produce small displacements, so persistence is strong. Paired supervision improves over the conditioned model here but the result is exploratory and does not overcome the blind baseline. RSSM uses roughly matched parameters/steps but substantially more CPU time; this is not an equal-FLOP comparison.

Calibration is `NOT_AVAILABLE` in the evaluator because none of these checkpoints produces a calibrated state-space probability. The evidence ledger at `results/v11-evidence-ledger.json` records SHA-256 values for final checkpoints, loss logs, metrics and the dataset manifest.

**IMPLEMENTED / TESTED:** model training, full checkpoint states, corrected one-step denominator, teacher forcing and free rollout, 47 tests. **BLOCKED:** large CUDA training on the installed CPU PyTorch build. **NOT RUN:** independent confirmatory dataset, equal-FLOP control, official V-JEPA 2-AC comparison, real robot.
