# V1.1 long-horizon and video report

**EXPERIMENTALLY VERIFIED:** State test errors at exact horizons (mean over three seeds, six held-out scenes, 42 branches, xyz MSE in m²):

| Model | H1 | H4 | H8 | H16 | H32 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Blind | 0.00000180 | 0.0000228 | 0.0000693 | 0.000154 | 0.000647 |
| Conditioned | 0.0000103 | 0.000151 | 0.000555 | 0.001949 | 0.007723 |
| Paired | 0.00000816 | 0.000119 | 0.000431 | 0.001501 | 0.006153 |
| Shuffled-pair | 0.0000110 | 0.000170 | 0.000666 | 0.002688 | 0.014288 |
| RSSM | 0.010062 | 0.009438 | 0.009666 | 0.008616 | 0.005204 |

The RSSM curve starts high and falls because it collapses toward a stable mean; this is **not** proof of superior long-range dynamics. Teacher-forced one-step and free-running errors are separate fields in each test JSON. The curve and raw values are `results/v11-815387b-analysis/horizon.svg`, `horizon.csv`, and `per_scene.jsonl`.

**EXPERIMENTALLY VERIFIED video:** Compact VideoFlow, seed 17, 162,979 parameters, 3 CPU epochs, 210 optimizer steps, validation loss 0.9746 → 0.9304 → 0.9046. Test uses 42 branches and six scenes, compares frame `t` to ground truth frame `t` for `t=1..32`, and excludes the known initial frame. Scene-mean metrics:

| Video mode | MSE | PSNR dB | SSIM |
| --- | ---: | ---: | ---: |
| Codec reconstruction | 0.001379 | 28.603 | 0.847 |
| Teacher-forced generation | 0.027867 | 15.549 | 0.535 |
| Free-running generation | 0.033865 | 14.703 | 0.517 |

Teacher forcing uses the true current/past frames, not the target current future. Free generation starts from frame zero and its own outputs. PSNR is the mean of six scene PSNR values; SSIM uses an 11×11 local RGB window. The visual branch-difference error is 0.1596 mean absolute RGB intensity. Raw aligned ground-truth/reconstruction/teacher/free RGB tensors are in six NPZ files under `results/v11-video-15eda96-test/`; `metrics.json` contains per-frame MSE, PSNR and SSIM plus per-scene values. Video evaluation commit: `15eda96e23d41557a4e7421fbe3953bb9eea6514`; checkpoint and training curves: `artifacts/v11-video-815387b-s17/`. This is a small CPU baseline, not competitive video generation quality.

**IMPLEMENTED / TESTED:** independent test evaluator, frame alignment test and no-future-frame-leakage test. **BLOCKED:** larger video model and CUDA scaling. **NOT RUN:** LPIPS, calibrated video uncertainty, video-based physical-state metric, closed-loop planning from video.
