# V2 joint world–action visual baseline

Status: **IMPLEMENTED and EXPERIMENTALLY VERIFIED as a single-seed exploratory baseline**. No joint-training benefit was observed.

Three `JointWorldAction` models used identical V2 MuJoCo RGB train/IID-validation branches, seed 17, batch 2, two epochs, 224 optimizer steps and AdamW. `world` optimizes visual-token prediction and frame reconstruction; `action` optimizes inverse action from adjacent frames and reconstruction; `joint` optimizes all three. The inverse action head receives current and future images but never the target action. The world head receives current images and proposed action but never a future image. Unit tests verify the information boundary and unused-head gradients. The architecture has 159,878 total parameters; active-gradient counts are 159,683 for world, 157,670 for action, and 159,878 for joint, so active capacity is close but not identical.

The first unbounded-latent world-only run `artifacts/v2-wam-world-ade0d9e/` had validation loss 19.63 and is preserved. A bounded `tanh` visual latent was then introduced to stabilize the moving representation target. The following matched runs are from commit `4ccb070`: `artifacts/v2-wam-world-4ccb070/`, `artifacts/v2-wam-action-4ccb070/`, and `artifacts/v2-wam-joint-4ccb070/`. Each contains the last/best checkpoint, optimizer and scheduler state, epoch losses, run manifest and source revision. The dataset manifest SHA-256 is `8c7815f8015ce075d20d6b44f4caecd875e6e1b3cd73e4a75c8802a0aa6730ea`.

| IID test metric | World only | Action only | Joint | Zero-action reference |
| --- | ---: | ---: | ---: | ---: |
| Decoded next-frame MSE, [0,1] RGB | 0.000334 | NOT_AVAILABLE | 0.002254 | NOT_AVAILABLE |
| Same-scene branch-difference MSE, [0,1] RGB | 0.00000749 | NOT_AVAILABLE | 0.00000745 | NOT_AVAILABLE |
| Inverse action MAE, newtons | NOT_AVAILABLE | 2.4301 | 2.4770 | 2.3810 |

The physics OOD and geometry OOD numbers are stored per scene in `results/v2-wam-*-0d722d9.json`; the pattern covers nine complete files. The world-only decoded MSE is about 0.000334 on both OOD families, the joint MSE about 0.002254, and action MAE remains around 2.43 N or 2.477 N. This is one-step **teacher-forced** visual prediction: every current frame is real. Free-running WAM video generation was not evaluated, and these numbers cannot establish long-horizon visual dynamics. Pixel changes between counterfactual branches average only 0.000262 in absolute [0,1] RGB units on IID, so background-dominated frame MSE is weak evidence for physical action response. Learned token coordinates differ across independently trained encoders; cross-run latent MSE is not a fair comparison. The three-way experiment is a software and baseline validation, not a multi-seed research conclusion.
