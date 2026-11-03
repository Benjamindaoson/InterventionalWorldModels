# V2 compact video-flow baseline

Status: **IMPLEMENTED and EXPERIMENTALLY VERIFIED as a single-seed, low-resolution baseline**. A reliable action-conditioned video improvement is **not** established.

The compact `VideoFlow` model uses a deterministic frame autoencoder, a causal temporal transformer, action embeddings, latent flow-matching loss, Euler sampling, and autoregressive rollout. This codec is **not a variational VAE** and the system is not a large Video DiT. Action-conditioned and action-blind controls have the same 162,979-parameter architecture, MuJoCo RGB data, seed 17, batch 2, 224 AdamW steps, and two epochs. Training used only 32 train scenes and eight IID validation scenes; each `artifacts/v2-video-*-0d722d9/` run has `last.pt`, `best.pt`, optimizer/scheduler state, raw epoch metrics, source revision and dataset-manifest hash. Best validation flow+reconstruction loss was 0.9182 for conditioned and 0.5618 for blind. That objective does not directly rank perceptual prediction quality.

Independent evaluation uses eight scenes and 56 branches in each of IID test, physics OOD, and geometry OOD; 64 aligned predicted frames after an observed context frame. Reconstruction decodes the real frame latent, teacher forcing uses real history through each current step, and free generation uses only the first frame, proposed actions, and its own generated frames. All displayed PSNRs average scene-level PSNR, with each scene PSNR computed from [0,1] RGB MSE over predicted frames 1–64; the context frame is excluded. SSIM uses the implementation's 11×11 local averaging. Full per-frame MSE/PSNR/SSIM and ground-truth/prediction arrays are in `results/v2-video-*-paired-1fab82c/`.

For **paired-noise evaluation**, the generator reuses the same sampled noise for every exact-snapshot action branch at each time step. This removes branch-specific sampling noise from counterfactual comparisons. It does not remove general sampling uncertainty; this is one noise seed and two Euler steps.

| IID test metric | Conditioned | Blind |
| --- | ---: | ---: |
| Reconstruction MSE | 0.001265 | 0.001523 |
| Reconstruction PSNR (dB) | 28.98 | 28.17 |
| Teacher-forced MSE | 0.057568 | 0.044408 |
| Teacher-forced PSNR (dB) | 12.40 | 13.53 |
| Teacher-forced SSIM | 0.4133 | 0.5042 |
| Free-running MSE | 0.058066 | 0.062765 |
| Free-running PSNR (dB) | 12.36 | 12.02 |
| Free-running SSIM | 0.4084 | 0.4533 |
| Same-scene visual branch-difference absolute error | 0.033047 | 0.000262 |
| Predicted branch-difference absolute magnitude | 0.032947 | 0 |
| Real branch-difference absolute magnitude | 0.000262 | 0.000262 |

The action-conditioned free PSNR is higher under paired noise while its free SSIM is lower and its visual action effect is over 100 times larger than the real mean branch pixel effect. The blind model's identical paired branches have zero predicted difference. Therefore, PSNR alone would give a misleading action-fidelity conclusion. Physics and geometry OOD results are close to IID: conditioned paired-noise free PSNR 12.36 dB and blind 12.02 dB in both families, with the same branch overresponse. The earlier **uncoupled-noise** results are preserved in `results/v2-video-*-0d722d9/`; they gave conditioned free PSNR 11.78 dB and blind 12.09 dB. Their branch-difference metric is confounded by independent sample noise and should not be used as action-fidelity evidence.

The aligned 65-frame examples `results/v2-video-conditioned-paired-iid-scene0-branch1.gif` and `results/v2-video-blind-paired-iid-scene0-branch1.gif` show ground truth, reconstruction, teacher-forced prediction, and free generation side by side. Each source `.npz` stores all seven raw branches. LPIPS and FVD are **NOT_AVAILABLE** because no locked implementation and adequate sample protocol were run. Physical state error from video is **NOT_AVAILABLE** because this model has no state estimator. These weak 32×32 pixels cannot establish object-level physical consistency.
