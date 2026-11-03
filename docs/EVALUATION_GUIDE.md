# Evaluation guide

`scripts/evaluate.py` loads a state checkpoint and independently evaluates the validated frozen test split: one-step MSE/MAE, teacher-forced MSE, free-running MSE and per-horizon drift. Unavailable contact, video and planning metrics have `NOT_AVAILABLE` reasons. `scripts/generate_video.py` exports generated and target RGB frames as NPZ and prints sample MSE/PSNR; this one sample is not an aggregate benchmark.

scripts/evaluate_action_selection.py loads a state checkpoint, ranks the three recorded candidate actions for each held-out scene and computes realized x-position regret from actual MuJoCo branch outcomes. This is an offline fixed-candidate check, not closed-loop planning or a success-rate benchmark.
