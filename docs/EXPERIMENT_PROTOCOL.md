# Experiment protocol

Freeze scene IDs and branch membership before fitting. Never compare different initial physics states as paired interventions. Match model width, training records, optimizer steps, seeds and horizons. Use train for fitting, validation for stopping, test once for final reporting.

The matched-step four-arm pilot (`scripts/run_ablation.py`) measured free-running MSE 0.147516 conditioned, 0.144570 shuffled, 0.140938 action-blind, and 0.135443 with paired difference loss on six test branches. All arms use the same 32-wide model, seed, 24 training branches and 20 optimizer updates. This has no uncertainty estimate and is **not evidence of benefit**: blind prediction is competitive. Pending: independent seeds, mass/friction/geometry OOD splits, geometry/action/temporal/re-anchoring ablations, scaling and planning. Preserve negative outcomes.
