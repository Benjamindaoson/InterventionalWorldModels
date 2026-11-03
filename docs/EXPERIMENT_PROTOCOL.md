# Experiment protocol

Freeze scene IDs and branch membership before fitting. Never compare different initial physics states as paired interventions. Match model width, training records, optimizer steps, seeds and horizons. Use train for fitting, validation for stopping, test once for final reporting.

The current matched-seed action-shuffle pilot (`scripts/run_ablation.py`) measured free-running MSE 0.169207 conditioned versus 0.174239 shuffled on six test branches. This has no uncertainty estimate and is **not evidence of benefit**. Pending: action-blind and capacity-matched controls, multiple seeds, paired supervision, mass/friction/geometry OOD splits, geometry/action/temporal/re-anchoring ablations, scaling and planning. Preserve negative outcomes.
