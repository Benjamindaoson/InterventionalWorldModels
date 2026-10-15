> **V2 upgrade status (2026-10-15): PLANNING / REMOTE AUDIT ONLY.** A development branch has been created to prepare a LeWorldModel-based, action-conditioned visuo-tactile world-model upgrade. **No v2 model training, tactile experiments, video generation, or GPU evaluation has been performed.** The Windows working directory and AutoDL connection still require verification.
>
> Read [V2 baseline audit](docs/V2_BASELINE_AUDIT.md), [research contract](docs/V2_RESEARCH_CONTRACT.md), [gated implementation plan](docs/V2_IMPLEMENTATION_PLAN.md) and [G0 Windows/AutoDL runbook](docs/G0_EXECUTION_RUNBOOK.md). The legacy v1 research and negative result remain below unchanged.

# Interventional World Models

Research code for **action-conditioned visual world models from sparse physical interventions**.

**Status:** Early-stage research. Existing simplified RGB threshold experiments did **not** establish that paired-branch supervision beats independent supervision. No claim of CVPR-level novelty, real physics validation, or published-model reproduction is made.

## Research question

Can a model infer the *differences between futures produced by alternative actions from the same physical state*, especially near contact or mode transitions, with fewer interventions and reliable uncertainty?

## Initial contribution to this repository

- A mathematically auditable threshold-intervention baseline (not a novel algorithm).
- Explicit abstention on actions for which observations do not determine the outcome.
- Tests including randomized finite-case verification and an indistinguishable-world counterexample.
- A research audit outlining how to compare against **Video Prediction Policy** (ICML 2025) and **OpenWAM** (October 2026) without misrepresenting their methods.

## Local setup

```bash
python -m pip install -e ".[test]"
pytest -q
python -m iwm.thresholds
```

## Research gates

1. Verify same-state physical replay and event labels using MuJoCo, then independently PhysX.
2. Validate the exact OpenWAM/FDM evaluation protocol and checkpoints before comparing scores.
3. Train equal-data, equal-capacity, equal-compute independent and paired conditional predictors.
4. Report shuffled-pair, target-only, auxiliary event loss, and capacity-matched ablations.
5. Freeze final held-out scenes, use independent seeds, confidence intervals, calibration and closed-loop outcomes.

## Prior research

- VPP (ICML 2025): https://proceedings.mlr.press/v267/hu25g.html
- OpenWAM: https://github.com/OpenWAM/OpenWAM

All research claims should identify the experiment, dataset, checkpoint, seed and statistical uncertainty that support them.
