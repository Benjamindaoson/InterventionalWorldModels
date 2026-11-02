# G0 -> W1 MuJoCo execution runbook (2026-11-03)

## Evidence status

- G0 toy contract: 16 locally executed Python unit tests passed in a CPU-only package.
- Optional MuJoCo adapter: **UNEXECUTED** in the current CPU package because MuJoCo is absent.
- This runbook does **not** imply G0 real-engine success, W1 dataset acceptance, neural training, or world-model improvement.

## Linux host quickstart

From the PR branch, with Python 3.11 or 3.12 and an isolated environment:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pip install 'numpy>=1.26' 'mujoco>=3.0'
pytest -q
python scripts/run_mujoco_paired_smoke.py \
  --revision "$(git rev-parse HEAD)" \
  --friction 0.5 \
  --horizon 80 \
  --seed 17 \
  --output artifacts/g0_mujoco_smoke.json
```

The script refuses to overwrite existing evidence. Save the simulator's output exactly as produced; do not edit the JSON for a favorable result.

## G0 physical-engine acceptance

- All three force branches (zero, positive, negative) start from exactly the same full-physics state digest.
- The engine and Python package versions are recorded; each output is finite and has an observed event result.
- Repeated seed + physics parameters reproduce within a declared numerical tolerance, to be independently verified.
- Bad/unknown reset state, source revision, or corrupted output blocks the run.

## W1 remains blocked until

- Store **full** per-step poses, velocity/contacts and action traces, not just this G0 final-state smoke.
- Freeze the full MuJoCo XML/model, solver settings, masses, camera and friction configuration hashes.
- Generate multiple states, actions, seeds, physical parameters, and contact regimes.
- Compare exact initial physical states within pairs; separate task/scene groups across train/val/test.
- Independently audit action conditioning, contact labels, zero/inverse controls, and all physical failures.
- Preserve raw GPU/CPU run outputs, tool versions, a failed-runs ledger and end-to-end reproduction commands.

After W1 acceptance, begin W2 with independent action-conditioned predictive baselines. No method improvement claims before a matched, held-out evaluation.
