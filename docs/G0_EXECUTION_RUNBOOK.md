# G0 execution runbook — Windows developer and AutoDL GPU

This runbook is **instructions for the human-owned machines**. It has not been executed on Windows or AutoDL by the GitHub documentation contributor.

## A. Windows PowerShell: locate existing clone before checkout
Do not assume `D:\robotics\InterventionalWorldModels` exists. The user's earlier clone was run from a C: PowerShell home directory, and the attempted D: `cd` failed. The actual current clone location has not been verified.

```powershell
$target = 'D:\robotics\InterventionalWorldModels'
$homeClone = Join-Path $HOME 'InterventionalWorldModels'
Write-Host "PWD = $(Get-Location)"
Write-Host "D Git = $(Test-Path (Join-Path $target '.git'))"
Write-Host "Home Git = $(Test-Path (Join-Path $homeClone '.git'))"
if (Test-Path (Join-Path $target '.git')) {
    $repo = $target
} elseif (Test-Path (Join-Path $homeClone '.git')) {
    $repo = $homeClone
} else {
    throw 'No existing clone found at D:\robotics or PowerShell home. Inspect before cloning or moving anything.'
}
git -C $repo status --short --branch
git -C $repo remote -v
```

Confirm `origin` is exactly `https://github.com/Benjamindaoson/InterventionalWorldModels.git`. If the working tree is not clean, stop and inspect modifications; do not reset or overwrite. If the found clone is under C: and the D: destination is empty, move it to D: only after choosing the intended path and confirming no uncommitted work will be lost. Ensure `D:\robotics` exists. Do not clone again on top of an existing directory.

## B. Fetch and use the remote development branch
Run after determining `$repo` from the preceding block and checking a clean worktree. This branch was created on GitHub; local branch state is unknown.

```powershell
git -C $repo fetch origin
git -C $repo branch -a
git -C $repo status --short --branch
# If local feat/world-model-v2 does not exist:
git -C $repo switch --track origin/feat/world-model-v2
# If a local branch already exists, inspect divergence before switching:
# git -C $repo switch feat/world-model-v2
git -C $repo log -1 --oneline
```

If `switch --track` reports an existing local branch, do not force it; inspect local history and working tree. Open this folder in Cursor/Codex only after confirming the correct path.

## C. CPU readiness in Windows or WSL
In the repository root:
```powershell
py -3.11 --version
git status --short --branch
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test]'
.\.venv\Scripts\python.exe -m pytest -q
```

If using WSL/Linux, use a separate local venv, rather than the Windows venv. CPU test success proves only legacy software contracts, not LeWM reproduction. Do not add heavy GPU deps to `pyproject.toml` before checking the official training environment and lock files.

## D. Upstream faithful LeWM integration (not executed yet)
Official: https://github.com/lucas-maes/le-wm , inspected commit `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`. Read upstream installation/training/evaluation. Before importing source, review upstream MIT and transitive code/data licenses.

Required source audit: `jepa.py`, `module.py`, `train.py`, `eval.py`, `config/train/`, `config/eval/`. Use exact config/model identity and record the target data download's verified hash/size. Official entry points listed upstream include `python train.py data=pusht` and `python eval.py --config-name=pusht.yaml policy=pusht/lewm`; these are **upstream commands**, not proof they were run here.

## E. AutoDL GPU preflight — only after access and authorization
The expected paths below are proposals, **not observed directories**. First inspect actual mounted persistent storage, GPU type, free VRAM, CUDA/PyTorch, rates and instance shutdown behavior.

```bash
pwd
nvidia-smi
python --version
python - <<'PY'
try:
 import torch
 print('torch', torch.__version__, 'cuda', torch.cuda.is_available())
 if torch.cuda.is_available(): print('gpu', torch.cuda.get_device_name(0))
except Exception as exc:
 print('torch unavailable:', repr(exc))
PY
df -h
git remote -v
git rev-parse HEAD
```

Target code `/root/autodl-tmp/InterventionalWorldModels`, data `/root/autodl-tmp/IWM-data`, and runs `/root/autodl-tmp/IWM-runs` only after verifying. Do not launch GPU training or start an instance using these lines without explicit budget and resource approval.

## F. G0 run acceptance record
Create `reports/G0/<run-id>/` with small, license-safe:
- `environment.json` (OS, Python, torch/CUDA, GPU, VRAM, package lock, upstream and local SHAs).
- `dataset_manifest.json` (origin, revision, data/file hashes, episode split policy).
- `training_manifest.json` (actual launch command, seed, batch, steps, model/checkpoint hashes, runtime).
- `evaluation.json` (independent non-mock eval, exact denominator, metrics and caveats).
- `g0_status.md` (PRECHECK/CPU_TEST/TRAINED/EVALUATED/VERIFIED/FAILED/BLOCKED).
Keep checkpoints/raw datasets/private media outside Git. Cite their approved and reproducible storage locations only.

## G. Stop points
- Current clone path not found: STOP and ask for the actual machine path.
- Uncommitted work or divergent branch: STOP and reconcile without destructive operations.
- Unreviewed upstream license or missing accessible datasets: BLOCKED, not training complete.
- Unknown GPU rate, persistence or resource budget: STOP before chargeable training.
- Missing checkpoint or independent evaluation: G0 is NOT PASSED.
