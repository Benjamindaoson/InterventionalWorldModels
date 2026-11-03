# Continuous execution state — WorldDynamics

Branch: `feat/worlddynamics-v2-complete`, forked from clean V1.1 commit `569e6ae`. Never overwrite `artifacts/v11-*` or `results/v11-*`; all are local ignored evidence. Initial V1.1 ledger verification: 72/72 SHA-256 matches. Local runtime: Windows, Python 3.12, torch 2.15.0+cpu, MuJoCo 3.17.0; GTX 1650 4 GiB has no CUDA support in this environment. Other projects are read-only.

| Phase | Current state | Evidence / next dependency |
| --- | --- | --- |
| 0 V1.1 audit | COMPLETE | Prior V1.1 reports and ledger verified; no dirty files at branch start. |
| 1 action causality | CPU_VERIFIED | `results/v12-action-causality-complete/`; report in `docs/ACTION_CAUSALITY_REPORT.md`. Action model overresponds and has reverse-sign local x-force gradient. |
| 2 data and confirmatory protocol | IN_PROGRESS | Need distinct initial velocity, geometry, IID/physics/geometry OOD and frozen confirmatory generation rule. |
| 3 strong baselines | NOT_RUN on new data | Preserve V1.1 negative action-blind result; retrain only after Phase 2 freeze. |
| 4 causal RSSM | V1.1 VERIFIED; new-data comparison pending | Future leakage regression test exists. |
| 5 video world model | V1.1 VERIFIED; new-data comparison pending | Existing 3-epoch video loss and aligned metrics preserved. |
| 6 visual / official V-JEPA | Light baseline only | Official adapter and source/license audit pending. |
| 7 joint WAM | One-epoch smoke only | Independent/joint ablation and validation pending. |
| 8 geometry | Math primitives tested | Real camera/depth/multiview experiment pending. |
| 9–10 planning | Negative V1.1 closed loop | Oracle reachability and velocity-aware model comparison pending. |
| 11 robot dynamics | NOT_RUN | Requires distinct robot simulator task and model training. |
| 12 cross-repo export | NOT_RUN | Read-only interface inspection allowed; write only in this repo. |
| 13 GPU/scaling | BLOCKED for large models | CPU-only torch wheel; no global GPU changes or paid compute. |
| 14 original method | NOT_RUN | Requires reliable action-sensitive baselines and prior-art audit. |
| 15 confirmation/paper | NOT_RUN | Must freeze before confirm set and report all outcomes. |

Resume rule: inspect `git status`, this file, latest commits and evidence paths; continue the first unfinished prerequisite. Do not infer EXPERIMENTALLY_VERIFIED from code/tests. Existing negative and failed experiments remain evidence.
