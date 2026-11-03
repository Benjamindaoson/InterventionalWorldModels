# Continuous execution state — WorldDynamics

Branch: `feat/worlddynamics-v2-complete`, forked from clean V1.1 commit `569e6ae`. Never overwrite `artifacts/v11-*` or `results/v11-*`; all are local ignored evidence. Initial V1.1 ledger verification: 72/72 SHA-256 matches. Local runtime: Windows, Python 3.12, torch 2.15.0+cpu, MuJoCo 3.17.0; GTX 1650 4 GiB has no CUDA support in this environment. Other projects are read-only.

| Phase | Current state | Evidence / next dependency |
| --- | --- | --- |
| 0 V1.1 audit | COMPLETE | Prior V1.1 reports and ledger verified; no dirty files at branch start. |
| 1 action causality | CPU_VERIFIED | `results/v12-action-causality-complete/`; report in `docs/ACTION_CAUSALITY_REPORT.md`. Action model overresponds and has reverse-sign local x-force gradient. |
| 2 data and confirmatory protocol | COMPLETE | `artifacts/v2-3f97ef9-32x8-h64`: 72 scenes, 504 paired branches, horizon 64, frozen source `3f97ef9`, manifest SHA `8C7815F8015CE075D20D6B44F4CAECD875E6E1B3CD73E4A75C8802A0AA6730EA`; `docs/CONFIRMATORY_PROTOCOL.md` committed before data generation. |
| 3 strong baselines | IN_PROGRESS | 24 model runs, eight arms times seeds 17/29/43, training in `artifacts/v2-baselines-3f97ef9`; only IID validation touched during training. Await completion, then exploratory IID/OOD evaluation, then one frozen confirmatory evaluation. |
| 4 causal RSSM | V1.1 VERIFIED; new-data comparison pending | Future leakage regression test exists. |
| 5 video world model | V1.1 VERIFIED; new-data comparison pending | Existing 3-epoch video loss and aligned metrics preserved. |
| 6 visual / official V-JEPA | Light baseline only | Official adapter and source/license audit pending. |
| 7 joint WAM | IMPLEMENTED, test pending full training | Removed target-action leakage from inverse head; added world-only/action-only/joint train and exploratory evaluation CLIs. New V2 training pending. |
| 8 geometry | Math primitives tested | Real camera/depth/multiview experiment pending. |
| 9–10 planning | Negative V1.1 closed loop | Oracle reachability and velocity-aware model comparison pending. |
| 11 robot dynamics | NOT_RUN | Requires distinct robot simulator task and model training. |
| 12 cross-repo export | NOT_RUN | Read-only interface inspection allowed; write only in this repo. |
| 13 GPU/scaling | BLOCKED for large models | CPU-only torch wheel; no global GPU changes or paid compute. |
| 14 original method | NOT_RUN | Requires reliable action-sensitive baselines and prior-art audit. |
| 15 confirmation/paper | NOT_RUN | Must freeze before confirm set and report all outcomes. |

Resume rule: inspect `git status`, this file, latest commits and evidence paths; continue the first unfinished prerequisite. Do not infer EXPERIMENTALLY_VERIFIED from code/tests. Existing negative and failed experiments remain evidence.
