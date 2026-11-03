# V2 held-out MuJoCo planning pilot

Status: **EXPERIMENTALLY VERIFIED negative closed-loop result**, with task-feasibility limitations. No learned planner success claim is supported.

The test consists of eight new MuJoCo cube scenes that are absent from the training and state-model test families. Each controller receives current cube xyz position and linear velocity, a fixed target 0.025 m away, and eight control cycles of four physics steps. Success requires final goal distance at most 0.015 m. Random, privileged one-step greedy physics Oracle, learned-model MPC over seven discrete forces, and learned-model CEM over continuous x/y forces execute in the real simulator after action selection. The Oracle alone evaluates candidate actions in restored MuJoCo snapshots. MPC and CEM rank candidates exclusively using the current observation, proposed force, and the frozen checkpoint. Tests exercise this information boundary and exact Oracle restore.

All three seeds 17, 29, 43 were run for blind position+velocity, action-conditioned position+velocity, and RSSM position+velocity checkpoints. Per-cycle actions, initial snapshots, goals, final distances, prediction errors, planning latencies, and exploitation flags are in nine `results/v2-planning-*-9e2da49.json` files. `results/v2-planning-summary-9e2da49.json` checks exact initial-scene/goal identity across runs and aggregates by independent scene. The first attempt failed before writing output because the checkpoint revision was read from the wrong key; commit `9e2da49` fixed that code path, and all nine evaluations subsequently completed.

| Policy and model | Successes per seed / 8 | Mean success rate | Mean final distance (m) | Mean regret vs greedy Oracle (m) | Mean position prediction error (m) |
| --- | --- | ---: | ---: | ---: | ---: |
| Random | 0, 0, 0 | 0.000 | 0.02898 | 0.01756 | NOT_AVAILABLE |
| Greedy physics Oracle | 5, 5, 5 | 0.625 | 0.01142 | 0 | NOT_AVAILABLE |
| Blind velocity + MPC | 2, 0, 1 | 0.125 | 0.04060 | 0.02918 | 0.00707 |
| Conditioned velocity + MPC | 1, 0, 0 | 0.042 | 0.03743 | 0.02601 | 0.03056 |
| RSSM velocity + MPC | 0, 0, 1 | 0.042 | 0.04434 | 0.03292 | 0.07557 |
| Blind velocity + CEM | 0, 0, 0 | 0.000 | 0.02417 | 0.01275 | 0.00723 |
| Conditioned velocity + CEM | 1, 0, 0 | 0.042 | 0.03224 | 0.02082 | 0.04004 |
| RSSM velocity + CEM | 0, 0, 0 | 0.000 | 0.03882 | 0.02740 | 0.06978 |

The mean conditioned-minus-blind MPC success difference is −0.0833, with an independent-scene bootstrap 95% interval [−0.25, 0]. For CEM the difference is +0.0417 [0, 0.125], arising from a single successful seed/scene and therefore not robust evidence. Across 24 conditioned-MPC episodes the defined model-exploitation flag fired 77 times over 192 control cycles; it means predicted goal distance improved while the executed physical goal distance worsened. The corresponding blind-MPC count was 34. Runtime and per-scene values are in the raw summary.

The Greedy Oracle reaches only five of eight scenes. A separate privileged reachability audit scanned 21 x-force values from −10 to +10 N; both a constant-action 32-step plan and a denser four-step greedy Oracle reached five of eight scenes (`results/v2-planning-reachability-f1cd1ba.json`). These finite candidate tests do not prove the other three scenes globally unreachable. The reported regret is relative to the **greedy reference**, not optimal control regret. No scene was removed after observing outcomes. The task needs a new feasibility-qualified, independently frozen scene set before a strong planning conclusion can be made. Better open-loop RSSM MSE did not translate into MPC success in this pilot.
