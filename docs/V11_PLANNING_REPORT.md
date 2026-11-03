# V1.1 closed-loop MuJoCo planning pilot

**IMPLEMENTED / TESTED:** Eight receding control cycles of four MuJoCo steps each, with a 16-step planning horizon and a 2 cm XY goal radius. Six **new** scenes use masses/frictions outside the training/test combinations. Each policy starts from the same initial full physics state per scene. Random selects a fixed discrete action; the physics oracle restores a simulator snapshot and simulates candidate futures; learned MPC evaluates five actions using only the current cube-center observation and model predictions; learned CEM uses two rounds of 12 continuous candidates, scored only by the model. Actual physics is accessed by learned policies only after an action is chosen and executed. Per-action records, predicted/actual short-step distance error, latency and exploitation failures are in `results/v11-planning-815387b-s17.json`, `s29.json`, `s43.json`.

**EXPERIMENTALLY VERIFIED:** On six scenes per model seed, oracle succeeded 4/6 and random 0/6. Learned conditioned-model MPC succeeded 0/6, 1/6 and 1/6 for seeds 17/29/43; CEM succeeded 0/6, 1/6 and 0/6. These reuse the same six scenes across model seeds, so they are not 18 independent scenes. Mean final goal distances (m): oracle 0.0161, random 0.0472; MPC 0.0571/0.0454/0.0453; CEM 0.0403/0.0349/0.0404. MPC regret versus oracle is 0.0410/0.0293/0.0293 m; CEM regret 0.0242/0.0189/0.0243 m. MPC exploitation failures total 33/32/18 of 48 control decisions; CEM 27/12/30. A failure means the model forecast a shorter distance after the chosen action while the actual four-step transition increased distance. Latency and prediction error for each policy/seed are in the raw JSON because concurrent local load affects timing.

This is a negative planning result: learned planning did not match the physics oracle and often exploited model errors. The model observes position but not velocity, a material limitation. The oracle has simulator access unavailable to the learned policies, so its score is a reference ceiling rather than a matched-information baseline.

| Model seed | MPC latency ms / prediction distance error m | CEM latency ms / prediction distance error m |
| --- | ---: | ---: |
| 17 | 2.790 / 0.00826 | 5.722 / 0.00764 |
| 29 | 2.523 / 0.01039 | 5.254 / 0.00891 |
| 43 | 2.991 / 0.00867 | 5.751 / 0.01081 |

Latency is a local CPU mean per control decision, measured while other local work could run; it is not a hardware-independent throughput benchmark.

**BLOCKED:** robust planning utility claim until velocity-aware state modeling, more independent scenes and independent replication. **NOT RUN:** real-robot control or video-conditioned closed-loop planning.
