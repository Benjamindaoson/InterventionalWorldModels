# V1.1 implementation and acceptance status

| Gate | Status | Evidence |
| --- | --- | --- |
| L1 training/checkpoint/independent test/prediction/reproduction | EXPERIMENTALLY VERIFIED for compact CPU baselines | MuJoCo 224-record dataset, 15 state-model runs, 1 video run, checkpoint files, frozen test JSON, generation outputs and 47 passing tests. |
| L2 multi-seed strong controls, OOD physics, long horizon | EXPERIMENTALLY VERIFIED as an exploratory pilot | Five arms × three seeds, matching sample/parameter/step budget, six held-out-physics scenes, 1–32-step errors and scene bootstrap. Strong action-blind negative control wins. |
| L2 reliable downstream planning improvement | NOT PASSED | Independent six-scene closed-loop MuJoCo pilot shows learned MPC/CEM below oracle and model exploitation. |
| L3 novel scientific method | NOT RUN | No novelty or official closest-paper comparison is claimed. |

**IMPLEMENTED / TESTED:** causal RSSM fix, robust paired data/split checks, scene-level evaluator, video train/validation/test modes, model-only MPC/CEM action proposals. **BLOCKED:** large GPU scaling with the installed CPU-only torch build; any RTX 3090/4090 campaign requires a verified Linux CUDA environment, exact wheel lock and preflight before use. No driver, global Python, other repository or paid cloud resource was changed. **NOT RUN:** confirmatory external dataset, equal-FLOP comparison, official V-JEPA adapter, real robot.

Historical V1 results remain in place and are explicitly superseded where the audit found leakage or metric ambiguity. Next research dependency: construct a task with stronger action effects and a velocity-aware observation, then preregister a fresh independent test set before comparing models again. Do not promote L2 to passed based on this exploratory pilot.
