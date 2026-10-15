# 面向机器人接触操作的动作条件视觉—触觉世界模型

**Action-Conditioned Visuo-Tactile World Models for Contact-Rich Robotic Manipulation**

本项目研究机器人在接触、遮挡和物理参数变化时，能否根据当下观测与候选动作，预测未来潜在状态、物体运动和接触事件，并利用预测帮助选择动作。目标路径是 **表征学习 → 动作条件潜动态 → 视觉触觉预测 → 生成式未来建模 → 泛化与规划**。这是一项按实验门槛推进的研究工程；下表是当前证据状态，而非计划目标。

## 当前状态（2026-10-15）

| 阶段 | 当前状态 | 可核查事实 |
| --- | --- | --- |
| v1 物理干预原型 | 已实现；局部测试已验证 | 单调阈值、图误差证书、实验指纹与小型 RGB 分支模型；旧文档明确记录配对监督没有稳定收益 |
| G0 官方 LeWorldModel Push-T | **训练被阻塞；本地预检已执行** | 上游代码/配置已审计；无真实训练日志、参数更新证据、Checkpoint 或独立评测 |
| G1 表征学习 | 尚未开始 | 无本项目训练/Probe 实测 |
| G2 动作潜动态 | 尚未开始 | 现有 MuJoCo 文件只是物理烟测生成器，缺少训练轨迹与同状态重放 |
| G3 视觉触觉 | 尚未开始 | 无触觉数据或训练结果 |
| G4 生成式未来 | 尚未开始 | 无生成式训练或模型 Checkpoint |
| G5 泛化与规划 | 尚未开始 | 无独立闭环评测 |
| G6 统一研究报告 | 尚未开始 | 只在各 Gate 有真实证据后汇总 |

G0 的阻塞具体是：当前官方训练配置引用 `pusht_expert_train.lance`，已发现同名的[公开 Lance 候选数据](docs/G0_LANCE_SOURCE_RESEARCH_2026-10-15.md)，但尚未用固定 Loader 实读，也未证实与原 HDF5 内容等价。`cfg.wandb.enabled` 由 Hydra 的 `launcher/local.yaml` 注入，静态配置核对已通过。现有 `autodl-p1` SSH 端点拒绝连接，GPU/磁盘/价格尚未核实。详见 [基线审计](docs/V2_BASELINE_AUDIT.md)、[G0 预检收据](reports/G0_PRECHECK_2026-10-15.md)和[数据及连接核验回执](reports/G0/DATA_ACCESS_AUDIT_2026-10-15.md)。

## 架构与代码归属

目标架构：`RGB/Depth + 本体状态 + 时间 + 动作 (+ 可选触觉) → Encoder → Latent State → Action-conditioned Predictor → 未来潜状态/物理状态/事件 (+ 可选未来观测)`。物理真值只用于监督和独立评分，不进入学习模型的测试时输入。当前源码尚未实现这一完整架构。[改造前运行架构图](docs/architecture/v1-runtime.html) 展示现有分离路径。

`src/iwm/thresholds.py`、`graph_theory.py`、`learning.py` 和 `mujoco_data.py` 是本仓库的早期研究原型，将作为 legacy 对照保留；`src/iwm/g0_preflight.py` 是本仓库新增的只读工程检查。LeWorldModel 的模型、SIGReg 与训练入口属于 [Maes 等作者的官方仓库](https://github.com/lucas-maes/le-wm)，以提交 `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` 为 G0 审计版本，MIT License，Copyright (c) 2026 Lucas Maes。本仓库没有复制上游模型源码，也未声称训练了它。论文：[LeWorldModel](https://arxiv.org/abs/2603.19312)。

## Windows 本地复核

使用 Python 3.10+，在本仓库独立环境中：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[test,learning]'
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m iwm.thresholds
.\.venv\Scripts\python.exe -m iwm.g0_preflight --help
```

若只有审计和不含 PyTorch 的基础测试需求，可安装 `.[test]`；PyTorch 相关测试会被明确跳过。通过这些命令只证明本地代码可运行，不证明世界模型训练复现。

G0 只读预检需要一个**已存在**的官方固定 SHA checkout 与数据目录；不会自动下载、转换或启动训练：

```powershell
.\.venv\Scripts\python.exe -m iwm.g0_preflight `
  --upstream 'C:\path\to\le-wm' `
  --data-root 'D:\path\to\IWM-data' `
  --mode local `
  --hash-data `
  --output 'reports\g0-local-preflight.json'
```

退出码 `2` 和 JSON 中的 `BLOCKED` 是预期的失败封闭行为；`PASS_STATIC_ONLY` 也仅代表静态检查通过，不代表数据可解码、GPU 可训练或研究结果有效。`--hash-data` 对实际配置的数据文件或目录逐字节计算 SHA-256，可能需要较长时间。AutoDL 模式还要求 `LOCAL_DATASET_DIR` 与 `--data-root` 一致、真实 GPU/CUDA/PyTorch 与正数 `--max-gpu-hours`、`--hourly-rate-cny`，不会调用训练入口。

## 实验门槛与局限

[研究契约](docs/V2_RESEARCH_CONTRACT.md) 规定 G0→G6 验收、来源、划分、成本与负结果保存；[实施计划](docs/V2_IMPLEMENTATION_PLAN.md) 记录下一步。正式训练需先恢复 AutoDL 连接、解决上游数据/配置差异、核实当前实例价格并给出 GPU 时间上限，经用户批准后执行。训练、评测、消融、泛化和闭环指标目前均为 `N/A`；此仓库尚无真机部署或已发表论文成果。

早期简化 RGB pilot 中，配对差分监督**未证明**优于独立监督。该负结果见 [原始研究审计](docs/research_audit.md)，不能通过重写 README 消除，也不能外推到尚未运行的 G2 实验。
