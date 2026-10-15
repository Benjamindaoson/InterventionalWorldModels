# v2.0 改造前基线审计

审计日期：2026-10-15。仓库：`D:\robotics\InterventionalWorldModels`；改造前提交 `c49cb7ca215762adaa1d30b381a58c2f8be7fecd`。本文件区分源码、测试和实际实验；后续结果应追加独立收据，不覆盖历史结论。

## 已有实现与证据

| 部分 | 源码事实 | 当前证据与限制 | v2 去向 |
| --- | --- | --- | --- |
| `src/iwm/thresholds.py` | 单调一维阈值、主动二分、区间预测与弃权 | `tests/test_thresholds.py` 检查有限样例；不代表接触操作世界模型 | 保留为 legacy 理论基线 |
| `src/iwm/graph_theory.py` | 基于图拉普拉斯的确定性误差证书 | `tests/test_core.py` 有随机数值断言；它不是神经模型训练定理 | 保留为分析工具 |
| `src/iwm/experiment_protocol.py` | 实验元数据 SHA-256 指纹与场景 ID 重叠检查 | 有局部测试；没有数据文件哈希、分组划分或训练收据 | 扩展而不破坏旧接口 |
| `src/iwm/learning.py` | 小型 RGB CNN，动作拼接，事件/位移损失，成对差分可选项，单轮训练与评价函数 | `tests/test_learning.py` 只验证形状、反向传播与掩码；无项目内真实训练记录、Checkpoint 或独立测试结果 | 原文件保留为 legacy baseline；不得称为 LeWM |
| `src/iwm/mujoco_data.py` | 可选 MuJoCo 单物体力/摩擦力学烟测，输出 JSONL | 无集成测试、RGB/Depth、机器人动作、同状态分支快照或重放证据；旧文档明确写未运行 | 后续 G2 再扩展，不作为 G0 数据 |
| `tests/` 与 `.github/workflows/tests.yml` | 纯 Python 测试、NumPy 测试、可选 PyTorch 测试 | CI 原 `test` extras 未声明 NumPy，存在依赖缺口；通过测试不等于实验通过 | 补依赖并持续验证 |
| `docs/`、`README.md` | 理论、路线图、先前审计和研究提案 | `docs/research_audit.md`、`docs/implementation_status.md` 保留早期负结果和证据边界 | 保留历史，README 改为中文的实时状态 |

源码运行关系图：[当前架构 HTML](architecture/v1-runtime.html)，其 JSON 来源为 [v1-runtime.json](architecture/v1-runtime.json)。图基于上述改造前提交；三条运行路径目前没有形成统一数据→训练→评测流水线。

## 必须保留的负结果

旧文档记录：仓库外的简化 RGB pilot 中，直接配对监督未显示稳定优于独立监督。没有 MuJoCo、Push-T 或闭环证据支持优势宣称。后续 G2 只有在数据、容量、训练预算、划分和独立测试均匹配时才可以重新检验该假设。

## 与 v2 目标的差距

- 当前模型只接收单帧 RGB 与候选动作，无 Depth、时间戳、机器人本体状态、触觉、潜空间多步预测或视频生成。
- 当前 MuJoCo 脚本不输出用于训练的视觉轨迹，也不支持完整物理状态复制与反事实重放。
- 当前评价函数只给事件准确率、Brier 和近零位移子集准确率；无独立模型加载、规划成功率、置信区间或 OOD 评价。
- 当前实验指纹只覆盖用户给定字段，未绑定代码、数据文件、上游依赖、Checkpoint 和机器环境。
- 未见项目内真实神经网络训练产物；已有 `docs/IMPLEMENTATION_ROADMAP.md` 将端到端 CPU 实验、多种基线和闭环验证列为未完成。

## G0 上游静态预检

官方仓库：[lucas-maes/le-wm](https://github.com/lucas-maes/le-wm)，审计固定提交 `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`。论文：[arXiv:2603.19312](https://arxiv.org/abs/2603.19312)。上游 MIT License，Copyright (c) 2026 Lucas Maes。没有复制其源码进入本仓库；后续若复制，必须带许可证与修改记录。

已核对 `jepa.py`、`module.py`、`train.py`、`eval.py` 和 Push-T 配置：Encoder 为 ViT，Action Encoder 为 `Embedder`，Predictor 为 `ARPredictor`，`SIGReg` 在 `module.py`，训练损失为下一潜变量 MSE 加加权 SIGReg。论文架构和当前源码配置不能混作同一版本的实测参数。

当前提交存在待解决的数据分发差异：`config/train/data/pusht.yaml` 使用 `pusht_expert_train.lance`，而 README 描述 `.h5` 路径；官方 [Push-T 数据仓库](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/main) 的 `main` 提交 `655cd446b9929369d7d406001da85c15d1457850` 显示 `pusht_expert_train.h5.zst`（页面标称 13.1 GB）。[stable-worldmodel 上游](https://github.com/galilai-group/stable-worldmodel) 提供格式转换接口，但此数据集的具体转换命令、字段一致性与训练加载仍未实测。`train.py` 所需 `cfg.wandb.enabled` 由 `config/train/lewm.yaml` 的 Hydra `launcher: local` 默认项加载 `config/train/launcher/local.yaml` 并注入全局配置，静态核对为 `False`；这不是配置缺口。未证明默认命令可直接完成训练；必须先完成真实数据加载和短时训练验证。任何修补必须单独记录与官方默认的差异。

## 本次审计命令

`git status --short --branch`、`git remote -v`、`git log -1`、`codegraph init --yes`、`codegraph status`、`codegraph explore`、`rg --files`、`git ls-remote`；上游放在系统临时目录进行只读审计，未建立第二个权威项目。
