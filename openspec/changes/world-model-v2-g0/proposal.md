## Why

当前仓库只有研究原型和局部测试，缺乏可审计的世界模型训练入口与复现证据。G0 首先需要把官方 LeWM Push-T 基线锁定、核实训练前置条件，并将真实训练与本地工程准备严格区分。

## What Changes

- 保存 v1 代码审计、v2 研究契约与 G0–G6 顺序计划，保留旧负结果。
- 固定上游 LeWM 代码和官方数据来源，记录作者与 MIT License，创建只读 G0 预检工具及测试。
- 修正仓库本地测试环境和中文 README 的实际状态说明。
- 记录 AutoDL 连接、配置与数据格式阻塞；GPU 训练、Checkpoint 与评测只有实际运行后才能标记完成。

## Capabilities

### New Capabilities

- `g0-preflight`: 对官方基线的上游身份、配置、数据路径、计算环境与预算做失败封闭的只读预检。
- `research-evidence`: 对各 Gate 的状态与实验工件建立可核查的证据契约。

### Modified Capabilities

无。旧基线接口保持原样。

## Impact

新增 `src/iwm/g0_preflight.py`、对应测试、`docs/V2_*`、G0 状态报告和 README 更新；测试 extras 明确声明 NumPy。本变更不复制上游源码、不修改无关仓库、不发起收费训练。
