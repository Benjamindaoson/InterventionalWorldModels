## ADDED Requirements

### Requirement: Evidence states
项目文档 MUST 按 Gate 分别记录已实现、已执行、已验证、未验证、失败、被阻塞和尚未开始。通过单元测试 MUST 不被转换成模型训练或研究效果结论。

#### Scenario: Local code passes tests without GPU run
- **WHEN** 本地预检和单元测试通过但没有真实训练日志及 Checkpoint
- **THEN** 报告只标注本地工程结果，G0 训练与评测仍标为被阻塞或未验证

### Requirement: Historical negative results
v2 文档 MUST 保留早期配对监督未稳定优于独立监督的记录；后续新结论 MUST 引用匹配实验与独立测试。

#### Scenario: Preparing README
- **WHEN** 更新项目首页
- **THEN** 首页明确旧负结果及 G0–G6 当前实测状态，不把目标架构写成已训练系统

### Requirement: Gate artifact provenance
每个完成的训练 Gate MUST 记录代码、上游和数据身份、配置与种子、GPU 日志、Checkpoint 哈希、独立评测和费用；未知项 MUST 填 `N/A` 或 `BLOCKED`。

#### Scenario: No GPU access
- **WHEN** 无法连接 AutoDL
- **THEN** 阶段报告保留缺失资源信息且 GPU 型号、费用和训练指标写 `N/A`
