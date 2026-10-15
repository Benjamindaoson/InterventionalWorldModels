## ADDED Requirements

### Requirement: Fixed upstream identity
G0 预检 MUST 核对上游 Git checkout 的 HEAD 等于契约中的 40 位 SHA，且仓库 URL 指向官方 `lucas-maes/le-wm`。它 MUST 不修改上游 checkout。

#### Scenario: Wrong upstream revision
- **WHEN** 上游 HEAD 与固定 SHA 不同
- **THEN** 预检返回阻塞状态和非零退出码，并报告预期和实际 SHA

### Requirement: Configuration and data readiness
G0 预检 MUST 检查 Push-T 训练配置、训练入口所需字段和所引用的数据路径。文件存在本身 MUST 不被表述为成功解码或完整复现。

#### Scenario: Public archive differs from configured format
- **WHEN** 配置引用 `.lance` 而数据根目录只有 `.h5.zst`
- **THEN** 预检列出格式不匹配并阻止训练就绪结论

#### Scenario: Missing composed training config field
- **WHEN** 训练入口引用 `cfg.wandb`，且直接配置和 Hydra launcher 注入的全局配置均没有 `wandb` 段
- **THEN** 预检列为阻塞，并要求记录后续修补差异

### Requirement: Resource and budget gate
AutoDL 模式 MUST 确认 GPU 探测、可用空间和显式给定的每小时价格及最长 GPU 小时，计算费用上限，但 MUST 不启动训练、购买资源或清理数据。

#### Scenario: Missing GPU or budget
- **WHEN** GPU 探测失败或预算输入缺失
- **THEN** 预检返回非零、报告缺失项，不生成训练成功宣称

### Requirement: Machine-readable receipt
预检 MUST 输出 JSON，包含检查时间、上游和数据路径、固定身份、每项检查状态、阻塞原因与整体状态；不得输出凭证值。

#### Scenario: Blocked preflight
- **WHEN** 任一强制条件未满足
- **THEN** JSON 状态为 `BLOCKED`，退出码非零，且不生成 Checkpoint 或指标
