## Context

现有 `src/iwm/learning.py` 是简单 RGB 分支预测器，缺少 LeWM 训练与模型工件。上游 `le-wm` 固定为 `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`；当前配置使用 `.lance`，官方公开 Push-T 数据为 `.h5.zst`，需要验证转换与加载。`train.py` 的 `cfg.wandb.enabled` 由 Hydra `launcher/local.yaml` 注入，静态核对已通过。AutoDL SSH 目前拒绝连接。

## Goals / Non-Goals

**Goals:**
- 保留历史结果；把已实现、已执行和已验证分开。
- 提供无需 GPU、不会启动训练的可测试预检，输出结构化 JSON 和非零阻塞退出码。
- 留下官方来源、固定 SHA、数据身份和实机训练所需的下一步。

**Non-Goals:**
- 在 Windows 上替代 AutoDL 完成官方训练。
- 用下载的权重或合成结果代替训练日志和 Checkpoint。
- 此阶段直接开发 G1–G6 的空模块。

## Decisions

1. **上游不复制入项目源码。** 临时只读审计缓存与未来云端固定 SHA checkout 保持上游权属；本仓库只维护预检与实验配置差异记录，避免双权威实现和许可证遗漏。
2. **预检仅依赖标准库。** `python -m iwm.g0_preflight` 接受上游 checkout、数据根目录及模式，读取 Git SHA、必要源码/配置、数据文件和环境；不会下载、写入云路径、运行训练或修改上游。
3. **阻塞信息是输出的一部分。** 配置/数据不匹配、缺失 GPU 或未输入预算必须显式列为 blockers。可选 `--output` 仅写一份本地报告，退出状态反映能否继续。
4. **研究报告按工件证据升级状态。** G0 的工程预检可以完成，正式训练状态仍为被阻塞；未来每次训练需独立 receipt。

## Risks / Trade-offs

- 上游包和数据分发可能继续漂移 → 锁定 SHA 与 Hugging Face revision，预检在执行时重新验文件哈希。
- 单纯文件存在不能证明数据可解码 → 正式训练前需在 AutoDL 运行真实加载检查和短时预算试跑。
- `nvidia-smi` 能看到 GPU 不代表 PyTorch 可用 → 云端另外核对 CUDA/PyTorch 并进行小批次反向传播。
- 费用无法从论文“几小时”推出 → 不把论文时长当成本估计；用当前实例价格和计时试跑计算。

## Migration Plan

旧代码与文档不删除。G0 上游运行另用固定 checkout 和独立环境；遇到失败保留日志与配置差异。回滚本变更只需恢复本仓库新增文件和 README，不影响历史源码或云端数据。

## Open Questions

- 哪个实际可连接的 AutoDL 实例和价格可用？
- 对官方当前 `.lance` 训练配置，应采用官方支持的转换流程还是固定较早的 HDF5 版本？须通过实际加载验证。
- 云端建议路径的持久化语义是什么？
