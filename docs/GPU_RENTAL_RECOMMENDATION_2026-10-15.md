# G0 GPU 租用建议（2026-10-15）

**建议先租一台 Linux 单卡 RTX 4090 24 GB 实例，用于 G0 Push-T 的环境、数据和显存计时预检。** 选择至少 8 个 CPU 逻辑核、32 GB 系统内存和约 200 GB 可用且持久的数据空间；这些 CPU、内存、磁盘数字是避免数据加载与格式转换受阻的采购建议，**不是已验证的最低需求**。先按小时租用，不预购长时段或多卡资源。正式训练的小时数和费用要在真实数据加载与短时计时试跑后估算、再获批准。

## 为什么选 4090

- [LeWorldModel 论文](https://arxiv.org/html/2603.19312)报告模型约 1500 万参数、单卡数小时训练；附录给出 batch 128、4 帧、224×224 的输入。论文没有在这些陈述中指定 4090 型号或给出该卡的峰值显存，因此**不能保证**官方原配置在 24 GB 中运行，也不能把“数小时”当作我们的预算实测。
- [固定上游配置](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/lewm.yaml)使用 `batch_size: 128`、`precision: bf16`、`max_epochs: 100`、6 个数据工作进程。论文 Push-T 实验正文说训练 10 epochs；这项协议差异须在正式复现前冻结并记录。先用原配置跑真实小批次，记录 `torch.cuda.max_memory_allocated/reserved`、`nvidia-smi` 峰值、batch/s 与首个错误；若 24 GB OOM，不直接减小 batch 后称为原配置复现。
- [NVIDIA 规格](https://www.nvidia.com/en-us/geforce/graphics-cards/40-series/rtx-4090/)确认桌面 RTX 4090 有 24 GB 显存，[RTX 4080 / 4080 SUPER](https://www.nvidia.com/es-es/geforce/graphics-cards/40-series/rtx-4080-family/)为 16 GB。就官方 4 帧、batch 128 的显存风险而言，4090 比标准 4080 提供更大的余量；这是硬件容量推断，不是本项目的训练基准测试。没有必要为了 G0 先租多卡。

## 型号澄清及升级条件

| 租赁页面写法 | 如何判断 | 本项目建议 |
| --- | --- | --- |
| `RTX 4090 24G` | 标准 24 GB 卡；以实例内 `nvidia-smi` 为准 | G0 首选；G1–G3 的小模型实验可继续复用，但仍须逐项测量 |
| `RTX 4080` / `4080 SUPER 16G` | NVIDIA 标准卡为 16 GB | 可以做少量预检；不作为完整 G0 原配置的首选租赁目标 |
| `4080(S)-32G`、`4090-48G` | [AutoDL Pro API 文档](https://www.autodl.com/docs/instance_pro_api/)将其列为平台显示的特定算力规格；不能从名称直接推断为标准零售卡 | 核对实例页面和 `nvidia-smi` 的实际显存、独占性、价格和限制后再比较 |
| `L40S 48G` | [NVIDIA 官方规格](https://www.nvidia.com/en-us/data-center/graphics-cards-for-virtualization/)为 48 GB | 仅在 G0 实测 OOM 且需保留原 batch，或 G3/G4 多模态、视频生成的实测峰值接近 24 GB 时升级 |
| `L4080` | 未在上述 NVIDIA/AutoDL 官方型号资料中核实为明确型号 | 请以租赁页面完整商品名和显存为准，避免据此下单 |

未来 G4 的低分辨率生成模型、G3 的视觉触觉融合、G5 的规划评测各有不同的内存与吞吐需求。是否升级到 48 GB、是否需要多卡，要以该 Gate 的冻结模型/批量/序列长度和实测峰值为准；**没有证据支持现在一次性为 G0–G6 购买大实例**。

## 租赁前需核对

1. 供应商页面的准确 GPU 型号、显存、单价、CPU/内存、可用磁盘、停机与持久化收费规则；价格随时段和区域变化，不在此文件写未经下单页面确认的固定价。
2. 实例启动后只读检查 `nvidia-smi`、驱动/CUDA、PyTorch `torch.cuda.is_available()`、`df -h`、CPU/内存、现有进程；核实 `/root/autodl-tmp` 是否持久保存。[AutoDL GPU 选型文档](https://www.autodl.com/docs/gpu/)说明平台按 GPU 数量分配 CPU 和内存，实例 GPU 独占，并建议每卡至少 4–8 个 CPU 核心以支持数据读取。
3. 先解决官方公开数据 `.h5.zst` 与当前训练配置 `.lance` 的格式差异，验证完整样本读取，再测短批次显存和吞吐。数据准备也会占用实例时间与磁盘。

**证据状态：**本仓库尚未在 AutoDL 连通任何 GPU，未完成 G0 真实训练或显存测量。以上是租赁方案，不是训练结果。数据路径、测试门槛和成本审批流程见 [G0 运行手册](G0_LEWM_RUNBOOK.md)。
