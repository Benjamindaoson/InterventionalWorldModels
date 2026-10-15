# G0 LeWorldModel Push-T 训练运行手册

**状态：BLOCKED — PRETRAINING。** 本文中的训练命令来自[固定上游 README](https://github.com/lucas-maes/le-wm/tree/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac)；当前未运行，不能当作复现记录。来源锁定见 [`configs/lewm/g0_source_lock.json`](../configs/lewm/g0_source_lock.json)。

## 训练前只读检查

1. 确认实际 AutoDL 主机与实例正在运行；只读采集 `hostname`、`nvidia-smi`、`nvcc --version`、`python -c 'import torch; ...'`、`df -h`、`pgrep -af python`。检查已有训练/数据，避免重复进程或覆盖工件。
2. 核实 `/root/autodl-tmp/InterventionalWorldModels`、`/root/autodl-tmp/IWM-data`、`/root/autodl-tmp/IWM-runs` 是否存在、容量和持久化规则。它们目前只是建议路径。
3. 在指定云端代码目录取得本仓库的 `feat/world-model-v2` 固定提交；在独立的**上游依赖 checkout** 中取得 `lucas-maes/le-wm` 的 `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`。上游 checkout 不是第二个权威世界模型项目，不能向它提交自主研究代码。
4. 在各 checkout 分别创建 `.venv`。上游 README 的 Python 3.10 与 `stable-worldmodel[train,env]` 只是安装请求，不是依赖锁。实机安装并测试后保存完整 `pip freeze`/`uv pip freeze`、wheel 来源和 CUDA/PyTorch 版本，固定后再执行正式训练。
5. 官方数据仓库固定 revision `655cd446b9929369d7d406001da85c15d1457850`；页面显示 `pusht_expert_train.h5.zst`，压缩文件约 13.1 GB。先确认空间、下载许可、哈希和实际解压/转换流程。[stable-worldmodel](https://github.com/galilai-group/stable-worldmodel) 提供格式转换接口，但本项目尚未验证此文件的具体转换配方。不要因训练配置需要 `.lance` 就简单改名；真实加载、字段、长度、动作维度和抽样值必须检查。
   - 可优先核验固定 revision `ea321e392348e3c65a18ab0d685f00e57be2c3e0` 的 [`galilai-group/lewm-pusht`](https://huggingface.co/datasets/galilai-group/lewm-pusht)。其 `pusht_expert_train.lance/` 三个数据分片合计 14,295,080,031 bytes。来源、字段和未验证项目见 [Lance 核验](G0_LANCE_SOURCE_RESEARCH_2026-10-15.md)。它目前是候选源，不替换 HDF5 来源锁；卡片声明不等于转换等价证据。
   - 固定版 `train.py` 将配置中的 `pusht_expert_train.lance` 传给 `swm.data.load_dataset(..., cache_dir=LOCAL_DATASET_DIR, num_steps=4, frameskip=5, keys_to_load=...)`。候选目录名与公开字段静态吻合。当前本地独立 `.venv` 未安装 `stable-worldmodel`/Lance 读取依赖，也未下载数据，故 **真实 Loader 兼容性仍未验证**。先锁定依赖版本和 HF 数据 revision，在可达实例上作有上限的小样本读取；记录实际 dataset 长度、episode 数、列维度、图像 dtype/shape/codec、前后窗口索引与读取耗时。
   - 全量传输前至少计入 14.30 GB 数据分片，另核算 Lance 元数据、下载缓存、运行缓存、Checkpoint 和日志占用。核实可用空间与持久化策略后再申请全量传输许可；不能以估算行数代替实测。
6. `train.py` 的 `cfg.wandb.enabled` 由 Hydra `launcher: local` 默认项加载的 `launcher/local.yaml` 注入，静态配置为 `False`。在安装上游依赖后输出实际组合配置，核对生效值。如需修改任何上游配置，保存原始文件、最小补丁和哈希。
7. 将 `LOCAL_DATASET_DIR` 设为核实后的数据根目录，使用 `python -m iwm.g0_preflight --upstream <固定上游目录> --data-root <同一数据根目录> --mode autodl --hash-data --max-gpu-hours <上限> --hourly-rate-cny <已核实价格> --output <收据>`。哈希可能耗时。即使显示 `PASS_STATIC_ONLY`，还要做真实数据加载和一小批前后向计算。

## 独立评测与划分

固定版上游 `train.py` 在构造时间窗口数据集后调用 `spt.data.random_split(dataset, [0.9, 0.1], generator=...)`。这不是显式按 episode 划分；相邻或重叠窗口可能分落训练与验证集。具体重叠量须在锁定的依赖和真实数据上核验。**官方训练协议原样保留**，其验证指标单独标为上游随机划分；研究结论另用冻结的 episode 分组测试集。按 `episode_idx`（多来源时按 `source_revision + episode_idx`）分组，固定种子并保存组 ID/哈希，保证任一 episode 及其全部窗口只在一个集合；另记录物体/场景来源，必要时按来源整体留出。独立评测仅加载冻结 Checkpoint，在训练结束后运行，不能用测试分组调参。

## 当前 AutoDL 连接核验（2026-10-15）

Windows `ssh -G autodl-p1` 显示 `root@connect.cqa1.seetacloud.com:14946`；TCP 端口检测失败，批处理 SSH 在 banner 阶段报告 `Connection refused`。未建立会话，GPU、CUDA、磁盘、进程、持久化和价格均未核验。须从 AutoDL 控制台核实原实例运行状态、当前 SSH 主机与端口、实例价格和磁盘持久化规则；不要发送密码或私钥。取得当前入口后重试只读预检，再据小规模计时结果计算正式训练预算。详见 [G0 回执](../reports/G0/DATA_ACCESS_AUDIT_2026-10-15.md)。

## 预算与正式执行门槛

首租 GPU 型号、CPU/内存/磁盘余量和型号歧义见 [GPU 租用建议](GPU_RENTAL_RECOMMENDATION_2026-10-15.md)。正式训练协议还须核对论文 Push-T 的 10 epochs 与当前上游 `max_epochs: 100` 的差异，冻结实际采用的 epoch 数及差异说明。

先完成有明确最长时长和停止条件的计时试跑，记录 batch/s、峰值显存、数据吞吐、每 epoch 时长与日志。按 `预计 epoch 时长 × 冻结 epoch 数 × 实例价格` 计算方案，再加上下载/准备占用和安全余量；价格、费用上限及中断/恢复方式提交用户批准。没有实例价格和计时数据时，GPU 小时及费用为 `N/A`。

获批准且前置条件通过后，官方训练入口是 `python train.py data=pusht`。如需覆盖 `wandb`、数据格式、运行目录或恢复参数，逐项记录精确命令与变更；禁止隐式调整训练预算。记录每次优化步参数哈希/差异、损失、显存、总 GPU 时间、日志及 Checkpoint 哈希。先完整训练官方 Push-T，再独立进程加载 Checkpoint，运行固定评测协议 `python eval.py --config-name=pusht.yaml policy=pusht/lewm`；评价数据不能来自训练拆分。评测路径约定和输出格式须以实际版本/工件验证。

任何训练失败需保存退出码、首个有效错误、配置和不完整产物；不得以预训练权重推理、Mock 或 README 目标值填补。只有这套收据完成并审核后才能开始 G1 正式实验。
