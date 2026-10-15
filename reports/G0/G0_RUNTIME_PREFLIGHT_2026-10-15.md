# G0 WSL Loader 与真实数据预检回执（2026-10-15）

**机器与身份：**Windows 仓库 `D:\robotics\InterventionalWorldModels`，WSL 路径 `/mnt/d/robotics/InterventionalWorldModels`，`feat/world-model-v2`；基线提交 `f81fed7172c3cf0da09a08712bc34e822b69b45e`。本轮未启动 GPU 实例、未训练模型。

## 运行环境与实际导入

在仓库内创建隔离的 `.venv-lewm-wsl`，Python `3.10.21`。先从 PyTorch CPU wheel 源安装 `torch==2.14.1+cpu`、`torchvision==0.29.1+cpu`，再安装 `stable-worldmodel[train]==0.1.1` 与 `stable-pretraining==0.1.8`。完整实际解析版本见 [`g0_wsl_loader_freeze.txt`](../../configs/lewm/g0_wsl_loader_freeze.txt)，含 `pylance==13.0.0`、`lancedb==0.40.0`、`pyarrow==25.0.1`。`stable-worldmodel` 首次导入因缺少 `imageio` 失败；补装 `imageio==2.38.0` 后，`stable_worldmodel`、`stable_pretraining`、`lance` 和固定 LeWM checkout 的 `train.py`、`jepa.py` 均实际导入成功。可选 ALE 环境缺失仅触发警告，未将其称为训练环境通过。

```bash
uv venv --python 3.10 .venv-lewm-wsl
uv pip install --python .venv-lewm-wsl/bin/python torch torchvision --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-lewm-wsl/bin/python 'stable-worldmodel[train]==0.1.1' 'stable-pretraining==0.1.8'
uv pip install --python .venv-lewm-wsl/bin/python imageio
uv pip freeze --python .venv-lewm-wsl/bin/python > configs/lewm/g0_wsl_loader_freeze.txt
```

## Loader 软件契约：隔离小表

执行：

```bash
.venv-lewm-wsl/bin/python scripts/g0_loader_probe.py fixture --output reports/G0/loader_fixture_2026-10-15.json
```

结果 [`loader_fixture_2026-10-15.json`](loader_fixture_2026-10-15.json)：**PASS_FIXTURE_ONLY**。用 50 行、2 个 episode 的小型 Lance 表调用真实 `swm.data.load_dataset`，配置 `num_steps=4`、`frameskip=5` 与上游四个输入列，得到 12 个窗口；样本 `pixels=[4,3,224,224]`、`action=[4,10]`、`proprio=[4,4]`、`state=[4,7]`。这证明已安装 Loader 的基本软件契约，不证明官方完整数据与 HDF5 等价。

## 官方 Lance 数据：有界远程实读

使用[Hugging Face 官方 Lance 远程读取路径](https://huggingface.co/docs/hub/datasets-lance)和仓库当前公开 HEAD 检查，执行：

```bash
HF_HOME=/tmp/iwm-g0-hf-cache .venv-lewm-wsl/bin/python scripts/g0_loader_probe.py remote --output reports/G0/loader_remote_2026-10-15.json
```

结果 [`loader_remote_2026-10-15.json`](loader_remote_2026-10-15.json)：**PASS_BOUNDED_LANCE_READ**。读取前后 HEAD 都为 `ea321e392348e3c65a18ab0d685f00e57be2c3e0`，底层 `lance.dataset` 返回 2,336,736 行及六列 schema；只请求第 0、1 行数值列和第 0 行图像。第 0 行图像 payload 为 5,789 bytes、JPEG、224×224 RGB，SHA-256 见 JSON。没有调用 `load_dataset("galilai-group/lewm-pusht")`，也没有下载全量数据。**限制：**当前 `pylance` 对带 `@SHA` 的 `hf://` URI 返回 401，因此实读使用当时 HEAD 路径并前后核 SHA；这不是不可变 revision URI，不能防止读取期间发生未被前后 HEAD 检查捕获的变化。

安装包源码显示 `stable-worldmodel` 的 Hugging Face 仓库名路径会先解析并下载缓存；官方 PushT 配置中的 `keys_to_cache=[action, proprio, state]` 还会扫描并加载这些整列。故本次未将官方完整配置直接用于远程实读。**真实官方数据经 `swm.data.load_dataset` 的完整兼容性仍未验证**，需有云端数据空间和传输批准后，在固定数据副本上测试 episode 结构、全量窗口索引与批次读取。HDF5 对应样本、完整图像编码统计、来源等价性也未验证。

## Epoch 协议与 GPU 状态

[LeWM 论文附录 E](https://arxiv.org/html/2603.19312)写明 PushT **10 epochs**；固定上游 [`config/train/lewm.yaml`](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/config/train/lewm.yaml)设 `trainer.max_epochs: 100`。定义两个独立身份：`upstream_config_100ep` 使用原配置，`paper_budget_10ep` 使用显式 `trainer.max_epochs=10` 覆盖。两者都**未执行**；训练日志、Checkpoint、评测不得混报。正式运行前还要核实其他论文与配置差异。

实际运行 Hydra `python train.py --cfg job data=pusht` 和 `python train.py --cfg job data=pusht trainer.max_epochs=10`，分别保存[100 epoch 组合配置](upstream_100ep_composed_2026-10-15.yaml)与[10 epoch 组合配置](paper_10ep_composed_2026-10-15.yaml)。两次均以退出码 0 **仅输出配置并退出**；验证字段分别为 `max_epochs: 100` / `10`，共同 `batch_size: 128`、`name: pusht_expert_train.lance`、`wandb.enabled: false`。这些不是训练日志。

用户尚未提供新的 AutoDL 控制台入口；旧 SSH 端口此前拒绝连接，本轮没有继续重试或猜测新地址。GPU 型号、CUDA、显存、磁盘持久化、价格、进程均 **N/A / 未预检**。取得有效入口后先只读核查，再提出不超过 30 分钟、含费用上限和停止条件的 smoke test 方案，等待授权。本轮真实训练与独立评测均未开始。

## 证据状态

本地工程回归：WSL 既有项目环境 `.venv-wsl/bin/python -m pytest -q` 为 **24 passed**；隔离环境对 `scripts/g0_loader_probe.py` 执行 `py_compile` 通过，三份 JSON 均可解析；`openspec validate world-model-v2-g0` 与 `git diff --check` 通过。运行命令与结果只证明本地预检，不是 GPU smoke test。

| 项目 | 状态 |
| --- | --- |
| WSL 软件导入与固定 LeWM 入口导入 | 已执行，通过；有 `imageio` 缺依赖修复记录 |
| 隔离小型 Lance 的 stable-worldmodel Loader | 已执行，通过；仅软件契约 |
| 官方 Lance 的底层有界读取 | 已执行，通过；两行数值和一张图像 |
| 官方 Lance 的 stable-worldmodel 完整 Loader | 未验证；整列缓存与 episode 扫描不能作为小样本调用 |
| HDF5/Lance 等价与全量数据身份 | 未验证 |
| AutoDL GPU 预检 | 被新入口信息阻塞 |
| 模型训练、Checkpoint、独立评测 | 未开始 |
