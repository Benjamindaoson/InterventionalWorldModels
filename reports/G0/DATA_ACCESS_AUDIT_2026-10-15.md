# G0 数据与 AutoDL 入口核验回执（2026-10-15）

## 身份与范围

- 运行机器：Windows 本地；仓库 `D:\robotics\InterventionalWorldModels`，分支 `feat/world-model-v2`。基线 HEAD `dbdda2180f80f2c81c5f9ff541f93343327d7ab0`，与当时 `origin/feat/world-model-v2` 一致；用户提及的 `ef735ba...` 是更早提交，本次未重置或覆盖后续提交。
- 上游 LeWM：`lucas-maes/le-wm@8edfeb336732b5f3ce7b8b210d0ba370a09e2cac`。本次仅审计数据来源、加载路径和连接，不训练、不更改上游算法、不下载约 14 GB 的全量数据。
- 阶段状态：**G0 仍阻塞**。G1–G6 未开始。无 GPU、Checkpoint 或研究指标。

## 数据源与 Loader

| 核验项 | 结果 |
| --- | --- |
| 原始来源 | [`quentinll/lewm-pusht@655cd446`](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/655cd446b9929369d7d406001da85c15d1457850)，`pusht_expert_train.h5.zst`，13,136,247,974 bytes；未在本机解压读取 |
| Lance 候选 | [`galilai-group/lewm-pusht@ea321e39`](https://huggingface.co/datasets/galilai-group/lewm-pusht/tree/ea321e392348e3c65a18ab0d685f00e57be2c3e0)，`.lance/` 三个分片合计 14,295,080,031 bytes；公开卡片称为官方数据的 Lance 版本，但转换配方与内容等价未证实 |
| 字段 | Hugging Face first-rows 预览有 `episode_idx`、`step_idx`、`action`、`pixels`、`proprio`、`state`；图像显示 224×224。预览 JPEG URL 不能证明 Lance 内部 codec |
| 静态接口 | 固定 LeWM `config/train/data/pusht.yaml` 请求 `.lance` 名称及 `pixels/action/proprio/state`；`train.py` 传给 `swm.data.load_dataset`。候选字段和目录名吻合，属于静态可行性证据 |
| 运行时兼容性 | **未验证**。本地 `.venv` 无 `stable_worldmodel`、`lance`、`lancedb`、`pyarrow`；无实际 Lance 文件或锁定的可运行依赖环境。不得将静态吻合写成加载成功 |
| 来源差异 | 两仓库提交记录为同一上传者 `quentinll`；尚无转换日志、行对行/像素对像素比较或本地数据哈希 |

来源、精确大小、schema 及证据边界详见 [一手资料核验](../../docs/G0_LANCE_SOURCE_RESEARCH_2026-10-15.md)。[stable-worldmodel 官方 README](https://github.com/galilai-group/stable-worldmodel/blob/main/README.md)说明 `.lance` 是支持的格式；其主分支文档是接口线索，**不是本项目已冻结的依赖版本或实读证据**。

## 官方划分风险与独立评测协议

[固定版 LeWM `train.py`](https://github.com/lucas-maes/le-wm/blob/8edfeb336732b5f3ce7b8b210d0ba370a09e2cac/train.py) 先创建 `num_steps=4`、`frameskip=5` 的窗口数据集，再对窗口调用 `spt.data.random_split(dataset, [0.9, 0.1], generator=...)`。源码没有显式 episode 分组。因此相邻/重叠窗口可能跨训练和验证集合；实际交叉数量尚未测量，不能断言已发生具体泄漏。官方复现保持原协议，报告中将其随机划分指标与独立评测分开。独立评测需冻结 `source_revision + episode_idx` 分组和随机种子，确保每个 episode 的全部窗口只在一个集合；测试集冻结后不得参与模型选择。待真实数据可读时保存组列表、哈希和交叉检查收据。

## AutoDL 连接

在 Windows 执行：

```text
ssh -G autodl-p1 | Select-String '^(hostname|port|user) '
Test-NetConnection -ComputerName connect.cqa1.seetacloud.com -Port 14946
ssh -o BatchMode=yes -o ConnectTimeout=8 autodl-p1 'true'
```

解析到 `user=root`、`hostname=connect.cqa1.seetacloud.com`、`port=14946`；DNS 得到 `58.144.141.28`，`TcpTestSucceeded=False`；SSH 报 `banner exchange: Connection to UNKNOWN port -1: Connection refused`。没有建立 SSH 会话。此结果无法区分实例停止、控制台地址端口变更或网络侧拒绝。**GPU 型号、CUDA/PyTorch、剩余磁盘、运行进程、持久化策略、小时价均未核验。** 不记录 SSH 凭据。

需要用户从 AutoDL 控制台提供：原实例是否运行、最新 SSH 命令中主机/端口/用户名（删除密码和密钥）、GPU 型号/显存、实例每小时价格、系统盘与数据盘容量和停机持久化规则。取得后先只读检查并避免影响现有任务。

## 容量与预算门槛

候选 Lance 全量仅数据分片就需传输约 **14.30 GB**，还需为元数据、缓存、训练日志与 Checkpoint 留余量；具体磁盘预留须在服务器容量核验后给出。当前没有 GPU 吞吐、每 epoch 时间或实例价格，**GPU 小时与人民币预算均为 N/A**。先申请有明确上限的小样本读取和 GPU smoke test，记录时间、显存、吞吐，再按冻结 epoch 数、实例小时价及数据准备时间计算正式训练成本；正式训练另行获得授权。

## 验证与未完成项

- 已执行：本地 Git 状态、固定上游配置和 `train.py` 静态审计、Hugging Face 公开元数据核验、Windows SSH 配置/TCP/非交互连接检查。
- 未执行：全量下载、LeWM/稳定 Loader 真实样本读取、HDF5 与 Lance 数值/像素对照、AutoDL 主机检查、GPU smoke test、模型训练和独立评测。
- Windows `.venv` 执行 `python -m pytest -q`：**收集失败**，`torch_python.dll` 加载返回 WinError 4551；这是本机 PyTorch 运行环境错误，未进入 `test_learning.py`。执行 `python -m pytest -q --ignore=tests/test_learning.py`：**22 passed**。
- WSL 项目环境执行 `.venv-wsl/bin/python -m pytest -q`：**24 passed**。`openspec validate world-model-v2-g0`：有效；JSON 可解析；`git diff --check`：通过。上述均为工程测试，不是数据 Loader/训练验收。
- Git Commit/Push：以提交及推送命令的实际结果为准，不以本段代替 Git 回执。
