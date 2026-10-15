# G0 Push-T Lance 数据源核验（2026-10-15）

**范围：**仅核查 Hugging Face 一手仓库卡片、固定提交、文件树 API、提交历史和数据预览 API。没有下载完整数据、执行 LeWM/stable-worldmodel Loader、训练模型或证明两种格式内容等价。

## 来源与版本

| 项目 | 压缩 HDF5 来源 | 现成 Lance 候选 |
| --- | --- | --- |
| 仓库 | [`quentinll/lewm-pusht`](https://huggingface.co/datasets/quentinll/lewm-pusht/tree/655cd446b9929369d7d406001da85c15d1457850) | [`galilai-group/lewm-pusht`](https://huggingface.co/datasets/galilai-group/lewm-pusht/tree/ea321e392348e3c65a18ab0d685f00e57be2c3e0) |
| 本次核查提交 | `655cd446b9929369d7d406001da85c15d1457850` | `ea321e392348e3c65a18ab0d685f00e57be2c3e0` |
| 卡片声明 | 用于 LeWorldModel Push-T 的 official dataset | 用于 LeWorldModel Push-T 的 official dataset 的 Lance version |
| 公开许可标签 | MIT | MIT |
| 公开数据文件 | `pusht_expert_train.h5.zst`，13,136,247,974 bytes | `pusht_expert_train.lance/`，三个数据分片合计 14,295,080,031 bytes，另有小型事务与 manifest 文件 |

文件名、精确大小与 Git LFS SHA-256 对象标识取自[压缩 HDF5 固定版本文件树 API](https://huggingface.co/api/datasets/quentinll/lewm-pusht/tree/655cd446b9929369d7d406001da85c15d1457850?recursive=true&expand=true)及[Lance 固定版本文件树 API](https://huggingface.co/api/datasets/galilai-group/lewm-pusht/tree/ea321e392348e3c65a18ab0d685f00e57be2c3e0?recursive=true&expand=true)。卡片原文见[压缩 HDF5 README](https://huggingface.co/datasets/quentinll/lewm-pusht/raw/655cd446b9929369d7d406001da85c15d1457850/README.md)和[Lance README](https://huggingface.co/datasets/galilai-group/lewm-pusht/raw/ea321e392348e3c65a18ab0d685f00e57be2c3e0/README.md)。两仓库均标为公开、无 gated 访问；[HDF5 API](https://huggingface.co/api/datasets/quentinll/lewm-pusht/revision/655cd446b9929369d7d406001da85c15d1457850)、[Lance API](https://huggingface.co/api/datasets/galilai-group/lewm-pusht/revision/ea321e392348e3c65a18ab0d685f00e57be2c3e0)显示 `license: mit`。这只是仓库声明，尚未对原始数据逐项做许可审计。

[HDF5 仓库提交历史](https://huggingface.co/api/datasets/quentinll/lewm-pusht/commits/655cd446b9929369d7d406001da85c15d1457850)显示 `quentinll` 于 2026-03-26 上传压缩文件；[Lance 仓库提交历史](https://huggingface.co/api/datasets/galilai-group/lewm-pusht/commits/ea321e392348e3c65a18ab0d685f00e57be2c3e0)显示同一用户 `quentinll` 于 2026-05-13 上传 Lance 文件夹。这支持同一上传者及卡片所称的来源关系，**不能证明**文件转换步骤、行顺序、图像像素或全部样本完全相同。卡片未给转换脚本、源文件哈希、逐行校验收据或图像重编码参数。

## 字段与编码证据

[Lance 固定提交的 Hugging Face first-rows API](https://datasets-server.huggingface.co/first-rows?dataset=galilai-group/lewm-pusht&config=default&split=train)展示六列：`episode_idx` `int32`、`step_idx` `int32`、`action` 固定长度 2 的 `float32`、`pixels` Image、`proprio` 固定长度 4 的 `float32`、`state` 固定长度 7 的 `float32`。样本预览把图像显示为 **224×224**。仓库[数据页](https://huggingface.co/datasets/galilai-group/lewm-pusht)估计约 **2,336,736** 行，但网页仅展示前 **815,360** 行；估计值不能作为实际 Loader 计数。

预览图 URL 以 `image.jpg` 结尾，这是 Hugging Face 预览服务的输出，**不能据此断定 Lance 内部图像以 JPEG 编码，更不能断定与 HDF5 原图逐字节一致**。Lance manifest、实际图像 payload/codec、色彩通道及解码后的像素仍需在受控小样本读取中核实。HDF5 仓库仅公开压缩文件，仓库卡片没有逐字段 schema；在未读取该文件或取得一手 schema 前，不能声称两份数据的字段和值匹配。

## 当前判定与下一项可验证工作

- **已验证：**现成 Lance 候选确实公开、具有训练配置期望的 `.lance` 目录名；元数据与预览显示适合继续测试的 Push-T 六字段。
- **未验证：**固定版本 LeWorldModel 与 stable-worldmodel Loader 的真实读取、实际行数/episode 数、时间索引连续性、所有数据分片哈希、原 HDF5 对应关系与图像编码差异。仓库卡片中的“Lance version”是发布者声明，不能替代实测。
- **下一步：**先核对固定 Loader 版本和能否从 Hugging Face 仓库以有边界的远程读取方式取得少量行；若需下载全量，先核实至少约 **14.3 GB 网络传输**，以及运行时缓存、解压/复制、日志的磁盘余量。获得一对可对应的 HDF5/Lance 样本后，比对 episode/step、数值字段及解码后像素；保存源码提交、数据提交、文件 SHA-256、读取代码和差异报告。全量下载和正式训练均不在本次核验范围。
