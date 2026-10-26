# 植物存在二分类

日期：2026-09-22。目标是判断整张图中是否有植物，不判断植物名称。
植物在背景中也算正类；不因物种未收录、没有中文名或物种识别失败而判为非植物。
本模块离线运行，不自动修改线上识别方案。

## 数据与标签

使用 Open Images V7 **人工核验图像级标签**，类别 `Plant`（`/m/05s2s`）：

- `Source` 只接受 `verification` / `crowdsource-verification`。
- `Confidence=1` 为正类，`Confidence=0` 为明确负类。这里是标注值，不是模型概率。
- **缺少 Plant 标签、机器生成标签、只有动物标签，都不能作为负类依据。**
- 排除正负冲突，以及 Plant=0 但树、花等官方后代类别或草、叶、苔藓、藻、蕨
  明确为正的冲突记录。这是保守的已知冲突过滤，不保证消除全部标注噪声。
- 按图片元数据保留 CC BY 2.0 许可、作者、来源链接；不下载私人用户照片。

元数据得到 5,821 个明确正样本、6,694 个通过冲突过滤的明确负样本。
初次选取 600 张、每位摄影者最多一张；摄影者标识哈希决定项目内的数据分组。
计划每类 200 train / 50 validation / 50 test，尺寸超限和近重复在训练前排除并记录。

这些图片来自官方 **validation 子集**，重新划分为本项目训练/验证/测试。
**这是小规模自定义实验，不是 Open Images 官方基准成绩。**
当前标签直接来自数据集，没有再逐张使用助手视觉判断生成标签。
之前 iNaturalist 候选图片及助手查看记录保留，但不用于这次训练。

官方来源（已直接读取下载页、类别表、标签和许可元数据）：

```text
https://storage.googleapis.com/openimages/web/download_v7.html#df-image-labels
https://storage.googleapis.com/openimages/v7/oidv7-val-annotations-human-imagelabels.csv
https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions.csv
https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json
https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv
https://raw.githubusercontent.com/openimages/dataset/master/downloader.py
```

原始元数据、来源页副本、SHA-256、选择清单及逐图许可位于：
`server/.cache/training/plant-presence-openimages/`。
图片按官方下载器的公开 S3 路径下载；未执行下载的第三方 Python。

## 模型与评估

- 已下载的官方 Pl@ntNet ResNet50，严格加载后移除原 1,081 类分类层。
- 冻结骨干，使用全图等比例缩放补边至 224×224、ImageNet 标准化、
  2,048 维 L2 归一化特征；训练 `Linear(2048, 2)`，共 4,098 个参数。
- 复用本地 `fit_head`：交叉熵、AdamW、学习率/权重衰减均为 0.01，
  固定随机种子，最多 100 epochs，按验证损失早停。
- 在验证集上选择满足至少 95% 植物召回率的最高阈值；95% 是验证集目标，
  **不是对测试集或线上准确率的承诺**。测试集不用于训练、选 epoch 或选阈值。
- 指标：植物召回率、非植物召回率、非植物误放行率、精确率及混淆矩阵。
  二分类不使用 Top-5，softmax 不视为已校准概率。

## 已完成训练与测试

实际使用 599 张、599 个不同摄影者分组；1 张超过 20 MP，在训练前排除。
训练集为 200 正 / 199 负，验证集与测试集均为 50 正 / 50 负。
训练运行 33 epochs，验证损失最优为第 18 epoch；固定阈值为
`0.3770689070224762`，只在验证集选择，测试后未调整。

| 指标 | 验证集 | 保留测试集 |
| --- | ---: | ---: |
| 植物正确放行 | 48 / 50 | 44 / 50 |
| 植物误拒绝 | 2 / 50 | 6 / 50 |
| 非植物正确拒绝 | 35 / 50 | 36 / 50 |
| 非植物误放行 | 15 / 50 | 14 / 50 |
| 植物召回率 | 96% | 88% |
| 非植物误放行率 | 30% | 28% |
| 整体准确率 | 83% | 80% |

**模型已真实训练并保存，但当前不适合直接上线硬拦截。**
测试中仍会误拒绝 6 张植物图片、放过 14 张非植物图片；不能将小样本 80%
写成线上准确率，也不能把验证集的 95% 目标写成已经在测试集达成。

- 分类头：`server/.cache/models/plant-presence/resnet50-linear-v1.json`，115,871 bytes；
  骨干仍依赖本地已校验的 ResNet50 权重，不包含在这个小 JSON 中。
- 完整测试结果：`server/.artifacts/plant-presence/resnet50-linear-v1-test.json`。
- 训练摘要、文件哈希与数据来源：`docs/research/plant-presence-training.json`。
- 本轮 Ruff 通过；后端 **361 项测试通过（7.68 秒）**。测试集阈值、
  模型文件和数据清单的哈希关联已比对一致。
- 未修改客户端、线上配置或生产数据库，未使用私人图片，未调用在线识别服务。

## 本地命令

以下从项目根目录运行。数据集和模型输出不得覆盖已有版本。
元数据已经缓存，采集入口无需安装新的依赖：

```sh
# samples.json 冻结前可复用已下载图片；冻结后拒绝重新采集
server/.venv/bin/python -m yizhu_server.plant_presence_data \
  --data server/.cache/training/plant-presence-openimages

server/.venv/bin/python -m yizhu_server.plant_presence check-data \
  --data server/.cache/training/plant-presence-openimages

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
server/.venv/bin/python -m yizhu_server.plant_presence train \
  --data server/.cache/training/plant-presence-openimages \
  --output server/.cache/models/plant-presence/resnet50-linear-v1.json

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
server/.venv/bin/python -m yizhu_server.plant_presence evaluate \
  --data server/.cache/training/plant-presence-openimages \
  --head server/.cache/models/plant-presence/resnet50-linear-v1.json \
  --output server/.artifacts/plant-presence/resnet50-linear-v1-test.json

server/.venv/bin/python -m yizhu_server.plant_presence predict \
  --head server/.cache/models/plant-presence/resnet50-linear-v1.json \
  --photo /path/to/photo.jpg
```

## 使用边界

标签可能有噪声，缩放可能丢失很小的背景植物；尚未专门验证中国用户拍摄分布、
仿真植物、印花、屏幕植物照片和极暗/模糊照片。摄影者和近重复隔离不能排除
与骨干预训练图片的重合。原骨干权重的独立许可仍需在公开部署前确认。

未接入线上拒识，所有输出保留 `productionApproved=false`。
旧物种训练的暂停标记不移除；这是独立数据集和独立二分类头。

## 扩展数据与骨干对照（2026-09-23 已完成）

新数据版本：`server/.cache/training/plant-presence-openimages-v2/`。
已冻结选择清单：6,000 张、6,000 个不同摄影者，目标 4,000 train /
1,000 validation / 1,000 test。仍使用同一官方人工标注来源；不是逐张重新标注。
旧的 100 张测试图片、其摄影者及相同原图 MD5 被排除，采集时还用旧测试图片的
精确哈希/感知哈希过滤重复。旧训练和验证图片仅在分组不变且哈希一致时复用。
实际最终数量须以下载、尺寸检查及去重完成后的 `samples.json` 为准。

**9 月 23 日续传、两轮训练与测试均已完成**：最终冻结 5,990 张、5,990 个不同摄影者；
训练集 1,996 正 / 1,996 负，验证集 500 正 / 499 负，测试集 499 正 / 500 负。
排除 7 张不支持的实际图片格式、2 张超过 20 MP 的图片、1 张精确/近重复图片。
已比对旧测试 ID/摄影者隔离和训练计划数据哈希，未改变冻结的选择规则。
数据 SHA-256：`ec8e4b45bb3ad059f5fb7a09c489d362d6cd2ee8043ffa8a98aff0599c86c85b`。
完整结果在北京时间 9 月 23 日 09:36:36 写入，任务状态为 `complete`。

比较两种初始化，均冻结 ResNet50 并训练相同的两类线性头：

1. 官方 Pl@ntNet ResNet50（沿用已校验权重）。
2. TorchVision `ResNet50_Weights.IMAGENET1K_V2`。已下载 **102,540,417 bytes**，
   校验官方文件名哈希前缀并固定完整 SHA-256：
   `11ad3fa62ca79e40addfd354a8ec4b7c75143b3038b8d2a807fbc68deab379ca`。
   `weights_only=True`、原始 1,000 类参数严格加载通过，再移除分类头。

两者使用相同的全图补边预处理，而非 ImageNet 权重默认中心裁剪；
因此这不是官方 ImageNet 分类基准复现。编码器、预处理、图片内容和
Torch/TorchVision 版本共同决定特征缓存命名空间，不能混用两个骨干的特征。
采用低优先级、2 个 CPU 线程。单张已有公开训练图片上的 ImageNet 特征提取
5 次测量中位耗时为 1 线程 0.3252 秒、2 线程 0.1532 秒、4 线程 0.1058 秒；
这只是本机资源检查，不是完整数据集吞吐或分类准确率。

流程由 `yizhu_server.plant_presence_compare` 执行：

- 两个模型只根据验证损失选 epoch；阈值只在验证集选择。
- 在达到 95% 验证集植物召回目标的候选中，以验证集误放行率选择骨干。
- **先保存 `validation-selection.json`，再评估两个模型的封存测试集**；
  不因测试结果改变选择，也不自动启用线上模型。
- 旧版 80% 与新版更大测试集不是同一批样本，不能直接相减宣称提升幅度。

### 同一封存测试集的结果

两者使用完全相同的 999 张测试图片（499 张植物、500 张非植物）。
模型权重、数据、固定阈值与测试报告的哈希关联已检查；
已从逐张结果重新计算混淆矩阵，并核对选择文件早于测试结果写入。
两个保存后的分类头也分别重新加载，用原始公开图片复现了各一条预测，
与对应缓存评测分数的差值均小于 `1e-6`。

| 指标 | Pl@ntNet 初始化 | ImageNet 初始化 |
| --- | ---: | ---: |
| 验证集植物召回率 | 95.00% | 95.00% |
| 验证集非植物误放行率 | 35.27% | 18.84% |
| 测试集植物召回率 | 96.59%（482/499） | 92.79%（463/499） |
| 测试集非植物误放行率 | 36.40%（182/500） | 21.80%（109/500） |
| 测试集准确率 | 80.08%（800/999） | 85.49%（854/999） |
| 植物误拒绝 | 17 张 | 36 张 |
| 非植物正确拒绝 | 318 张 | 391 张 |
| 固定阈值 | 0.1935620903968811 | 0.3697161078453064 |
| 训练轮数 / 最佳轮数 | 95 / 80 | 27 / 12 |

按预先约定的**验证集**规则，选择已冻结为 ImageNet，之后没有依据测试结果改选。
相比同轮 Pl@ntNet，ImageNet 少放行 73 张非植物，但多拒绝 19 张植物；
这是权衡，不是所有指标都更好。**ImageNet 的测试召回率未达到 95% 目标**，
当前仍不适合直接上线硬拦截；不能把 85.49% 当作真实用户环境的准确率保证。

模型与报告：

- `server/.cache/models/plant-presence/comparison-v2/plantnet-resnet50-linear.json`
- `server/.cache/models/plant-presence/comparison-v2/imagenet-resnet50-linear.json`
- `server/.artifacts/plant-presence/comparison-v2/summary.json`
- `docs/research/plant-presence-comparison-v2.json`：包含结果、数据来源、哈希与限制的摘要。

两个分类头都已保存；预测时自动读取头文件中的骨干类型，无须改客户端或服务器环境变量：

```sh
server/.venv/bin/python -m yizhu_server.plant_presence predict \
  --head server/.cache/models/plant-presence/comparison-v2/imagenet-resnet50-linear.json \
  --photo /path/to/photo.jpg
```

未启用线上拒识、未重启服务、未恢复旧物种训练。
后续若进行局部微调，应继续只在训练/验证数据上开发，不能根据这次测试误差反复调阈值，
也不能把这些已观察的测试结果重新包装成全新的最终验收。

9 月 23 日续跑检查：Ruff 通过；后端 **369 项测试通过（14.33 秒）**。
此前新版加载器还使用原始 v1 分类头，
复现了 3 条已保存的公开图片预测（分数误差小于 `1e-6`）。

后台任务当前状态与日志：

```sh
cat server/.artifacts/plant-presence/comparison-v2/run-status.json
tail -n 30 server/.artifacts/plant-presence/comparison-v2/run.log
```

9 月 22 日的下载在约 4,700 张时遇到连接超时，旧等待脚本随后在两小时后退出，
未启动训练。失败日志保存在 `run-attempt-1.log` 与 `run-attempt-1-status.json`。
9 月 23 日已获授权实际续跑，复用已有图片；改为同一 CLI 顺序执行下载、训练、
冻结验证集选择和评估。网络失败立即写 `failed` 状态，不再让训练队列空等。

`preparing-data` 表示下载/复核缓存，`training` 包含特征提取与分类头优化，
`evaluating` 表示封存测试，`complete` 才表示两轮全部结束。
数据阶段的 `processedImages` 是本次已复核条目数，不是累计下载文件数。
本次已为 `complete`，最终结果已写入同目录 `summary.json`。
分类头已写入 `server/.cache/models/plant-presence/comparison-v2/`。
后台运行期间不要重复启动相同任务。若中断，已编码特征可复用，输出不会静默覆盖。

首次启动或中断后的续跑命令（已有冻结数据时不会重新下载）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
server/.venv/bin/python -m yizhu_server.plant_presence_compare \
  --data server/.cache/training/plant-presence-openimages-v2 \
  --models server/.cache/models/plant-presence/comparison-v2 \
  --reports server/.artifacts/plant-presence/comparison-v2 \
  --threads 2 --prepare-data
```
