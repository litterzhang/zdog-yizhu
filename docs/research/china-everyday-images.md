# 972 个候选物种的公开图片采集

日期：2026-09-23。用户确认可从 Pl@ntNet-300K 之外的来源下载训练图片。
本轮只采集，不自动训练、替换线上模型或修改植物目录。

**14:08 后续更新：用户确认仅 B 的 883 种需要新增训练，原 972 类采集服务已停止，
合格图片已复用到独立新增数据集，旧文件保留。当前运行任务与入口见
`plantnet-additions-data.md`；下文为原任务的历史记录。**

## 固定范围

候选池 `everyday-candidate-pool-20260922-v3/candidates.json` 中，
`priorityReview=true`、`rankStatus=种` 且有物种 ID 的 **972 个唯一物种**。
保留全部目标，包括中文显示名缺失的条目；不按照片数量换物种。
物种 ID、学名、区域与频次证据，以及源文件 SHA-256 冻结在新数据集
`species.json`。这仍是区域资料支持的候选，不是全国常见度排名。
旧 `china-common-1000/paused.json` 不删除，原候选池不修改。

## 来源和处理

- iNaturalist 官方 `v1/taxa` 精确匹配学名、有效 species 级分类及植物祖先。
  不使用模糊名称匹配；异名、歧义、截断结果列为缺口，不擅自替换。
- 官方 `v1/observations` 获取全球 research-grade 记录；每张照片独立检查
  CC0 / CC BY 许可、公开图片桶与署名。不调用任何图片识别 API。
- 沿用每观察最多一图、每摄影者每物种最多三图、摄影者全局固定划分，
  SHA-256 与 dHash 去重，500px 图片重编码并去除元数据。
- 每物种目标 **70 train / 15 validation / 15 test**，合计最多 97,200 张。
  最多读取 20 页，每页 100 条；完成采集轮次不等于达到目标。
  训练入口读取数据集自己的最低样本数，不回退到旧的 20/5/5。
- 单请求流最多 30 次/分钟，403/429 停止；保留至少 4 GiB 可用磁盘。
  临时任务 CPUQuota=50%、MemoryMax=768M，无收费资源或自动重启。

研究级标签不等于专家金标准；栽培植物可能不足。照片来自全球，不是中国现场
专集。与基础模型预训练或以前实验照片的重合尚未排除，不称为全新验收集。
不会逐张人工看图标注，也不访问用户私人记录。

## 运行与进度

从项目根目录运行（可续跑；同一数据集有排他锁）：

```sh
server/.venv/bin/python -m yizhu_server.everyday_training_data prepare
server/.venv/bin/python -m yizhu_server.everyday_training_data collect
```

独立临时服务：`yizhu-everyday-images-20260923.service`。
不会随系统启动自动恢复；失败后先检查原因，再使用同一命令续跑。

数据目录：`server/.cache/training/china-everyday-972/`：

- `species.json`：固定的 972 类及证据；不要编辑。
- `taxon-matches/`、`resolution.json`：匹配结果与未解决的分类。
- `samples/`：逐物种样本、来源、许可、署名及哈希，实时落盘。
- `progress.json`：阶段、当前物种及图片数量，失败记录异常类型。
- `collection.json`：固定采集时间上界、目标数、页数及清单哈希。
- `coverage.json`、`manifest.jsonl`：一次采集遍历结束后输出；覆盖报告包含缺口。

日志：`server/.artifacts/china-everyday-972/collect.log`。
图像采集受可用空间限制，不能保证本机能容纳全部目标；不足时停止，
不删除旧模型、用户照片或其他文件，也不自动开通存储。

官方接口及图片说明：
`https://api.inaturalist.org/v1/docs/`、
`https://github.com/inaturalist/inaturalist-open-data`。

## 实际状态

已创建 972 类清单并启动独立后台任务，首个物种山樱花已匹配，
已有公开图片落盘。完整数量以实时 `progress.json` 和 `samples/` 为准。
尚未完成全量下载，也未启动本批分类器训练。

13:27 左右读取的快照：已检查 13 个目标，10 个精确匹配、3 个未匹配；
532 张图片已落盘，涉及 8 个物种，其中 4 个满足完整 70/15/15 划分。
已读取 systemd 状态为 active/running。Ruff 通过，408 项后端测试通过；
这是采集实现和离线测试验证，不是分类准确率或微信真机验收。
