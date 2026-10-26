# Pl@ntNet 新增物种 B 的图片采集

2026-09-23。按用户确认，当前已有支持 A=420 种；新增计划 B=883 种。
40 个与 A 重叠的候选不再采作新增类别。B 未训练完成前，不扩大支持元数据。

## 实际范围与迁移

新数据目录：`server/.cache/training/plantnet-china-additions-883/`。
目标来自固定的 `docs/research/plantnet-plant-sets-20260923/sets.json` 中 B，
并校验 A/B 无交集、稳定 ID/学名唯一、中文名存在及数量一致。
`species.json` 保存源集合与名录哈希，不手工修改；A 的 ID 保留作去重和边界检查，
不作为新数据集的训练类别。计划合并范围 1,303 不等于已经支持 1,303。

14:08 已停止旧 `yizhu-everyday-images-20260923.service`，保留全部旧文件；
在两端数据锁下导入 **1,377 张、27 个 B 物种**的已有授权图片，
另外 183 张不属于 B 的旧照片未复制，但没有删除。
来源、署名、许可、观察者划分和图像哈希保持不变，导入记录为 `seed-import.json`。
没有复制旧“完成一轮但不足额”的状态来假装新目标达成。

新服务：`yizhu-plantnet-additions-20260923.service`，已确认 active/running；
旧服务已确认 inactive/dead。两者没有同时采集。
14:12 快照：新目录已有 1,478 张图片，9 类满足 70/15/15；
仍远未完成全部 883 类。数量继续变化，应读取当前逐类样本文件。

## 精确名称证据

继续使用 iNaturalist 官方元数据与 CC0/CC BY 开放照片：

- 保留直接学名精确匹配。
- 另允许来源平台在 `names` 中**明确列出的同名 scientific-names 条目**
  关联到唯一、有效、species 级植物分类。不是相似字符串或俗名匹配。
- 照片的当前来源学名必须与该来源分类一致；保存
  `sourceScientificName` 和 `sourceNameEvidence`，不改写本地物种 ID/学名。
- 若返回的任何科学名称已对应另一个 A/B 目标，或多个目标对应同一来源分类 ID，
  标为冲突并跳过，不静默把两个物种合成一类。
- 歧义、非植物、非 species、失效分类和截断检索结果均不接受。

这表示来源平台的名称关联证据，不宣称跨所有分类体系的最终接受名。
只记录科学名称证据，不导入额外语言名称或改变中文目录。

## 可续跑命令

从项目根目录运行：

```sh
server/.venv/bin/python -m yizhu_server.plantnet_training_data prepare

# 仅在采集前运行；源和目标都需要能取得排他锁，不覆盖已启动采集的数据
server/.venv/bin/python -m yizhu_server.plantnet_training_data seed \
  --source server/.cache/training/china-everyday-972

server/.venv/bin/python -m yizhu_server.plantnet_training_data collect
```

目标每物种 train=70 / validation=15 / test=15，最多 88,300 张。
一次观察一图、每作者每种最多三图、作者固定划分、全局去重、脱除元数据，
每种最多查 20 页。完整轮次不等于每个物种足量。
403/429 停止；至少保留 4 GiB 空间；CPUQuota=50%、MemoryMax=768M；
无收费资源或自动重启。失败后先看日志再续跑，不绕过访问限制。

进度：`progress.json`、`resolution.json`、`samples/`；
一轮结束输出 `coverage.json` 和 `manifest.jsonl`。
日志：`server/.artifacts/plantnet-additions-20260923/collect.log`。

## 验证与未完成事项

### 2026-09-23 16:47 串行回放实验更新

已停止 `yizhu-plantnet-additions-priority-20260923.service`，确认 inactive/dead。
新的 `yizhu-plantnet-replay-20260923.service` 已确认 active/running，
先采集单独 A 池的首批 12 类，再运行两组离线对照训练，成功后自动续跑原 B 池。
新的资源限制、实际阶段和数据缺口见 `plantnet-retained-replay.md`。
这不是增加重复的 A 输出，也未改变 B=883 的冻结目标。

### 2026-09-23 下午续跑更新

已测试 `collect --priority-species bioclip:plant-009`：只将桂花（木犀）
排到已有 B 集合的前面，不增加目标、不修改冻结的 `species.json`，
不提前扩展线上支持范围。8 项采集模块测试及该模块 Ruff 检查通过。
执行顺序与目标文件哈希记录在 `run-order.json`。

旧 `yizhu-plantnet-additions-20260923.service` 已确认 inactive/dead；
续跑服务为 `yizhu-plantnet-additions-priority-20260923.service`，
已确认 active/running。两者不并行采集，沿用同一数据目录、日志、
限流和资源上限。下面的历史验证数字不代表当前模型支持范围。

Ruff 与 429 项后端测试通过。已核对新增目录中的照片全部属于 B，和 A 无交集；
当前方案 `supportedPlantSet` 仍为 420，没有部署或新增支持合同。
仅训练 B 的实验头不能自动声称承接 A；后续组合/扩展模型还需验证 A 的保留能力。
完整采集、数据缺口、模型扩展及独立中国场景验收仍待完成。
