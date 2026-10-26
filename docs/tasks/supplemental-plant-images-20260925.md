# 缺图物种：其他来源补采

2026-09-25，用户明确要求从其他来源补图，同时保持当前训练。

另外针对旧采集器漏掉的具名杂交目标，已启动独立的严格来源桥接补采。此项使用既有iNaturalist来源，不冒充新的图片来源，详情见 `docs/tasks/named-hybrid-source-recovery-20260925.md`。

## 实际执行结果

已实现并运行 `server/src/yizhu_server/commons_plant_collection.py`，不再只是备选来源方案。

| 物种 | 本次下载的候选照片 |
| --- | ---: |
| 木犀 / 桂花（Osmanthus fragrans） | 6 |
| 重阳木（Bischofia polycarpa） | 3 |
| 橘（Citrus reticulata） | 9 |
| 合计 | **18** |

18张均有逐图CC BY许可及署名，文件存在、SHA校验和图像完整性检查通过。它们是额外来源的**候选图片**，尚未计入训练配额或模型支持数量；桂花原有24张不能直接加上这6张就宣称可训练。

## 降速恢复进展（2026-09-25 12:59）

冷却超过30分钟后，按官方API Etiquette的串行请求、错误退避和maxlag建议进行了一次显式恢复，未换主机或绕过权限。官方文档实际读取证据位于 `server/.cache/research/commons-policy-20260925/etiquette.json`。

- 新客户端 `commons_transport.py`：请求之间至少10秒，明确项目User-Agent，API请求带`maxlag=5`。
- 429/503记录服务端`Retry-After`（秒或HTTP日期）与最早恢复时间，等待不少于服务端指示及本地10分钟冷却；403不允许自动恢复。
- 原429响应头未被旧代码保留，因此第一次恢复的冷却采用本地保守规则，不声称服务端要求了10分钟。历史停止记录已单独保存。
- 恢复必须显式传入`--resume-after-cooldown`，默认拒绝重跑已停止任务；服务`Restart=no`，没有无限重试循环。
- 当前服务 `yizhu-commons-resume-20260925.service` 实际active/running，InvocationID `cb77e758e5ed4684a54b98548632cd23`。
- 截至核验共 **24张**候选照片：原18张，新增孔雀肖竹芋2张、金丝桃4张。全部SHA已核验，仍未划入训练集。
- 82类训练保持原InvocationID，最新检查为第10/30轮、第320/359批，未改参数。
- 传输退避与采集过滤测试合计 **30 passed**，Ruff通过。

独立的249种iNaturalist采集已正常完成：14,065张原图，覆盖218种，109种达到原70/15/15启动配额。不要把109种备齐与82类在训混为一谈。

## 首轮停止记录（历史）

- 任务：`yizhu-commons-20260925.service`。
- InvocationID：`d9b83be29666430f8c0dce045d70c841`。
- 冻结计划包含启动时仍缺图的150个物种，桂花排第一。
- 完成前11个物种的目录扫描后，Commons API 返回 **HTTP 429**。
- 服务已停止（failed/exit-code）；**不自动重启、不换主机绕过限制**，已下载文件和扫描进度保留。还没有完成全名单补采。
- 82类物种训练未受影响；2026-09-25 12:18检查仍为同一InvocationID，第6/30轮、第130/359批。

## 来源筛选

- 只读取明确学名对应的Commons分类，并要求标题或描述出现对应学名；不使用中文关键词模糊匹配自动贴标签。
- 只接受每张媒体明确声明的CC0或CC BY URL。CC BY-SA、非商业、禁止演绎、许可不明、泛称Public domain未纳入当前策略。
- 保留署名、原页面、许可地址、源文件SHA1及下载后SHA256。
- 排除元数据明确标为插画、雕版、标本、分布图、茶叶/干花等项目；这不是逐张人工图片审核。
- 单一作者同一物种最多3张；与主数据池做图片哈希及近重复检查；压缩并移除元数据。
- 每次请求至少间隔2秒，响应大小上限、4GiB磁盘保护生效。即使如此仍遇到限流，不能把客户端限速当成来源许可。
- 原图标签仍标记 `metadata-only`，`split=unassigned`、`trainingEligible=false`。进入后续训练前还需核对物种和跨来源作者/划分冲突，并重新与届时完整主数据池去重。

## GBIF 桂花核查

实际读取603条带图片的人类观察记录：

- 566条来自iNaturalist，不能算作新的独立来源。
- 35条来自Pl@ntNet，其媒体标注为CC BY-SA，超出现有许可白名单，未下载。
- 2条来自NABU/naturgucker，媒体字段要求进一步查看逐图权利声明，未下载。

因此本次没有把GBIF记录数量冒充新增图片数量，也没有用观察记录的许可代替照片许可。

## 文件及验证

- 主数据池剩余140种的分类缺口（14个来源名称问题、17个已匹配但本轮零图、109个划分配额不足）见 `docs/research/landscape-image-gaps-20260925.md` 及同名CSV/JSON。
- 候选图和逐图来源：`server/.cache/training/china-landscape-commons-20260925/`。
- 运行日志与核验摘要：`server/.artifacts/commons-collection-20260925/`。
- GBIF来源核查缓存：`server/.cache/research/gbif-backup-20260925/`。
- 采集器测试 **21 passed**，Ruff通过，包括许可白名单、学名证据、下载域名限制、非照片过滤及429停止行为。
- 没有修改首批/第二批训练、验证、测试图片，也没有接入线上或启动预训练权重方案。
