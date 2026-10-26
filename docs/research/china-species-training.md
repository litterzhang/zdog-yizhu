# 中文名录拆分后的中国植物训练

日期：2026-09-23。用户要求重构名录后继续训练中国植物分类器。
这与“图片是否有植物”的二分类是两个任务；本轮不替换线上识别方案。

## 本轮实际范围

先进行一个 **10 物种 / 300 张公开图片** 的独立训练版本：
构、鸭跖草、萝藦、莲、诸葛菜、酢浆草、红花酢浆草、垂序商陆、
白车轴草、阿拉伯婆婆纳。

- 稳定物种 ID、学名、中文主名/别名来自新分表名录。
- 类别必须精确匹配候选池中的稳定 ID 和学名，且有中国地区调查记录。
- 这 10 种现有证据来自上海调查资料，不宣称区域均衡或全国常见度排名。
- 有中文名本身不构成中国分布证据；没有证据、仅商品名/名称线索、品种、
  模糊映射和外部地区条目不进入此实验。
- 每种需具备 20 train / 5 validation / 5 test 图片。马齿苋缓存不足，未纳入。
- 共 200 训练 / 50 验证 / 50 测试，保持原摄影者分组及去重约束。
- 图片为已有 CC0 / CC BY 公开照片；可以来自世界各地的同种植物，
  不是中国境内拍摄专集。不读取私人观察照片。
- 原候选池仍待更广泛的分类/区域核对，不将其中全部 3,026 条标为训练已批准。

这是复用已有授权数据的有界实验，不是完成了 1,000 种或全部中国植物训练。
原 `china-common-1000/paused.json` 保留，旧观察频次排名的采集/训练不恢复。

## 模型和资源

沿用已安装的 BioCLIP 2 冻结图像编码器，训练 10 类线性分类头。
复用 90 份匹配图片/编码器版本的缓存特征，其余 210 份补算。
用验证集早停，报告 Top-1 / Top-5、逐类召回及实际类别集合。
这些公开测试图片曾用于以前的模型试跑，不包装成全新、从未观察过的验收集；
也不能将 10 类闭集成绩推广为中国全部植物的准确率。

为避免影响现有 8 GiB 主机上的 API，训练使用独立用户级临时服务，
CPU 上限 1 核、Nice=10、MemoryHigh=2 GiB、MemoryMax=2,500 MiB、
MemorySwapMax=1 GiB。达到限制可能让训练失败，不能因此静默换模型或移除限制。

## 文件和运行

- 数据：`server/.cache/training/china-species-20260923/`
- 类别及资料依据：上述目录 `species.json`
- 输出模型：`server/.artifacts/models/china-species-bioclip2-v1/`
- 训练进度：数据目录 `progress.json`
- 日志：`server/.artifacts/china-species-20260923/train.log`
- 任务：`yizhu-china-species-training-20260923.service`

准备命令已执行；现有输出不会覆盖：

```sh
server/.venv/bin/python -m yizhu_server.chinese_species_training \
  --source server/.cache/training/china-common-1000 \
  --pool server/.cache/research/everyday-candidate-pool-20260922-v3/candidates.json \
  --catalog server/.cache/catalog/treeoflife-10m/catalog.sqlite3 \
  --output server/.cache/training/china-species-20260923
```

训练完成会保存 `classifier.safetensors`、`model.json`、训练清单及
`model.metadata.json`，元数据包含准确的 10 类集合、稳定 ID、来源和限制。
之后刷新方案元数据索引即可查询；不添加为默认线上 Provider。

本次准备已完成并实际启动资源受限任务，已观察到特征提取进度；
截至本次记录，训练尚未完成，没有新的 Top-1 / Top-5 成绩。
范围筛选和迁移相关代码已通过 Ruff 与 395 项后端测试。
这不代表真实模型训练或微信端验收已经通过。
