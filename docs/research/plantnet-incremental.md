# 保留 A、训练 B 的 Pl@ntNet 增量分类头

2026-09-23。当前支持合同 A=420，新增目标 B=883；40 个重叠目标不重复训练。
这条实现替代“随机新建小分类头并丢弃原有类别”的实验方式，但不会自动部署。

## 模型结构

- 使用已校验的原版 Pl@ntNet ResNet50（原始 1,081 类，不是 V2）。
- 冻结全部图像骨干；使用原 resize256 / center-crop224 预处理及
  **未经 L2 归一化的 2,048 维原始特征**，保持原 FC 的输入含义。
- 从原 FC 按经核对的输出索引、源 ID、原学名提取 A 的 420 行权重与 bias；
  原参数不进入优化器。
- 只为准备好图片的 B 类训练新增行。训练 batch 中不能出现 A 标签。
  新类别和旧类别共同参与 softmax，训练数据是 B，A 的已有照片仅用于验证／测试。
- 验证集按已观察物种平均交叉熵早停；未观察到的 A 类不伪造验证成绩。
  参数选择不读取 test 结果。
- 合并分类头以 safetensors 保存。重载时再次逐项核对 A 的权重、bias 和输出索引。

**保留参数不等于保留排名或准确率。** B 的新 logits 仍可能把正确 A 类挤出 Top-5。
评估额外报告 A-only 排名基线、混合 A+B 后的 A 成绩、B 成绩，
以及实际有测试图片的 A 物种数量，不能把少数 A 测试推广到全部 420 种。
A-only 基线也不是原始完整 1,081 输出排名，报告中明确区分。

## 数据及命令

新增图片来自 `plantnet-china-additions-883`；A 的验证／测试照片复用此前公开缓存，
不恢复旧采集器、不下载重复的新类别，不读取用户私人照片。
固定快照合并时按稳定 ID 过滤并验证所有权无关的公开来源、许可、哈希与划分，
检查跨 A/B 重复。A 的旧训练照片可以保留在快照中，但不会进入新增参数训练。

从项目根目录执行：

```sh
server/.venv/bin/python -m yizhu_server.plantnet_incremental \
  --source server/.cache/training/plantnet-china-additions-883 \
  --retained-data server/.cache/training/china-everyday-972 \
  --data server/.cache/training/plantnet-china-incremental-snapshot-20260923

bash server/scripts/train-plantnet-cascade.sh \
  server/.cache/training/plantnet-china-incremental-snapshot-20260923 \
  server/.artifacts/models/plantnet-china-incremental-v1 \
  --allow-partial --incremental
```

当前快照：**1,835 张照片，420 个保留类别，11 个准备好的新增类别**，
请求目标仍为 1,303。只训练实际有足量数据的 11 个新增行，不宣称 B 全部完成。

真实任务：
`yizhu-plantnet-incremental-20260923.service`，已确认 active/running。
CPUQuota=100%、MemoryMax=1500M、MemorySwapMax=512M、Nice=10。
日志：`server/.artifacts/plantnet-incremental-20260923/run.log`。

## 元数据与验收边界

只有实际训练成功后，才生成该模型版本的 `model.json` 与 `model.metadata.json`，
标明 `retainedClassCount`、`trainedAdditionalClassCount`、原输出索引、
真实类表及 `retentionAccuracyGuaranteed=false`。
不会因为出现训练计划就把 883 种全部登记为支持。
现有 `plantnet-300k-resnet50` 的 A=420 支持合同不改写；
新的离线模型使用独立版本身份，也不等于线上已切换。

重用植物存在二分类与验证集阈值选择流程，测试中拦截、未覆盖和技术失败仍计为失败。
这是开发快照，不是独立中国现场验收集。完整 B 数据、A 保留验证、范围外拒识及
端到端 Top-5≥90% 仍需实证；当前不能判定目标达成。

实现通过 Ruff 与 433 项后端测试。合成测试覆盖仅更新 B、A 参数与 logits 不变、
错误索引拒绝、保存重载和完整评测；合成分数不是植物准确率。
部分骨干微调尚未执行，需要在完整数据验证后决定，不能根据小样本 test 调参。

## 本次真实训练与评估已完成

已确认任务 inactive/dead、Result=success、ExecMainStatus=0。
真实输出为 **431 类：420 个 A 类＋11 个完成训练的 B 类**，
而不是完整 1,303 类；模型重载已逐项验证 A 的原始参数。

| 测试范围 | 端到端 Top-1 | 端到端 Top-5 |
| --- | ---: | ---: |
| 已覆盖类别的 171 张照片 | 157/171（91.81%） | 170/171（99.42%） |
| 包含未覆盖 B 的全部 240 张照片 | 157/240（65.42%） | 170/240（70.83%） |

有 69 张未覆盖类别照片仍计为失败；按有测试照片的物种平均 Top-5 为 39.78%。
这里**没有验证全部 431 类**：A 的测试照片只有南瓜一种、6 张，
其 A-only 与加入 B 后的 Top-5 均为 6/6；不能外推到剩余 419 个 A 类。
B 的全部测试照片命中 164/234（70.09%），其中包含未覆盖类别。
本次与先前六类实验的图片/类别集合不同，不能直接当作同一基准上的涨跌。

固定验证阈值在原二分类 test 集上：植物召回 493/499（98.80%）、
非植物误放行 243/500（48.60%）。测试没有用于回调阈值，门控仍有明显不足。
独立中国现场验收尚未建立，**90% 的完整目标没有达成**。

模型及逐图结果：
`server/.artifacts/models/plantnet-china-incremental-v1/`。
已各取一张公开 A（南瓜）和 B（紫薇）原图重放完整两级推理，
候选 ID 顺序与缓存评估相同、二分类分数差小于 1e-6；
记录在 `server/.artifacts/plantnet-incremental-20260923/raw-photo-replay.json`。

训练完成后已刷新新离线版本的元数据：
`plantnet-resnet50-incremental-plantnet-china-incremental-v1`，实际集合 431 种。
原 `plantnet-300k-resnet50` 的明确支持集合仍为 420，内容 revision 未改变。
没有把另外 872 个未训练 B 类登记为支持，没有自动切换线上识别方案。
