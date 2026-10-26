# Pl@ntNet 的中文支持集合与微调新增集合

2026-09-23，从本地原版 Pl@ntNet ResNet50 的 1,081 类标签和当前数据库只读导出。
不是 V2 标签，也不是刚训练的六类实验头；未改数据库或切换线上模型。

## 两个集合

- **A：420 种**。原预训练输出可保守匹配到稳定物种 ID，且
  `species_zh.has_chinese_name=1 AND species_zh.is_visible=1`。
  完整文件：`pretrained-supported.csv`。
- **B：883 种**。当前 972 个优先候选中，有可用中文信息但不在 A 的新增训练目标。
  完整文件：`finetune-additions.csv`。不是已训练支持的物种。

972 个候选中，923 个具有可用中文信息；其中 40 个已在 A，不重复加入 B。
49 个缺少可用中文信息，列入 `excluded-candidates.csv`，没有删除候选或伪造中文名。
40 个重叠物种在 `candidate-already-supported.csv`。
按用户后续确认，这 40 种不进入新增训练／采集任务；A 仍作为已有能力保留。
B 的 883 种在完成实际训练前不进入方案支持元数据。

计划扩展方案应保留 **A ∪ B = 1,303 种**，不能只训练 B 然后宣称仍支持 A。
A 中有 380 种不在原 972 个候选中，保留这些能力需要训练/回放数据及独立评估；
只拷贝旧分类头参数或保留名字不证明其准确率被保留。
之前六类链路实验不满足这个新的完整物种范围。
当前方案已明确登记 A 的 420 种，查询：

```sh
server/.venv/bin/python -m yizhu_server.solution_metadata plants \
  --solution plantnet-300k-resnet50 --scope supported --limit 20
```

## 为什么不是仅检查表中有无行

当前语言拆分迁移为 108,372 个物种都保留了 `species_zh` 行，
其中许多 `chinese_name` 为 NULL。原预训练标签映射到这张表的物种有 843 种，
但只有 420 种具有中文名且可见。仅判断 JOIN 成功会重新引入没有中文名的植物。

匹配使用项目既有的学名保守匹配逻辑，保留原输出索引与标签 ID；
歧义或未核实异名不猜测。不把“有中文名”当作“中国常见”或准确率保证。
CSV 的名字来自数据库，按学名排序，不是常见度排名。

`sets.json` 包含完整集合、差集、来源文件/目录哈希和筛选规则。
已有导出不可覆盖；后续更新使用新目录：

```sh
server/.venv/bin/python -m yizhu_server.plantnet_plant_sets \
  --output docs/research/plantnet-plant-sets-NEXT
```
