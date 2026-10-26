# 识别方案元数据与植物集合

每个已注册方案必须声明 `plant_set_source`，识别结果的 `solution.metadataId`
指向对应元数据 ID。元数据从实际名录、输出索引表或训练分类表生成，
不手写“支持所有植物”，也不把某个名称存在于名录当作模型已支持它。

## 数据结构

每份方案快照包含：

- `id`、`revision`、`task`、`serverSelectable`：身份、内容版本、任务类型和能否由服务器配置选择。
- `plantSet`：模型输出标签/配置参考集合，含数量、单位、集合文件和 SHA-256。
- `appPlantSet`：其中能映射到名录且当前可见的唯一植物 ID 集合。
- `supportedPlantSet`：方案明确登记的支持集合（当前 Pl@ntNet ResNet50 已登记 A 的 420 种）。
  与原始 1,081 类输出区分；新增训练计划不会自动进入这个集合。
- `source`、`stages`：来源版本、标签/权重指纹、召回/重排/回退阶段的范围。
- `artifact`：已有模型文件或资产状态；元数据存在不意味着权重完整或方案正在运行。
- `accuracyGuarantee=false`：范围是候选资格，不是准确率、食用安全或个体识别保证。

集合成员保存稳定 ID、学名、现有中文名/别名、名录 ID、可见性；
固定分类器还保留**原始类别索引、来源物种 ID 和原始学名**。
中文名变化不会重排分类器输出；无法映射的来源标签保留，不强行合并异名。

`plantSet.kind` 有三种含义：

| 值 | 含义 |
| --- | --- |
| `finite` | 有明确的标签/参考成员列表，成员在独立文件中 |
| `unknown` | 必需标签未安装，数量为 `null`，不是“支持零种” |
| `not-applicable` | 植物存在二分类只输出有/无植物；物种集合为 `null`，不是“支持全部物种” |

固定模型的集合单位为 `source-output-class`，类别数不一定等于去除异名后的独立物种数。
应用集合是可见性/映射筛选结果，不保证每张图片都能通过识别门控。

## 文件与查询

默认输出：`server/.cache/solution-metadata/`（生成产物，不提交大量名录到 Git）。

```text
index.json
solutions/<solution-id>/<revision>.json
sets/<content-sha256>.json
```

共享集合按内容去重；刷新只更新索引，旧版本快照保留，不覆盖历史集合。
这些是本地 Python/CLI 接口，本轮没有新增 HTTP API 或修改客户端。

从项目根目录运行：

```sh
# 安装/修改名录、中文名/可见性、输出标签或模型后刷新
server/.venv/bin/python -m yizhu_server.solution_metadata refresh

server/.venv/bin/python -m yizhu_server.solution_metadata list

server/.venv/bin/python -m yizhu_server.solution_metadata show \
  --solution bioclip-v2-rerank

server/.venv/bin/python -m yizhu_server.solution_metadata plants \
  --solution bioclip-v2-rerank --scope app --q 桂花 --limit 20

server/.venv/bin/python -m yizhu_server.solution_metadata plants \
  --solution plantnet-300k-resnet50 --scope model --offset 0 --limit 20

# 查询明确登记的中文支持范围（420 种，不含待训练 B）
server/.venv/bin/python -m yizhu_server.solution_metadata plants \
  --solution plantnet-300k-resnet50 --scope supported --offset 0 --limit 20
```

Python：`get_solution("bioclip-v1").metadata()`，或使用
`MetadataStore().get(id)` / `.plants(id, scope="app", q="银杏")`。
默认读取最近一次导出快照，**不是实时数据库视图**；名录修改后需执行 `refresh`。
读取时检查快照版本和集合哈希，分页上限 100，不在每次识别响应中塞入整个植物集合。

## 训练后自动维护

植物存在分类头和自训物种分类头保存后，自动生成同目录的
`<原文件名去掉.json>.metadata.json`。既有模型首次刷新时补齐旁路元数据：

- 二分类头：记录骨干、数据摘要、二类标签，明确无物种输出集合。
- 物种分类头：从真实 `classes` 按训练顺序生成精确成员集合。

旁路元数据不改变原始头文件、权重或其 SHA-256；模型变化时应保存新版本。
全方案索引仍通过 `refresh` 发布更新。
**单独修改元数据不能让模型学会新的植物。**

### Pl@ntNet 当前支持与新增训练计划

当前支持登记为 A 的 **420 种**。来源合同文件：
`server/.cache/solution-metadata/contracts/plantnet-300k-resnet50.json`。
它绑定实际预训练权重和标签文件，每个成员保留原输出索引与来源 ID。
首次登记使用 `initialize-plantnet-support`；已有合同不可静默覆盖。

B 的 **883 种**只是新增训练计划，不写入 `supportedPlantSet`。
40 个与 A 重叠的候选已从 B 排除，不作为新增类别或新增采集任务。
独立实验头（例如六类实验）的出现也不会修改当前方案的支持合同。
后续扩展必须来自完成训练且确实包含这些输出类别的实际模型产物，
并确认原 A 能力仍由该方案承接，才能登记相应模型版本；不能只增加元数据数字。

`refresh` 可更新派生索引，但不会因为计划文件增加物种就扩大这个支持集合。
合同校验会拒绝没有对应原模型输出的成员；新的微调模型需自己的模型身份和支持合同，
不能把新物种塞入不可能输出它们的原始预训练权重合同。

## 2026-09-23 实际导出

已生成 12 个方案的元数据，未训练模型、修改数据库、切换方案或重启服务。

| 方案 | 原始标签/参考成员 | 当前应用可用植物 |
| --- | ---: | ---: |
| BioCLIP v1 / v2-rerank / v2-whole（分别有元数据） | 各 108,372 | 各 19,716 |
| 多视图 BioCLIP + PlantCLEF | 110,090 | 19,716 |
| PlantCLEF 2024 原分类头 | 7,806 类 | 2,427 |
| Pl@ntNet SqueezeNet / ResNet50（分别有元数据） | 各 1,081 类 | 各 420 |
| BioCLIP 默认自定义标签探针 | 24 | 23 |
| 既有 BioCLIP 2 自训小分类头 | 3 | 3 |
| 三个已训练植物存在分类头 | 不适用 | 不适用 |

重排方案沿用召回集合，不能凭 BioCLIP 2 的骨干名称声称范围更大。
级联范围按已知阶段合并、通过名录 ID 去重；未知阶段不会伪装成完整集合。
PlantCLEF 类别表的公开固定哈希已校验，**准备好的权重仍缺失**，
因此其标签范围已知不代表该回退阶段可以启用。
当前导出不重新校验全部大型权重；权重可用性仍由模型加载器负责。

验证：Ruff 通过，后端 **379 项测试通过（7.16 秒）**；
实际读取并校验了全部 12 个方案的快照版本、两种范围的集合哈希/数量，
确认三个既有二分类头的 SHA-256 未变。真实查询 Pl@ntNet 的 `Ginkgo` 返回 0，
BioCLIP 应用集合可通过中文别名查询木犀；未执行新的识别推理或上线操作。
