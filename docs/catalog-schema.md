# 物种名录：基础信息与中文信息分离

2026-09-23 已迁移现有共享名录并启用新结构。这里的物种不是用户株记里的某一株；
未迁移私人株记、观察记录或用户昵称，也没有重新生成物种 ID。

## 主要表

### `species_core` — 与中文显示无关的物种基础信息

| 列 | 内容 |
| --- | --- |
| `id` | 原有稳定 ID，主键；禁止修改 ID |
| `scientific_name` | 学名，索引列 |
| `english_name` | 来源英文名，可为空字符串 |
| `taxonomy_json` | 原始界、门、纲、目、科、属信息 |
| `taxonomy_alternatives_json` | 不同来源分类路径，原样保留 |
| `growth_form` | 来源支持的形态代码或 `unknown` |
| `extra_json` | 其余语言无关信息，如原有来源、形态来源、形态集合 |

基础表不存中文名、中文俗称或中文简介。修改中文名称不改基础 ID、学名或分类路径。

### `species_zh` — 中文显示与中文知识

| 列 | 内容 |
| --- | --- |
| `species_id` | 主键，同时外键关联 `species_core.id` |
| `chinese_name` | 中文主名，可为 NULL |
| `aliases` | 中文俗称/别名 JSON 数组，不是用户给个体起的昵称 |
| `description` | 中文简介 |
| `knowledge_metadata` | 简介来源、审核状态、AI 标识等，保留原知识对象除 `summary` 的字段 |
| `name_origin` / `name_sources` | 名称来源及证据 |
| `taxonomy_chinese` | 原有中文分类名称映射 |
| `display_name` | 兼容原中文界面显示名 |
| `search_text` | 中文界面搜索索引，保留原学名/英文检索词 |
| `has_chinese_name` | 按中文主名计算的生成列，不能手工写入 |
| `is_visible` | 原有可见性策略：无中文主名时不能设为可见 |

未翻译的物种仍完整保存在基础表。本次为原有所有物种保留对应中文行，
缺失值保持 NULL / 空数组，不编造名字或简介。简介应同时保留 `knowledge_metadata`
中的来源和审核信息；既有知识对象迁移后完全一致。

## 分类名如何映射中文

- `taxa_core(id, rank, name, parent)`：现有目、科、属浏览节点及稳定父子关系。
- `taxa_zh(taxon_id, chinese_name, name_source)`：按同一个分类节点 ID 保存中文翻译。
- `membership(species_id, taxon_id)`：物种与分类节点关联，未改变。
- 上层界/门/纲及替代分类路径继续保存在物种完整分类 JSON 中，没有凭空补齐节点。

```sql
SELECT c.id, c.scientific_name, z.chinese_name, z.aliases, z.description
FROM species_core c
LEFT JOIN species_zh z ON z.species_id = c.id
WHERE c.id = 'bioclip:plant-001';

SELECT t.id, t.rank, t.name, z.chinese_name
FROM membership m
JOIN taxa_core t ON t.id = m.taxon_id
LEFT JOIN taxa_zh z ON z.taxon_id = t.id
WHERE m.species_id = 'bioclip:plant-001';
```

通过稳定 ID 关联，不用中文名称做唯一键，不根据相同别名合并物种。
中文名不等于“中国分布”或“中国常见”证据；训练范围仍需独立来源。

## 兼容现有程序

`species` 和 `taxa` 现在是兼容视图，不再混存两种信息：

- 旧查询仍得到原有字段，包括合并后的 `data` JSON。
- 旧导入/补名代码的 INSERT/UPDATE 通过视图触发器写入分离后的表。
- 新导入直接生成第二版结构，名录元数据增加 `storageSchemaVersion=2`。
- 搜索改用基础表与中文表的索引连接；识别标签索引显式按 `idx` 排序。
- 名称补全流程仍负责同步搜索索引和名录统计；不要绕过流程只改一个显示字段。

## 迁移及回滚

迁移程序：`python -m yizhu_server.catalog_schema --source 原库 --output 新文件`。
先用 SQLite backup 建副本，再迁移；不覆盖输入，已有输出路径会被拒绝。
逐条比较物种 JSON、分类节点、识别标签和关联的逻辑哈希，再做完整性/外键检查、
压缩副本后发布。空值、布尔值、名称证据、简介来源和替代分类路径均保留。

本次校验范围：

| 数据 | 数量 |
| --- | ---: |
| 物种 | 108,372 |
| 分类节点 | 12,253 |
| 识别标签 | 384,490 |
| 物种-分类关联 | 325,116 |

当前数据库：`server/.cache/catalog/treeoflife-10m/catalog.sqlite3`。
旧库备份：同目录 `catalog.before-locales-20260923.sqlite3`。
迁移/启用报告：`server/.artifacts/catalog-schema-20260923/`。
API 已按授权重启；本机 health、名录信息及银杏详情均 HTTP 200，
识别引擎状态为 `ready`。这不等于微信真机验证。

回滚前先保存迁移后修改并确认：停止 API，用备份建立一个新临时文件，
**原子替换**当前数据库，再启动 API。不要直接复制覆盖运行中的数据库，
尤其这些快照可能是硬链接。回滚会丢失迁移后的名录修改，不应自动执行。

名录变更后运行 `python -m yizhu_server.solution_metadata refresh` 更新方案元数据；
原模型候选集合不因本次存储拆分而改变。
本次已刷新，并核对迁移前全部方案的原始/应用集合数量不变。
以旧数据库文件 SHA-256 绑定、尚未应用的补名候选可能需要重新生成；
这不影响已应用名称、来源证据和稳定 ID。

验证：Ruff 通过，后端 **395 项测试通过（10.39 秒）**。
新库压缩后为 225,533,952 bytes；名录搜索空词、桂花、Ginkgo biloba 的本机测量
约为 0.057 / 0.077 / 0.076 秒，只是本轮本机检查，不是性能承诺。
