# 离线植物名称补全

更新：2026-09-21。来源别名已上线；按用户要求，**8,438 个物种已补中文名并设为可见**。
入口为 `server/src/yizhu_server/name_enrichment.py`，不调用网络或识别模型。
没有新增依赖、在线接口或客户端代码。

## 数据源与版本

### ScienceDB 原文件

已取得 DOI `10.57760/sciencedb.01711` 的 V1 原文件，许可元数据为 CC BY 4.0：

- 缓存：`server/.cache/research/plant-info-20260921/sciencedb-original.xlsx`
- 下载地址：`https://china.scidb.cn/download?fileId=62628f3cc2024a59763d5b70`
- 大小：1,408,491 字节，与发布方一致。
- MD5：`2119022b9aea4f0d014ff98fec0c2518`，与发布方一致。
- 本地 SHA-256：`f01d5f9f915cee8343a90321e122f8fedbadcd54b746bf21852937c714594341`。

导入器固定文件尺寸、SHA-256、表头、行数和 XML 大小限制；拒绝公式。
只读取 ZIP 内指定 XML，不解压路径或执行 Excel 内容。
来源保留作者、DOI、许可链接及字段提取说明。

实际原文件为 **14,710 行、14 列，2,909 行有别名**；并非上一轮期刊附件的
15 列、2,910 行别名。两文件有 6 条学名字符串不对应；
学名相同的条目中另有 4 条中文名／别名字段不同，不将二者混用。
使用当前原文件已解决许可元数据与下载快照不一致的问题；这不是对期刊附件授权的推断。

### Wikidata 公开快照

仅读取明确提供目录中的 `Q*.json`，记录各文件 SHA-256。
不下载完整数据库，也不读取浏览器、用户资料或私有手账。

仅提取未被标为 deprecated 的中文 `P1843` 俗名声明，不使用 labels/aliases
直接补名。本次仍是上一轮的 5 个物种和 2 个文献实体样本，没有全量覆盖声明。
目前可核查的引用链限定为缓存文献实体明确属于《中国植物志》系列；
其他来源声明标记 `unverified_reference`，不恢复 TaiCOL 或手工名称表。
这不是保证 Wikidata 内容正确：即使引用完整，名称仍为待审状态。
源码中的系列 ID 是文献来源规则，不是逐物种中文名映射。

## 运行

在仓库根目录执行，Python 环境沿用已配置的 `server/.venv`：

```sh
server/.venv/bin/python -m yizhu_server.name_enrichment \
  --catalog server/.cache/catalog/treeoflife-10m/catalog.sqlite3 \
  --sciencedb server/.cache/research/plant-info-20260921/sciencedb-original.xlsx \
  --wikidata-dir server/.cache/research/plant-info-20260921 \
  --output server/.cache/catalog/name-candidates-next.sqlite3
```

`--sciencedb` 与 `--wikidata-dir` 至少提供一个；`--output` 必须是尚不存在的新文件。
可单独只使用 ScienceDB。命令输出 JSON 统计；不需要运行额外审查脚本。
程序使用只读 SQLite 连接读取参考库，比对前后 SHA-256，临时生成后原子发布输出，
拒绝覆盖现有文件或符号链接。失败不发布半成品。

**不要把输出设为运行中 API 的目录库。** 它是另一种表结构的待审数据，
不是 `yizhu-catalog` 的替代物，也不包含模型向量、私人记录或可用的在线名录。

## 输出与规则

| 表 | 内容 |
| --- | --- |
| `metadata` | 构建时间、输入目录校验值、统计、用途 |
| `sources` | 来源版本、许可、署名、下载快照校验值 |
| `records` | 原始名称字段或 Wikidata 声明证据、行号／声明 ID |
| `candidates` | 名称、语言、类型、匹配 Species ID、现有中文名和可见性、风险标记 |

原始候选保留 `review_status='pending'`，不改写研究证据。
新增 `--apply-from` 按版本化来源规则生成可部署的目录副本；采纳依据写入物种
`nameSources.acceptancePolicy` 和目录元数据，不冒充人工审核。
候选 ID 由来源记录、名称类型、语言和名称生成；重复运行不产生新身份。
Species／Plant ID、已有中文名、原始识别结果和可见性均不变。

- 仅精确学名匹配；不做模糊拼写、中文反向合并或未经验证的异名连接。
- Wikidata 检查物种等级；ScienceDB 没有独立等级列，只接受双名形式进行初步匹配，
  并保留 `rank_inferred_from_binomial` 标记。杂交种、亚种等不自动关联。
- ScienceDB 别名按列表分隔符拆分；Wikidata 复合声明不自动拆解为已确认名称。
- 同名多种、与其他现有中文名冲突、非物种部位／制品词、限定声明、未核查文献、
  首选名差异和隐藏物种均标记。启发式标记并非完整语义判断。
- 没有中文名就不生成中文名；保留无名称的来源记录供统计。
- 不导入省份／地理分布、未经解释的生活型编码或整篇正文；不生成 `knowledge`，
  也不恢复生长形态筛选。简介补全是后续独立步骤。

## 本次实际生成

输出：`server/.cache/catalog/name-candidates-20260921-r3.sqlite3`。
前两次中间输出保留在忽略缓存中，不是线上目录。
可复核摘要：`docs/research/name-enrichment-run-20260921.json`。

| 指标 | 数量 |
| --- | ---: |
| ScienceDB 来源记录 | 14,710 |
| Wikidata 中文俗名声明 | 40 |
| 总名称候选（含来源重复，不是物种数） | 17,256 |
| 精确关联的名称候选 | 12,648 |
| 有名称候选且匹配的现有可见物种 | 1,294 |
| 有名称候选且匹配的现有隐藏物种 | 8,438 |
| 跨匹配物种共享的名称 | 93 |
| 无可用中文名称的来源记录 | 1,049 |

“桂花”的文献声明关联现有 `Osmanthus fragrans` ID，现已作为别名上线，
没有将首选名“木犀”替换，也没有宣称其全部其他俗名都可信。
统计不能解释为识别准确率，或可直接公开的新增物种数。

## 验证及下一步

本轮运行了 Ruff 和完整后端测试：173 项通过。
测试覆盖只读保护、拒绝覆盖、失败不发布、稳定候选 ID、同名歧义、
等级与文献限制、缺名不编造、公式及校验值拒绝。
实际产物 `PRAGMA integrity_check` 为 `ok`，输入目录 SHA-256 未变化。
未运行前端构建、真机检查、微信上传或服务重启。

以上为候选构建阶段的验证，后续应用与部署见下一节。

## 首批线上更新

应用逻辑在 `server/src/yizhu_server/catalog_names.py`，仍使用同一个命令入口：

```sh
server/.venv/bin/python -m yizhu_server.name_enrichment \
  --catalog server/.cache/catalog/treeoflife-10m/catalog.sqlite3 \
  --apply-from server/.cache/catalog/name-candidates-20260921-r3.sqlite3 \
  --output server/.cache/catalog/treeoflife-10m/catalog-with-aliases-next.sqlite3
```

策略 `visible-sourced-aliases-v1` 只更新已有全国名录首选名的可见物种。
ScienceDB 只采纳别名列，要求学名、科属一致且没有冲突标记；
Wikidata 只采纳有已核查文献引用、无额外限定和风险标记的声明。
不替换首选名、不恢复隐藏物种。写入别名、逐条来源和搜索字段，
拒绝过期候选及覆盖输出；重复应用不重复增加别名／来源。

实际更新 **196 个物种、241 个别名**。候选中当前可见物种仅有
242 种带 ScienceDB 别名，共 297 条；大部分其他候选是首选名或隐藏物种。
17,256 是名称候选条数，不是可新增物种数。

替换前已对比物种保护字段与 `labels`／`taxa`／`membership`，均未改变。
产物完整性检查通过；旧库备份为
`server/.cache/catalog/treeoflife-10m/catalog-before-aliases-20260921.sqlite3`，
新版同目录 `catalog-with-sourced-aliases-20260921.sqlite3` 原子替换线上路径。
这些目录快照只读；后续不能原地修改与备份共享的文件。
如需回退，用旧库生成新的部署副本，再原子替换线上路径并重启 API。

Ruff 与 **179 项后端测试通过**。已重启 API 清除缓存，本地及公网 health 均为
`ready`，方案仍为 `bioclip-v2-rerank`；两端木犀详情均返回别名“桂花”。
未改客户端、上传小程序或做真机验证。

该别名批次没有采纳的 8,438 个隐藏物种，现已通过下节的独立操作补名并公开。

## ScienceDB 补首选中文名并设为可见

用户明确要求将上述 8,438 个物种设为可见，已执行：

```sh
server/.venv/bin/python -m yizhu_server.name_enrichment \
  --catalog server/.cache/catalog/treeoflife-10m/catalog.sqlite3 \
  --reveal-sciencedb server/.cache/research/plant-info-20260921/sciencedb-original.xlsx \
  --expected-count 8438 \
  --output server/.cache/catalog/treeoflife-10m/catalog-sciencedb-visible-20260921.sqlite3
```

策略 `missing-primary-sciencedb-exact-v1`：

- 从已校验的 DOI 原文件读首选中文名，仅按双名学名精确关联；不猜测或翻译名称。
- 仅处理原本 `is_visible=0`、`has_chinese_name=0` 且 JSON 中缺名的物种。
- 填写 `chineseName`／`commonName`，将两项数据库标记及 JSON `isVisible` 同步更新。
- `chineseNameOrigin="sciencedb"`，逐条保留 DOI、行号、名称、许可、署名及采纳规则。
  不冒称全国名录来源。既有客户端未使用 origin 判别，仍能使用名称及来源字段；
  本轮未改客户端，后续维护前端类型定义时需把该 origin 值加入联合类型。
- 不改 ID、已有首选名／别名、私人手账、物种知识、分类树或向量标签。
  31 个条目与来源的科级分类不同，差异记录在来源字段，不擅自重分类；
  所有匹配条目的属名一致。
- `--expected-count` 是数量保护，不符则不发布输出。重跑不重复添加来源；
  已执行后期望新增数应为 0，不能继续使用 8438。

实际目标 ID 集合与原候选库的 8,438 个隐藏物种完全相同。
当前 **19,716 条可见、88,656 条隐藏**，全国名录原有 11,278 条及既有别名保持不变。
元数据计数、中文搜索、详情和分类浏览计数同步生效。

部署前备份：
`server/.cache/catalog/treeoflife-10m/catalog-before-sciencedb-visible-20260921.sqlite3`。
新库已原子替换线上路径，并重启 API。本地／公网名录均返回 19,716，
“澳洲凤尾蕨”搜索与详情均为 200，来源为 ScienceDB；两端模型状态均为 ready。
快照校验值与检查摘要见 `research/sciencedb-visibility-20260921.json`。

Ruff 与 **185 项后端测试通过**，实际 SQLite 完整性为 ok；
保护字段、标签和分类成员均通过前后对比。未进行微信上传或真机测试。

从头重建时：全国基库 → 生成新名称候选 → 应用别名 → ScienceDB 补首选名／可见性
→ 部署。不要用未补全的基库直接覆盖线上，也不要把旧候选强行应用到不同基库。

## 识别结果优先显示中文别名

2026-09-21：新增服务端 `species_presentation.journal_species`，
仅在返回手账快照的 `species` 时投影显示名称，不修改物种数据库。

- `commonName` 优先使用已采纳且有逐条来源的中文别名；没有时回退到
  `chineseName`，再回退到学名。多个别名按现有存储顺序稳定选择，
  这是显示偏好，不声称别名的普及度或植物分类权威性。
- 保留 `chineseName`、学名、ID、来源和可见性；公开名录详情与搜索仍返回名录名称。
- 隐藏物种不通过别名投影变为可见。没有采纳记录的别名不参与优先显示。
- 结果页标题、确认按钮、其他候选以及手账物种标签通过既有 `commonName` 字段生效。
  Plant 的私人昵称不变；已保存识别无需重新推理，重新加载手账即可更新显示。
- 当前库验证：木犀显示为“桂花”，荷花木兰显示为“荷花玉兰”；
  澳洲凤尾蕨没有已采纳别名，仍显示原中文名。

Ruff 与 **195 项后端测试通过**。隔离 Flask／合成账号测试覆盖历史记录、
已确认候选、账号隔离、原始结果及个体昵称不变、公开名录缓存不被改写。
线上物种库 SHA-256 未变，API 已重启，本地／公网 health 均为 ready。
未修改客户端、上传小程序、访问真实私人记录或做真机 UI 验证。
