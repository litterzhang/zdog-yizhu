# 植物参考名录与广域识别

更新：2026-09-21。

## 已实现的范围

- 使用原有 BioCLIP v1 图像模型，替换手写 24 项在线候选表。
- 官方向量共有 **384,490** 个生物标签。`Plantae` / `Viridiplantae` 的学名合并后，
  排除 **203** 条 `sp.` / `cf.` / `aff.` 等未定种标签，得到 **108,372 个植物参考条目**。
  这是参考标签覆盖，不是已接受植物物种总数，更不是“所有植物都能准确识别”。
- 同学名的不同分类路径保留为 `taxonomyAlternatives`，默认采用来源第一条路径；
  不擅自将旧名、错拼或同物异名宣称为最新接受名。名录仍可能包含这些来源问题。
- 现有 24 项的 `bioclip:plant-xxx` ID 保留；其他 Species ID 按来源命名空间与学名生成，
  不使用数组序号。**Plant 个体 ID 不变**，旧足迹不重新推理、不改写原始候选；无中文名的候选不再展示。

## 分类与形态

“草本、灌木、乔木、藤本”是生长形态信息，不是分类阶元。
客户端已移除生长形态筛选及相关提示，仅保留中文名/学名搜索与分类导航；详情仍可显示已有形态资料。
后端形态字段与 API 参数保留兼容，前端浏览和分页不再发送该参数。
浏览路径为 **目 → 科 → 属 → 种**，物种详情保留完整来源分类。
浏览节点只按展示层级建立，避免来源中的历史“纲”差异制造两个同名“目”。

中文名现由全国名录补齐：11,278 个植物参考条目、4,044 个浏览节点有中文名，
缺少中文名的 97,094 条记录标记为不可见，公开浏览/搜索仅展示 11,278 条。来源、精确匹配与限制见 `chinese-names.md`。
未匹配的条目明确保留学名，另保留来源英文名供搜索。
形态仅补入 `server/src/yizhu_server/data/growth-forms.json` 中 **5 个核实条目**，
其余标为未知。这些资料不是完整的形态清单，不根据名称猜测缺失属性。
银杏、绣球、龟背竹、酢浆草的形态事实取自 NC State Extension Plant Toolbox；
白车轴草取自 USDA NRCS 的 White Clover Plant Guide。具体出处逐条保存在 JSON。
不引入食用、药用或树龄保证。已另行保存 6 条带出处、标明 AI 辅助/未人工校对的短简介；
私人描述与物种简介严格区分，见 `national-names-and-descriptions.md`。

## 识别方式与不确定性

`CatalogRecognitionProvider` 使用本地图像向量与完整预计算文本向量做余弦相似度检索：

1. 在植物参考中取候选，并按 Species ID 去重。
2. 同时检查所有生物标签中的首选；非植物或未定种参考领先时返回 `uncertain`。
3. 极低像素变化的照片同样返回 `uncertain`。这是拍摄质量启发式，不是通用未知物种检测器。
4. `confidence` 始终为 null，余弦分数不是正确概率。
5. 原始结果保留模型 revision、名录 revision、评分方式和全生物首选反证。
6. 数据库隐藏条目不进入展示候选；原始首选被隐藏时仍返回不确定，不能将次选提升为成功。
   不删除向量或缩小原始模型排序范围。可见性规则见 `chinese-names.md`。

不确定结果也可展开参考候选，或在名录搜索后手动确认；确认仍同步关联株记。
模型未覆盖的植物、近缘种、园艺品种与特殊拍摄角度仍可能误判，不能据此安全处置植物。

## 公开 API 与私有手账

以下为不要求登录的公开参考数据，不返回用户或照片：

| GET | 内容 |
| --- | --- |
| `/api/catalog` | 名录版本、参考条目数、形态资料数、是否安装 |
| `/api/catalog/taxa?parent=…&growthForm=…&offset=0&limit=30` | 分页下一级分类；根节点为目 |
| `/api/catalog/species?q=…&parent=…&growthForm=…&offset=0&limit=30` | 中文名、学名、来源英文名搜索 |
| `/api/catalog/species/{id}` | 单个条目、分类、来源与形态 |

每页最多 50 条；搜索最多 80 字，SQL 参数绑定并转义 LIKE 通配符。
SQLite 只读打开，索引与有界内存缓存限制重复查询成本。
`/api/journal` 只附带当前账号记录实际引用的物种，不传输整个十万级名录。
私人照片、确认、建档等接口仍需可信登录与所有权校验。

## 安装与再现

依赖仍由原有 `recognition` extra 管理，没有增加模型训练或付费 API：

```sh
uv sync --project server --extra recognition --locked
```

将下列官方发布文件下载到 `server/.cache/catalog/treeoflife-10m/`。
固定来源为：
`https://huggingface.co/datasets/imageomics/TreeOfLife-10M/tree/91debffb7146c32c89d76feb1eb575b555e2ecc7/embeddings`

| 文件 | 字节数 | SHA-256 |
| --- | ---: | --- |
| `txt_emb_species.json` | 65,731,052 | `844e6fabc06cac072214d566b78f40825b154efa9479eb11285030ca038b2ece` |
| `txt_emb_species.npy` | 787,435,648 | `91ce02dff2433222e3138b8bf7eefa1dd74b30f4d406c16cd3301f66d65ab4ed` |

这两个文件总计约 814 MiB，不包含训练照片。另按 `chinese-names.md` 准备约 3.5 MiB 的全国 XLSX 文件；
校验并生成本地只读名录：

```sh
uv run --project server --extra recognition --locked yizhu-catalog
```

工具验证尺寸与 SHA-256，使用临时库构建、原子发布，拒绝覆盖已有库。
需要重建时指定新的 `--output`，验证后在维护窗口切换，不手工修改库。
`.cache/` 和派生数据库不提交 Git；更新来源版本时必须显式迁移旧物种引用。
离线实验 `yizhu-recognize` 默认使用完整参考，`--labels` 仅用于自定义小集合比较。

官方数据卡将汇编标记为 CC0，但明确指出原始图像/文本有各自条款；
不能将其解读为所有原始照片都可任意搬运。这里只使用发布的向量与分类事实，
保留来源，不分发训练照片。模型仍为此前固定的 MIT BioCLIP v1 权重；
不把软件许可当成所有上游素材的许可。

## 本机实测与限制

2026-09-21，CPU 两线程，真实模型、完整参考、每个案例重复两次：

- 初始化 **15.39 秒**，进程峰值 RSS **1,876 MiB**。
- 第一张冷向量页读取约 **5.29 秒**；随后单张约 **0.93–0.97 秒**。
- 原有银杏样例：首选银杏。
- 整朵向日葵（旧 24 项之外）：首选 `Helianthus annuus`，仍要求人工确认。
- 向日葵花盘微距：全生物首选为非植物，返回不确定；植物候选中向日葵第二。
- 纯白、固定随机噪声：都返回不确定。此小样本不能证明拒识率或识别准确率。

附加测试照片只在 `/tmp` 用于实验，不随客户端发布：
`File:Sunflower macro wide.jpg`（Muhammad Mahdi Karim，GFDL 1.2）；
`File:Sunflower (Helianthus annuus) D35 5966 01.jpg`（Adarsh Patel，CC BY-SA 4.0）。
文件页、作者与许可由 Wikimedia Commons 官方 `imageinfo/extmetadata` 返回；
它们的标注用于冒烟检查，不代替专业验证集。

后续：建立有可靠标签的本地验证集，再比较更大模型/多角度策略；补充有许可的中文名称、
形态数据和接受名/异名映射。不会为了填满分类树而让模型编造资料。
