# 53 条名称：百度百科浏览器核对

核对日期：2026-09-24。通过独立 Chromium 窗口读取页面及分类表；不是仅用搜索摘要。

**后续已确认只做物种级。** 下文分组仅描述当时查到的来源资料，不要求产品管理品种。
最新物种 ID 关联见 `landscape-plants-db-mapping-20260924.json`，
范围约定见 `docs/species-level-scope.md`。

**只更新资料文件，不修改数据库、训练标签、线上模型或支持清单。** 原始拼写和原始顺序均保留。

## 结果概览

| 结果 | 原始名称数量 |
| --- | ---: |
| 物种／杂交植物参考 | 18 |
| 品种／变种／园艺类群参考 | 20 |
| 名称或范围待确认 | 14 |
| 未取得可用条目 | 1 |

合计 53 条名称，不是 53 个不同物种，也不是新增模型支持数。
38 条取得可供后续核对的植物或园艺参考；另外 14 条仍有名称、范围或来源冲突，1 条未取得可用条目。
41 条的来源学名或明确标注的母种／候选学名可查到现有数据库记录；这些只是建议关联，不改变原来 97 条精确中文名称匹配的统计。

## 阅读方式

- “参考名称”和“来源学名”反映百度页面，不是已核准的接受名。
- 科、属、目等在 JSON / CSV 中按页面报告保留；不同条目可能采用旧体系，也可能互相矛盾。
- 栽培品种、变种、杂交种、园艺类群与母种分别标注，不能按每个中文名称新增一个物种。
- 大叶栀子花本页未提供科属；母种栀子的分类放在 `relatedPlantReferences`，不伪称为本页原始字段。
- 简介是自行概括的短摘要，见 CSV / JSON；不搬运图片、全文或食用药用建议。未认定百度百科内容具有开放再利用许可。
- 尚未获得页面修订号；来源保存实际访问时间、规范化页面链接及本地证据 SHA-256。

## 逐项结果

### 物种／杂交植物参考

| 原始名称 | 参考名称 | 来源学名 | 科／属（来源原文） | 核对说明 |
| --- | --- | --- | --- | --- |
| 毛鹃 | [锦绣杜鹃](https://baike.baidu.com/item/%E9%94%A6%E7%BB%A3%E6%9D%9C%E9%B9%83/807899) | Rhododendron × pulchrum Sweet | 杜鹃花科 / 杜鹃花属 | 默认页是人物。实际展开同名词条并选杜鹃花科植物后跳到锦绣杜鹃；原人物资料不采用。保留杂交标记。 |
| 紫鹃 | [锦绣杜鹃](https://baike.baidu.com/item/%E9%94%A6%E7%BB%A3%E6%9D%9C%E9%B9%83/807899) | Rhododendron × pulchrum Sweet | 杜鹃花科 / 杜鹃花属 | 默认是《红楼梦》人物；菜单另有铁线莲新品种。按原清单与毛鹃、春鹃、夏鹃相邻的语境选择杜鹃义项，不代表所有“紫鹃”同名用法都等同于它。 |
| 金叶女贞 | [金叶女贞](https://baike.baidu.com/item/%E9%87%91%E5%8F%B6%E5%A5%B3%E8%B4%9E) | Ligustrum × vicaryi Rehder | 木樨科 / 女贞属 | 来源学名含 ×，并说明杂交育成；不删除杂交标记或当作任意女贞的别名。 |
| 熊掌木 | [熊掌木](https://baike.baidu.com/item/%E7%86%8A%E6%8E%8C%E6%9C%A8) | Fatshedera lizei (Hort. ex Cochet) Guillaumin | 五加科 / 五角金盘属 | 按杂交植物参考处理。原页面生长温度等文字存在明显异常，本次不采纳相关栽培数值。 |
| 瓜子黄杨 | [黄杨](https://baike.baidu.com/item/%E9%BB%84%E6%9D%A8/3657) | Buxus sinica (Rehder & E. H. Wilson) M. Cheng | 黄杨科 / 黄杨属 | 基本信息与概述均明确列出瓜子黄杨这一别名。仍只作来源参考，未新增数据库别名。 |
| 滨海木槿 | [海滨木槿](https://baike.baidu.com/item/%E6%B5%B7%E6%BB%A8%E6%9C%A8%E6%A7%BF/4417251) | Hibiscus hamabo Sieb. & Zucc. | 锦葵科 / 木槿属 | 记录百度采用的名称“海滨木槿”；保留原始输入，不直接改写数据库名称。 |
| 构骨 | [枸骨](https://baike.baidu.com/item/%E6%9E%B8%E9%AA%A8/2776280) | Ilex cornuta Lindl. & Paxton | 冬青科 / 冬青属 | 页面用字为“枸骨”，原文“构骨”保留在 input_name；该对应尚未作为纠错别名写入数据库。 |
| 黄金条 | [金钟花](https://baike.baidu.com/item/%E9%BB%84%E9%87%91%E6%9D%A1/4433404) | Forsythia viridissima Lindl. | 木犀科 / 连翘属 | 页面标题为黄金条，中文名字段为金钟花，别名字段明确包含黄金条；保留来源的旧目名，不自动写回分类。 |
| 月季 | [月季花](https://baike.baidu.com/item/%E6%9C%88%E5%AD%A3%E8%8A%B1/14505544) | Rosa chinensis Jacq. | 蔷薇科 / 蔷薇属 | 百度提供 Rosa chinensis 的种级参考，但不能据此把所有现代杂交月季品种都并入该种。 |
| 金银花 | [忍冬](https://baike.baidu.com/item/%E5%BF%8D%E5%86%AC/758885) | LonicerajaponicaThunb. | 忍冬科 / 忍冬属 | 保留基本信息中连写的拉丁名原文。数据库检索使用概述中可核对的 Lonicera japonica，不把药材名称推广到其他种。 |
| 丁香 | [紫丁香](https://baike.baidu.com/item/%E7%B4%AB%E4%B8%81%E9%A6%99/1076778) | Syringa oblata Lindl. | 木犀科 / 丁香属 | 条目明确将丁香列为别名，按原园林清单采用紫丁香参考；不据此把香料丁香等同名植物合并。 |
| 五针松 | [五针松](https://baike.baidu.com/item/%E4%BA%94%E9%92%88%E6%9D%BE) | Pinus parviflora | 松科 / 松属 | 基本信息明确给出 Pinus parviflora；这一条目的范围不是所有具有五针一束特征的松树。 |
| 红橘 | [红橘](https://baike.baidu.com/item/%E7%BA%A2%E6%A9%98/5252312) | Citrus reticulata Blanco | 芸香科 Rutaceae / 柑橘属 Citrus | 已从多义词页排除小说角色。拉丁名来自条目概述；这是物种范围参考，不代表所有红橘地方品种可在图像上区分。 |
| 青冈栎 | [青冈](https://baike.baidu.com/item/%E9%9D%92%E5%86%88/71159) | Quercus glauca Thunb. | 壳斗科 / 栎属 | 百度采用栎属 Quercus glauca；与数据库或旧名录中的属级异名是否同一分类实体，须另行核准。 |
| 桔子树 | [柑橘](https://baike.baidu.com/item/%E6%9F%91%E6%A9%98/55204) | Citrus reticulata Blanco | 芸香科（Rutaceae） / 柑橘属（Citrus） | 只作为该来源的物种候选；日常桔子树称呼不能唯一确定具体栽培品种。 |
| 青桐 | [梧桐](https://baike.baidu.com/item/%E6%A2%A7%E6%A1%90/6199) | Firmiana simplex (L.) W. Wight | 锦葵科 / 梧桐属 | 基本信息为锦葵科，概述仍出现梧桐科旧名称；分开保留，不把它与悬铃木类的“法国梧桐”混为一谈。 |
| 檫树 | [檫木](https://baike.baidu.com/item/%E6%AA%AB%E6%9C%A8/2039142) | Sassafras tzumu (Hemsl.) Hemsl. | 樟科 / 檫木属 | 原园林语境与檫木候选相符；保留先前维基的其他同名用法，不视为跨来源同名歧义已彻底解决。 |
| 慈孝竹 | [孝顺竹](https://baike.baidu.com/item/%E5%AD%9D%E9%A1%BA%E7%AB%B9/7784080) | Bambusa multiplex (Lour.) Raeusch. ex Schult. & Schult. f. | 禾本科 / 簕竹属 | 保留 Bambusa multiplex 的来源参考；与名单内其他竹类是否重复，应按稳定物种ID审核，不按名称数量计数。 |

### 品种／变种／园艺类群参考

| 原始名称 | 参考名称 | 来源学名 | 科／属（来源原文） | 核对说明 |
| --- | --- | --- | --- | --- |
| 四季桂 | [四季桂](https://baike.baidu.com/item/%E5%9B%9B%E5%AD%A3%E6%A1%82) | Osmanthus fragrans 'Semperflorens' | 木犀科 / 木犀属 | 条目以 'Semperflorens' 标为品种；不能把它计为独立物种。 |
| 花叶蔓长春 | [花叶蔓长春](https://baike.baidu.com/item/%E8%8A%B1%E5%8F%B6%E8%94%93%E9%95%BF%E6%98%A5) | Vinca major Linn. cv. Variegata Loud | 夹竹桃科 / 蔓长春花属 | 条目学名带 cv. Variegata，按蔓长春花的栽培品种参考处理，不误作常春藤属植物。 |
| 大叶栀子花 | [牡丹栀子](https://baike.baidu.com/item/%E7%89%A1%E4%B8%B9%E6%A0%80%E5%AD%90/12604037) | — | — | 原页别名为大叶重瓣栀子、大叶栀子，但未给拉丁学名和科属。栀子母种仅在 relatedPlantReferences 补充，不能把母种学名冒充品种学名。 |
| 金边黄杨 | [金边黄杨](https://baike.baidu.com/item/%E9%87%91%E8%BE%B9%E9%BB%84%E6%9D%A8) | Euonymus japonicus 'Aurea-marginatus' | 卫矛科 / 卫矛属 | 属卫矛科卫矛属，尽管名称含黄杨，也不能并入黄杨科黄杨属。 |
| 龟甲冬青 | [龟甲冬青](https://baike.baidu.com/item/%E9%BE%9F%E7%94%B2%E5%86%AC%E9%9D%92) | Ilex crenata Thunb. | 冬青科 / 冬青属 | 正文称齿叶冬青的变型，信息表列品种，但学名只给 Ilex crenata。该学名只可作母种候选，不能说唯一确定了园艺型。 |
| 洒金桃叶珊瑚 | [花叶青木](https://baike.baidu.com/item/%E8%8A%B1%E5%8F%B6%E9%9D%92%E6%9C%A8/9376317) | Aucuba japonica var. variegata Dombrain | 丝缨花科 / 桃叶珊瑚属 | 来源采用 Aucuba japonica var. variegata，并列为青木的变种；不将园艺型独立计为物种。 |
| 红花绣线菊 | [红花绣线菊](https://baike.baidu.com/item/%E7%BA%A2%E8%8A%B1%E7%BB%A3%E7%BA%BF%E8%8F%8A) | Spiraea japonica“Dart's Red” | 蔷薇科 / 绣线菊属 | 拉丁名来自正文概述而非基本信息；来源指向 'Dart's Red'，不推广为所有红花绣线菊园艺材料。 |
| 金焰绣线菊 | [金焰绣线菊](https://baike.baidu.com/item/%E9%87%91%E7%84%B0%E7%BB%A3%E7%BA%BF%E8%8F%8A) | Spiraea japonica 'Goldflame' | 蔷薇科 / 绣线菊属 | 页面明确给出 'Goldflame' 及母种粉花绣线菊；母种候选不是该品种已被单独收录的证明。 |
| 羽毛枫 | [羽毛槭](https://baike.baidu.com/item/%E7%BE%BD%E6%AF%9B%E6%A7%AD/10330447) | Acer palmatum var. dissectum (Thunb.) Miq. | 无患子科 / 槭属 | 基本信息明确列羽毛枫为别名，来源按鸡爪槭变种处理；不是额外的独立物种。 |
| 素心腊梅 | [素心蜡梅](https://baike.baidu.com/item/%E7%B4%A0%E5%BF%83%E8%9C%A1%E6%A2%85) | Chimonanthus praecox 'Luteus'、Chimonanthus praecox var. concolor | 蜡梅科 / 蜡梅属 | 原查询跳到蜡梅母种；另查“素心蜡梅”。页面并列 'Luteus' 与 var. concolor，且目级分类与母种条目不同，保留冲突不强选接受名。 |
| 丰花月季 | [丰花月季](https://baike.baidu.com/item/%E4%B8%B0%E8%8A%B1%E6%9C%88%E5%AD%A3) | Rosa hybrida E.H.L.Krause | 蔷薇科 / 蔷薇属 | 按园艺类群参考保留页面的 Rosa hybrida，不把丰花类群直接认作一个已核准的野生物种。 |
| 银边黄杨 | [银边黄杨](https://baike.baidu.com/item/%E9%93%B6%E8%BE%B9%E9%BB%84%E6%9D%A8) | Euonymus Japonicus var.alba-marginata T.Moore | 卫矛科 / 卫矛属 | 学名大小写、空格按信息表保留；正文给出较规范的 var. alba-marginata 写法。仍属于卫矛属而非黄杨属。 |
| 紫叶酢浆草 | [紫叶酢浆草](https://baike.baidu.com/item/%E7%B4%AB%E5%8F%B6%E9%85%A2%E6%B5%86%E8%8D%89) | Oxalis triangularis 'Mijke' | 酢浆草科 / 酢浆草属 | 本次百度页指定 'Mijke' 栽培品种；不能据此认定所有同名紫叶酢浆草均属于此品种，维基原歧义记录保留。 |
| 银纹沿阶草 | [银边麦冬](https://baike.baidu.com/item/%E9%93%B6%E8%BE%B9%E9%BA%A6%E5%86%AC/11030244) | Ophiopogon jaburan Argenteivittatus | 百合科 / 沿阶草属 | 别名字段明确包含银纹沿阶草；来源采用百合科旧分类。学名指 Ophiopogon jaburan 的栽培品种，不按中文“麦冬”自动连到其他种。 |
| 花叶芦竹 | [花叶芦竹](https://baike.baidu.com/item/%E8%8A%B1%E5%8F%B6%E8%8A%A6%E7%AB%B9) | Arundo donax 'Versicolor' | 禾本科 / 芦竹属 | 页面明确标 'Versicolor'；数据库中的芦竹母种与这一栽培品种分开标注。 |
| 金桂 | [金桂](https://baike.baidu.com/item/%E9%87%91%E6%A1%82/4230383) | Osmanthus fragrans var. thunbergii | 木樨科 / 木樨属 | 已在多义词页选择木犀科植物义项；原文保留 var. thunbergii，尚未核准它与园艺金桂品种群的边界。 |
| 中山杉 | [中山杉](https://baike.baidu.com/item/%E4%B8%AD%E5%B1%B1%E6%9D%89) | Taxodium 'Zhongshanshan' | 柏科 / 落羽杉属 | 页面学名为属名加 'Zhongshanshan'，种字段却填落羽杉属；仅作为园艺名称，不能据此生成独立种级分类。 |
| 碧桃 | [碧桃](https://baike.baidu.com/item/%E7%A2%A7%E6%A1%83/502277) | Prunus persica 'Duplex' | 蔷薇科 / 李属 | 默认是景天科多肉；实际展开14个同名词条后选蔷薇科李属植物。采用 Prunus persica 'Duplex'，拒绝混入 Echeveria 'Peach Pride'。 |
| 寿星桃 | [盆栽寿星桃](https://baike.baidu.com/item/%E7%9B%86%E6%A0%BD%E5%AF%BF%E6%98%9F%E6%A1%83/132830) | Prunus persica f.densa Mak. | 蔷薇科 / 李属 | 来源给 f. densa，却在信息表写变种，且目字段与蔷薇科归属不协调；保留来源原文，不作为核准分类。 |
| 花梅 | [花梅](https://baike.baidu.com/item/%E8%8A%B1%E6%A2%85/4056462) | Armeniaca mume | 蔷薇科 / 杏属 | 拉丁名 Armeniaca mume 来自概述，不把整组栽培品种当作新增独立物种；未自动转换为其他属名。 |

### 名称或范围待确认

| 原始名称 | 参考名称 | 来源学名 | 科／属（来源原文） | 核对说明 |
| --- | --- | --- | --- | --- |
| 苦楮 | [苦郎树](https://baike.baidu.com/item/%E8%8B%A6%E9%83%8E%E6%A0%91/1683512) | Volkameria inermis L. | 唇形科 / 苦郎树属 | 保留原文“苦楮”。百度跳转为苦郎树；“苦槠”只是结合园林清单提出的字形近似候选，不能自动纠错或并种。 |
| 春鹃 | [春鹃](https://baike.baidu.com/item/%E6%98%A5%E9%B9%83) | RhododendronsimsiiPlanch. | 杜鹃花科 / 杜鹃花属 | 正文按花期解释“春鹃”，基本信息却填入杜鹃的种名且拉丁名缺空格；不能将整个园艺花期类别唯一映射为该种。 |
| 夏鹃 | [夏鹃](https://baike.baidu.com/item/%E5%A4%8F%E9%B9%83) | Rhododendron pulchrum | 杜鹃花科 / 杜鹃花属 | 基本信息写 Rhododendron pulchrum，正文同时写 R. simsii，并将杜鹃按花期分组；种级归属存在冲突。 |
| 常春藤 | [常春藤](https://baike.baidu.com/item/%E5%B8%B8%E6%98%A5%E8%97%A4/70130) | Hedera nepalensis var. sinensis (Tobler) Rehder | 五加科 / 常春藤属 | 百度采用 Hedera nepalensis var. sinensis；先前维基还涉及 H. helix 的同名使用。保留百度候选，但不据默认页排除其他常春藤。 |
| 草本绣球 | [绣球](https://baike.baidu.com/item/%E7%BB%A3%E7%90%83/4639080) | Hydrangeamacrophylla (Thunb.) Ser | 绣球科 / 绣球属 | “草本”与页面的灌木描述不一致，基本信息也未列此原始名称；跳转仅是候选，不能据此认可“草本绣球”为种名。 |
| 小叶栀子花 | [栀子](https://baike.baidu.com/item/%E6%A0%80%E5%AD%90/82358) | Gardenia jasminoides J.Ellis | 茜草科 / 栀子属 | 支持栀子作为候选，但不能据省略“花”字的查询判断原清单是否特指矮生或小叶栽培品种。 |
| 豆瓣黄杨 | [黄杨](https://baike.baidu.com/item/%E9%BB%84%E6%9D%A8/3657) | Buxus sinica (Rehder & E. H. Wilson) M. Cheng | 黄杨科 / 黄杨属 | 已读取的基本信息和概述未明确列“豆瓣黄杨”为别名；不能仅凭跳转断言与瓜子黄杨完全同义。 |
| 木香 | [木香花](https://baike.baidu.com/item/%E6%9C%A8%E9%A6%99%E8%8A%B1/7114637) | Rosa banksiae Aiton | 蔷薇科（Rosaceae） / 蔷薇属（Rosa） | 木香总条目明确涉及多个科。按园林语境补读其链接中的木香花，但原清单缺照片和学名，仍保留同名歧义。 |
| 红帽子月季 | [月季](https://baike.baidu.com/item/%E7%BA%A2%E5%B8%BD%E5%AD%90%E6%9C%88%E5%AD%A3) | Rosa chinensis Jacq. var. chinensis | 蔷薇科 Rosaceae / 蔷薇属 Rosa | 标题为红帽子月季，中文学名却填月季，正文又称其为月季花别名；不足以确认特定“红帽子”栽培品种。 |
| 白玉兰 | [白玉兰](https://baike.baidu.com/item/%E7%99%BD%E7%8E%89%E5%85%B0/867) | Michelia × alba DC.；Yulania denudata (Desrousseaux) D. L. Fu | 木兰科 / 含笑属（白兰）、玉兰属（玉兰） | 来源明确包含两个植物，且标题、正文与目字段存在不同分类表述；不能只选其中一个学名覆盖原名。 |
| 花桃 | [山桃](https://baike.baidu.com/item/%E5%B1%B1%E6%A1%83/2065274) | Prunus davidiana (Carrière) Franch. | 蔷薇科（Rosaceae） / 李属（Prunus） | 基本信息别名未列花桃，原园林清单也可能泛指观赏桃；不能仅凭跳转把花桃确定为山桃。 |
| 金枝垂柳 | [金丝垂柳](https://baike.baidu.com/item/%E9%87%91%E4%B8%9D%E5%9E%82%E6%9F%B3) | Salix × aureo-pendula | 杨柳科 / 柳属 | 两篇补查条目均不能证明与原名完全同义。只提供候选及其分类，禁止自动改名或并种。 |
| 悬铃木 | [二球悬铃木](https://baike.baidu.com/item/%E4%BA%8C%E7%90%83%E6%82%AC%E9%93%83%E6%9C%A8/1414455) | Platanus × acerifolia (Aiton) Willd. | 悬铃木科 / 悬铃木属 | 维基阶段取得的是属级范围；百度跳到具体杂交种不足以排除一球、三球等其他悬铃木。 |
| 花石榴 | [月季石榴](https://baike.baidu.com/item/%E6%9C%88%E5%AD%A3%E7%9F%B3%E6%A6%B4/1714912) | Punica granatum 'Nana' Pers. | 石榴科 / 石榴属 | 页面把花石榴列为 'Nana' 的别名，但同页又列多个不同观花品种；不能认定原清单专指 'Nana'。 |

### 未取得可用条目

| 原始名称 | 参考名称 | 来源学名 | 科／属（来源原文） | 核对说明 |
| --- | --- | --- | --- | --- |
| 哺鸡竹 | — | — | — | 精确名称及补查“哺鸡毛竹”均返回百科首页，未取得可用条目。只记本次未查到，不能断言植物不存在，亦不猜填学名。 |

## 保存位置

- 主文件：`landscape-plants-200-zh-check.csv` / `.json`，仅在原记录后新增 `baidu_*` / `baidu` 字段。
- 快照备份：`landscape-plants-200-zh-check.before-baidu-20260924.csv` / `.json`。
- 浏览器证据：`server/.cache/research/landscape-baidu-20260924/`。
- 人工核对标注、只读合并脚本与校验结果：`server/.artifacts/landscape-baidu-20260924/`。

“哺鸡竹”与补查“哺鸡毛竹”返回百科首页，不能据此断言植物或资料不存在。

后续入库前应优先处理名称和范围歧义，再使用稳定物种 ID 建立有来源的中文别名关系；本轮未接受任何新映射。
