# 249种目标：图片缺口分类（2026-09-25）

后续更新：下述14个来源名称问题中已有12个通过具名杂交来源核对，并开始在独立池补采；详见 `docs/tasks/named-hybrid-source-recovery-20260925.md`。此表保留原主池快照，不将尚未合并的补充图片直接抵扣配额。

这是数据配额盘点，不是模型支持清单。主采集已完成，补充来源仍在运行。

主数据池：14,065张、218种有图、109种达到70/15/15启动配额。仍有140种未达配额。

| 缺口类型 | 物种数 | 下一步 |
|---|---:|---|
| 来源学名匹配未解决 | 14 | 对照官方分类资料核对来源名称及杂交符号，不模糊匹配或随便改物种ID |
| 来源物种已匹配，但本轮未采到合格图片 | 17 | 定向查其他授权来源；不能把本轮未采到合格CC0/CC BY图片说成植物不存在 |
| 有图片但划分配额不足 | 109 | 补充不同作者/观察的原图，保留原验证和测试划分 |

来源匹配未解决指iNaturalist来源名称，不是数据库249个稳定ID失效。补充图片仍为未分配候选，不用其数量直接抵扣训练/验证/测试缺口。

## 优先项：桂花

原主数据仍为19训练、2验证、3测试，共24张；离启动配额还缺51/13/12张。补充来源的6张候选仍需标签和跨来源划分核验，尚未转成训练图片。

## 14个来源名称待核对项

- 丽格秋海棠 — `Begonia hiemalis`
- 球根秋海棠 — `Begonia tuberhybrida`
- 菊花 — `Chrysanthemum morifolium`
- 熊掌木 — `Fatshedera lizei`
- 草莓 — `Fragaria ananassa`
- 金叶女贞 — `Ligustrum vicaryi`
- 二乔玉兰 — `Magnolia x soulangeana`
- 牡丹 — `Paeonia suffruticosa`
- 矮牵牛 — `Petunia hybrida`
- 毛白杨 — `Populus tomentosa`
- 日本晚樱 — `Prunus lannesiana`
- 东京樱花 — `Prunus yedoensis`
- 锦绣杜鹃 — `Rhododendron pulchrum`
- 海滨木槿 — `Talipariti hamabo`

## 17个已匹配来源但本轮未采到合格图片项

- 重阳木 — `Bischofia polycarpa`
- 鸡冠花 — `Celosia cristata`
- 柑橘 — `Citrus reticulata`
- 铁线莲 — `Clematis florida`
- 翡翠珠 — `Curio rowleyanus`
- 孔雀肖竹芋 — `Goeppertia makoyana`
- 金丝桃 — `Hypericum monogynum`
- 箬竹 — `Indocalamus tessellatus`
- 铺地柏 — `Juniperus procumbens`
- 乐昌含笑 — `Magnolia chapensis`
- 锦葵 — `Malva cathayensis`
- 椤木 — `Photinia bodinieri`
- 淡竹 — `Phyllostachys glauca`
- 早园竹 — `Phyllostachys propinqua`
- 樱桃 — `Prunus pseudocerasus`
- 银柳 — `Salix argyracea`
- 多花白鹤芋 — `Spathiphyllum floribundum`

完整140项明细、每个划分缺口及来源证据见同名CSV和JSON。原始查询状态来自主数据池；对延后重查的物种，回查原162种数据池记录，避免误报“没有映射”。

本次没有更改任何运行中的训练或采集参数，没有开放新的权重路线，也没有增加模型支持元数据。
