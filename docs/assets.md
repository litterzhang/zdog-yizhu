> 前端整体迁移后，下文的 `src/` 与 `design/` 路径均相对于 `client/`。

# 资源来源与处理

核查日期：2026-09-20。现有 `design/generated/icon-v1/` 文件被完整保留，本轮没有把该实验图作为正式品牌头像，也没有认定其已获用户批准。

## 示例照片

通过 Wikimedia Commons 官方 API 的 `imageinfo / extmetadata` 核实：以下两张图的作者为 **Tangopaso**，`LicenseShortName = Public domain`、`Copyrighted = False`、`Credit = Own work`。

| 本地文件 | 原文件 | 原页面 |
| --- | --- | --- |
| `src/assets/photos/ginkgo-tree.jpg` | Ginkgo leaves (Vaires) 1.jpg | `https://commons.wikimedia.org/wiki/File:Ginkgo_leaves_(Vaires)_1.jpg` |
| `src/assets/photos/ginkgo-leaf.jpg` | Ginkgo leaves (Vaires) 5.jpg | `https://commons.wikimedia.org/wiki/File:Ginkgo_leaves_(Vaires)_5.jpg` |

取得的是官方 1280px 缩略版本，然后以 ImageMagick 缩小到 900px / 1000px 长边、JPEG 质量 76 / 80，移除元数据。未添加滤镜或修改植物色彩。应用中的裁切仅用于版面，点击可查看完整的打包照片，不宣称它是摄影原始文件。

两图仅说明页面外观，**不证明示例档案是同一株植物**。档案的昵称、地点、观察时间与故事均为虚构。应用有「示例照片」标记与来源说明入口。

示例照片随包提供，不在运行时请求外部图片地址，不上传用户照片。

## 原创线稿与图标

`design/source/branch.svg`、`sprig.svg`、`camera-white.svg`、`album.svg`、`plus-white.svg` 为本轮直接编写的原创矢量图，没有复制第三方图标库或图形。允许本项目使用和修改。

应用资源统一位于 `src/assets/line/`，使用 ImageMagick 从上述 SVG 导出为 PNG，并降为 8 位通道以控制大小。源 SVG 不是 React DOM 组件，不在小程序中通过浏览器 SVG API 渲染。

修改线稿后应同步重新导出 PNG，并检查透明底、深浅背景对比、体积和真机显示。

## 首页快捷图标

`src/components/PageHome.tsx` 与 `src/app.scss` 的 `page-home__*` 使用 Taro View 和
原创 SCSS 线条绘制房屋图标，允许本项目使用和修改。不依赖字体字符、第三方素材、
浏览器 SVG 或生成图片；点击回到“发现”，屏幕阅读器名称为“回到首页”。

## 四 Tab 导航图标

`client/design/source/tabs/` 中的四组 SVG 为直接绘制的线性导航图标（发现、足迹、株记、我的），包含普通/选中颜色。`client/src/assets/tabs/` 是对应 72×72 静态 PNG，用于 Taro 标准 tabBar。未调用图像生成服务。
