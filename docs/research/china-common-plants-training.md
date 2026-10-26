# 中国常见植物候选集与离线分类器

日期：2026-09-22。用户授权获取约 1,000 种常见植物清单并训练分类器。
不修改客户端、线上识别方案、目录可见性或私人记录。

**后续状态：用户已要求暂停并重选（2026-09-22）。旧后台任务已停止，旧清单不再作为
常见植物训练范围。保留 312 张图片、90 条特征及三类冒烟模型；完整模型未产出。
采集/训练入口由 `paused.json` 阻止误续跑。下文运行记录为暂停前的历史，
当前调整方案见 `china-everyday-plants-selection.md`。**

## 已取得的清单

- `china-common-plants-1000.csv`：**1,000 种**，UTF-8 BOM，可直接用表格软件打开。
- `china-common-plants-1000.sources.json`：来源、筛选策略和快照校验值。
- 完整本地快照：`server/.cache/training/china-common-1000/species.json`。

实际方法：iNaturalist 中国区域（place ID 6903）、植物界（taxon ID 47126）、
research-grade、species 级观察次数降序，逐页读取；与本地可见中文物种目录做
**唯一、学名精确匹配**，按稳定 Species ID 去重，取前 1,000。
截断前有 120 个未满足条件的候选被排除；最后一个入选物种有 49 条观察。
无手写俗名补丁、无模糊匹配，也未修改数据库。

这不是全国普查或“全国最常见 1,000 种”的权威排名。
存在观察者、地域、可见中文名筛选和野生植物偏差；室内盆栽与栽培植物可能不足。
没有假称已完成地区平衡。首个计数页取得于 9 月 21 日，其余于 9 月 22 日，
分页数据不是原子快照。需要后续单独补充园艺场景及审核异名对应关系。

## 图片来源与划分

训练图片在**全球**公开记录中寻找，而非只使用中国照片，增加目标物种的数据可得性。
这不意味着模型已经在中国用户拍照场景上通过测试。

- 使用 iNaturalist 官方 API 获取 public research-grade 图片元数据。
- 每张图片独立检查许可，仅纳入 **CC0 / CC BY**；不因观察记录获许可就假定所有照片获许可。
- 仅从官方开放图片桶获取 medium（最长边约 500px）版本。
- 保留公开观察页、照片 ID、许可链接、摄影者署名及处理说明。
- 一次观察最多一图；去除 EXIF，保存像素重编码 JPEG。
- 不保存坐标、个人简介、评论或用户私人图片；observer ID 只保存不可逆散列分组。
- 观察者全局固定划分 70% / 15% / 15%；同一观察者不能跨训练、验证、测试。
- 每物种目标为 **20 / 5 / 5** 张，每摄影者每物种最多 3 张。
- 全局排除内容相同图片及 dHash 汉明距离 ≤4 的近重复图。

这种分组比随机拆分照片保守，但**仍不能证明不同观察者没拍过同一株**；
dHash 也不能找出所有裁切/近重复。research-grade 标签不是专家审核金标准。
与基础模型预训练数据的重合尚未排除，因此离线分数不是独立线上准确率。

每物种最多读取 5 页、每页 100 条；不足则如实记为数据缺口。
不通过伪标签、重复图片或文本向量补齐类别。类别达到三个集合最低数量才参与训练。

## 训练方案

冻结现有校验通过的 BioCLIP 2（同一 revision/权重 SHA），官方图像预处理，
提取归一化 768 维特征，只训练线性分类头。只使用本机 CPU，不租用算力。
无需新增 Python 依赖，沿用已安装 recognition extras。

- 单线程、batch 256、AdamW、学习率 0.01、weight decay 0.01、固定种子。
- 最多 200 轮，验证损失早停（patience 15），测试集不选参数。
- 对照：相同训练特征的最近类别原型，不把它冒充线上完整识别方案对照。
- 产物：`classifier.safetensors`、`model.json`、实际样本 `training-manifest.jsonl`。
- 记录类别映射、基础模型/依赖版本、校验值、训练参数及分集合/分类别指标。
- 仅为闭集实验分类器，没有训练非植物/库外拒识，也没有概率校准。
- 不自动上线；识别输出始终要求人工核对，`confidence=null`。

## 实际可执行命令

在项目根目录，使用已有 `server/.venv`：

```sh
# 已有清单直接复用，不重复抓取
server/.venv/bin/python -m yizhu_server.training_data list

# 可续跑；单请求流 <=30 次/分钟，403/429 停止，不绕过限制
server/.venv/bin/python -m yizhu_server.training_data collect

# 默认要求全部目标物种均有足够数据；输出目录必须不存在
server/.venv/bin/python -m yizhu_server.training_classifier train \
  --output server/.artifacts/models/china-common-v1

# 一次运行采集与训练；数据不足时显式训练满足条件的子集
bash server/scripts/train-common-plants.sh server/.artifacts/models/china-common-v1

# 对自己有权使用的本地图片离线预测，不访问线上记录
server/.venv/bin/python -m yizhu_server.training_classifier predict \
  --output server/.artifacts/models/china-common-v1 --photo /path/to/plant.jpg
```

首次采集保存 `collection.json` 的固定时间上界，续跑不会混入更晚发布的记录。
单独运行训练时，只有显式 `--allow-partial` 才允许导出少于 1,000 类的模型；
模型元数据同时记录 requestedClassCount 和 classCount，不会标称为完整 1,000 类。
脚本不会启动网络服务器、修改 `.env` 或重启 Flask。

数据缓存、图片、特征及模型位于 Git 忽略目录，不提交第三方照片。
完成标记表示该物种的有限采集轮次结束，不代表图片一定足够；
缺口见 `coverage.json`，过程见 `progress.json`。
保留至少 4 GiB 可用磁盘；暂时失败后重跑可复用已落盘照片与特征。

## 官方来源（本轮实际读取）

1. iNaturalist API schema：`https://api.inaturalist.org/v1/swagger.json`
2. 地点查找：`https://api.inaturalist.org/v1/places/autocomplete?q=China`
3. 频次接口：`https://api.inaturalist.org/v1/observations/species_counts`
4. 公开图片来源和尺寸、署名说明：
   `https://github.com/inaturalist/inaturalist-open-data`
5. 本地模型校验/预处理实现：`server/src/yizhu_server/recognition_rerank.py`。

官方 API 推荐实践网页本轮返回 403；没有假装读到该网页。
使用保守单请求流和遇到访问/限流错误即停的策略。

## 验证状态

清单已生成；训练链路包含合成特征单元测试与安全权重导出回归。
合成测试仅验证实现，不能作为植物识别准确率。
真实采集/训练执行情况、产物和限制见本文件后续运行记录；没有模型产物就不能称为训练完成。

## 2026-09-22 实际运行记录

### 小规模真实训练已完成

首次仅取清单前三类：构、白车轴草、诸葛菜，共 **90 张**公开授权照片、
68 个观察者分组；训练 60、验证 15、测试 15。
实际完成 BioCLIP 2 图像编码、200 轮上限内的分类头训练、safetensors 导出、
重新加载权重并对其中一张公开测试照片执行离线 `predict`。
预测返回 `requires_review` / `experimental-closed-set` / `confidence=null`。

模型目录：`server/.artifacts/models/china-common-smoke-20260922/`。
这只是 **3 类的训练链路冒烟模型**，不是完整 1,000 类模型，更不能用于线上替换。
15 张测试图上分类头和最近原型基线都为 15/15；样本少、类别差异大、
预训练重叠未排除，**不能据此声称识别率 100% 或优于现有方案**。
最佳验证轮为 199；全部参数、分类别指标和来源见模型目录。

独立 systemd 用户任务于北京时间 10:16:21 开始、10:27:52 成功结束；
日志报告墙钟约 11 分 31 秒、CPU 约 11 分 15 秒、内存峰值 2.4G、
swap 峰值 407.3M。主要时间是图片编码，不是小分类头训练。
这也说明本机全量任务可能持续很久，不能承诺几分钟完成。

### 完整清单任务已启动，尚未完成

后台用户任务：`yizhu-classifier-training-20260922.service`。
已实际核实 `active/running`，当前为图片采集阶段；
阶段性读取到 163 张图片、5 类已完成有限采集轮次。数量会继续变化。
随后自动提取特征、训练满足条件的物种子集；目标仍为 1,000 类，
数据不足会在 `coverage.json` 与模型实际类别数中明确体现。

资源限制：Nice 10、最多约 1 核 CPU、MemoryMax 2500M、
MemorySwapMax 512M；磁盘不足 4 GiB、来源 403/429 等情况主动停止。
它是独立临时用户任务，不是 Flask 服务，也未配置开机自动恢复。
机器重启或任务失败后，需要人工检查并重新运行可续跑脚本。

最终预期目录（**尚未产出**）：
`server/.artifacts/models/china-common-20260922/`。

```sh
# 查看状态与公开数据任务日志
systemctl --user status yizhu-classifier-training-20260922
journalctl --user -u yizhu-classifier-training-20260922 -n 30 --no-pager
cat server/.cache/training/china-common-1000/progress.json

# 如需暂停任务，保留已下载照片/特征
systemctl --user stop yizhu-classifier-training-20260922
```

实际校验：新增文件 Ruff 通过、`bash -n` 通过、后端 **260 项测试通过**。
线上本机 `/api/health` 仍返回 `ready`，方案仍为 `bioclip-v2-rerank`。
没有改动客户端、生产配置、用户记录，也未重启 API 或发布模型。
