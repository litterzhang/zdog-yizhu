# Server

Python 3.13+、uv、**Flask + Waitress**。认证通过 `zdog-auth`，不在本服务保存密码或自行签发 JWT。

```sh
uv sync --extra recognition --locked
cp -n .env.example .env
# 在本机编辑 .env，不要把密钥发到聊天或提交 Git。
chmod 600 .env
uv run --extra recognition --locked yizhu-server
uv run --extra recognition --locked pytest -q
```

默认由 Waitress 监听 `127.0.0.1:8091`。公网 API 为 `https://yizhu-h.942295.xyz/api/`，经 `host.942295.xyz` 的现有代理去掉 `-h`，再通过 SSH 隧道转发到本机 Nginx 的 `yizhu.942295.xyz`。`--reload` 使用 Flask 开发重载但不开启 debugger。公网 H5 为 `https://yizhuh5-h.942295.xyz`，最终仍由本机 Nginx 直接服务，不依赖 Python。

## zdog-auth 配置

`yizhu-server` 会读取本目录 `.env`；已存在的进程环境变量优先：

- `ZDOG_AUTH_ISSUER`：默认 `https://auth.942295.xyz`。
- `ZDOG_AUTH_CLIENT_ID`：一株在认证中心注册的客户端 ID。
- `ZDOG_AUTH_CLIENT_SECRET`：该客户端的服务端密钥。
- `CORS_ORIGINS`：可选、逗号分隔的明确 H5 origin，默认允许公网 `https://yizhuh5-h.942295.xyz` 与本机直连 `https://yizhuh5.942295.xyz`；不接受 `*`。若 `.env` 已显式设置，该列表会覆盖默认值，需包含带 `-h` 的公网 H5 origin；代理会改写 Host，但不会改写浏览器 Origin。
- `YIZHU_DATA_DIR`：私有 SQLite 和照片目录，默认 `server/.data`；必须在 H5 静态目录之外，不提交 Git。
- `YIZHU_RECOGNITION_SOLUTION`：服务端识别方案，默认 `bioclip-v1`，可选方案见下表；未知或空值会明确报错。

需要在认证中心确认一株的小程序 AppID 对应关系。这些凭据不是小程序代码上传私钥；本服务不直接使用微信 AppSecret。未配置时登录返回 **503 / AUTH_NOT_CONFIGURED**，没有 mock 登录后门。

修改 `.env` 后需重启后端：`systemctl --user restart yizhu-server`。正式用户级单元位于 `server/systemd/yizhu-server.service`，已链接到 `~/.config/systemd/user/` 并启用；当前账号已开启 linger，因此主机重启后会自动启动，不依赖交互式登录。

## 接口

| 方法与路径 | 用途 / 请求字段 |
| --- | --- |
| `GET /api/health` | 健康检查、Flask 框架标记、认证配置状态；不会探测认证中心 |
| `POST /api/auth/login` | H5 账号登录：`username`、`password` |
| `POST /api/auth/register` | 注册：`username`、`password`、可选 `nickname` |
| `POST /api/auth/wechat` | 小程序登录：来自 `Taro.login()` 的 `code` |
| `GET /api/auth/me` | 使用 Bearer token 校验身份并返回 `user.id`、`user.nickname` |
| `POST /api/auth/bind/wechat` | 使用 Bearer token 和 `code` 绑定当前账号的微信身份 |

登录成功仅返回 `code` 与 `access_token`，不会转发上游的密码、密钥或微信 session_key。所有私有业务路由使用 `auth_required` 和 `g.user_id`；身份取自认证中心返回的 `sub`，不接受客户端 ownerId。手账 API 与数据库说明见 `../docs/journal-service.md`。

认证中心超时/不可用返回 503，和令牌无效的 401 区分。只接受 Authorization Bearer header，不支持 URL query token。没有未经核实的刷新/撤销接口，也不会在启动时注册角色、开通资源或创建用户。

## 当前限制

- 前端已接入小程序 code 登录、H5 账号密码登录和私有手账；用户此前已反馈微信登录成功，新业务链路仍需真机验证。
- 自动化认证测试使用模拟 HTTP，不请求真实账号；`auth.configured=true` 仅说明本机配置存在。
- 认证 POST 按 IP 做单进程内存限流（20 次/分钟），重启清空，不适用于多实例部署。
- CORS 允许明确来源的 GET/HEAD/POST/OPTIONS 和 Authorization/Content-Type/X-Photo-Id，不使用 Cookie 凭据。
- 单进程 Waitress + 一个后台识别 worker，只加载所选方案需要的模型；组合方案会同时驻留 v1 和 v2。请勿以多个 Web worker 或开发重载方式运行实际识别服务，避免重复加载/处理任务。尚无自动异地备份。

本地静态预览仍可通过 `http://127.0.0.1:8091` 使用；默认读取 `../client/dist/h5`，可用 `YIZHU_H5_DIST` 调整。生产 H5 使用 Nginx 的 `root`，不是此本地预览路由。

## 植物识别

`uv sync --extra recognition --locked` 安装锁定的 CPU-only BioCLIP 依赖。
`RecognitionRuntime` 在后台预热所选方案并处理数据库中的任务，
普通 API 仍可响应；`/api/health` 的 `recognition` 为 `warming`、`ready` 或 `failed`。
本机离线运行入口 `yizhu-recognize` 仍可用于独立实验。
权重 revision、校验值、样例命令与局限见 `../docs/recognition-baseline.md`。
模型保存在 `.cache/models/`，不提交 Git；推理报告默认只输出到终端，
可用 `--output` 写入 Git 忽略的 `.artifacts/`，拒绝覆盖已有报告。

默认在线使用 `CatalogRecognitionProvider`，需要额外安装已校验的完整参考向量和名录。
按 `../docs/plant-catalog.md` 准备两个固定文件和
`../docs/national-names-and-descriptions.md` 中的全国 XLSX，使用
`uv run --extra recognition --locked yizhu-catalog`
生成 `.cache/catalog/treeoflife-10m/catalog.sqlite3`。
导入同时保存版本化的物种简介，私人描述则保存在独立的用户手账库中。
首选中文名优先使用全国名录，缺失部分已通过 ScienceDB 补充；另有来源别名。
无需字形转换依赖或在线翻译。
可选离线对照使用 `yizhu-recognize --rerank`，需准备已校验的 BioCLIP 2 权重；
不自动启用第二模型。方法和实测限制见 `../docs/recognition-reranking.md`。
未安装时名录返回明确的 503，识别不偷偷回退到 24 项旧候选；已有私有手账仍保留。
名录读取无需登录，分页每次最多 50 条，只返回 `is_visible=1` 的物种；缺少中文名的记录保留但隐藏。
旧库需重新导入以增加可见性列，见 `../docs/chinese-names.md`。
私有手账只附带被引用的物种数据，历史隐藏引用仍可读，不因名录隐藏而删除私人记录。

## 中文名称补全

新增 `python -m yizhu_server.name_enrichment`，从已校验的 ScienceDB 原文件及
Wikidata 公开快照生成独立待审 SQLite 数据集；`--apply-from` 将符合来源规则的别名
写入新目录副本。`--reveal-sciencedb` 明确用于补缺失首选名并恢复可见性，
可用 `--expected-count` 防止误改范围；命令不联网或自动部署，保留已有名称及 ID。
196 个物种的 241 个别名、8,438 个补名物种均已部署，现有 19,716 个可见物种。
API 已重启并验证。命令、来源规则、备份与统计见 `../docs/name-enrichment.md`。

识别／手账快照的 `species.commonName` 优先使用已采纳、有来源的中文别名
（例如桂花），回退顺序为名录中文名、学名。`chineseName` 仍保存名录名称
（例如木犀），公开名录接口不变。此逻辑只在返回时处理，不更新数据库或私人昵称，
旧客户端重新加载手账即可使用；已有识别不需要重新推理。

## 保存与选择识别方案

版本化方案保存在 `src/yizhu_server/recognition_solutions.py`，不把代码存进用户数据库。
旧 BioCLIP 方案版本为 `1`；改变算法参数或语义时须更新方案版本，不覆盖历史结果。

本机在 2026-10-06 将线上方案切换为
`target285-paraguayense-bioclip2`。该版本在284类基础上新增物种胧月
`Graptopetalum paraguayense`，不把园艺品种另立类别。该模型保留原249类并加入35个完成物种级
审核、训练与独立终验的多肉物种，共284类；植物二分类门控开启。第一套284类
独立集未达标并降级为均衡回放，第二套全新独立集的宏端到端 Top-1 / Top-5
为 **82.39% / 92.25%**，纯物种 Top-1 / Top-5 为85.56% / 95.77%。本机和公网
健康检查均确认 `ready`、`supportedPlantCount=284`、`goalMet=true`。
旧 `target249-bioclip2` 与 `plantnet-china` 592类模型仍保留为显式回滚方案。
这与代码/配置模板的安全默认值 `bioclip-v1` 不同；实际运行值以 `/api/health` 为准。

| 配置值 | 方案 | 状态 |
| --- | --- | --- |
| `bioclip-v1` | 原 v1 全参考检索 | 默认 |
| `bioclip-v2-rerank` | v1 前 50 召回 + v2 默认中心裁剪重排 | 实验 |
| `bioclip-v2-whole` | v1 前 50 召回 + v2 整图缩放重排 | 实验 |
| `bioclip-multiview-plantclef` | 整图／局部召回 + v2 重排，不确定时本地 PlantCLEF | 新增实验方案；正式权重及真实推理验收待完成 |
| `target285-paraguayense-bioclip2` | ImageNet植物门 → BioCLIP 2有限285类分类头 | 当前线上 `20261006.1`；新增胧月；285类封存集宏端到端Top-1 82.46%、Top-5 92.28% |
| `target284-succulent-bioclip2-replay284` | ImageNet植物门 → BioCLIP 2有限284类分类头 | 保留回滚方案 `20261005.2`；纵向照片整图门控失败时增加中心方形复核；284/284第二套独立覆盖，宏端到端Top-1 82.39%、Top-5 92.25% |
| `target249-bioclip2` | ImageNet植物门 → BioCLIP 2有限249类分类头 | 保留回滚方案 `20261001.1`；宏端到端Top-1 85.54%、Top-5 94.38% |
| `plantnet-china` | ImageNet 植物存在二分类 → Pl@ntNet A=420＋已训练 B=172 | 保留回滚方案 `20260924.1`；592 类，完整开发测试 Top-5 70.59%，未达独立验收 |
| `plantnet-china-431` | 原 A=420＋B=11 两级模型 | 保留的显式回滚方案 `20260923.1`，不自动选择 |

新方案没有第三层 LLM，不自动外发图片，不自动改变线上选择。
仅代码及合成测试通过不代表模型完整可用；先按 `../docs/recognition-cascade.md`
准备并校验 PlantCLEF 权重，再安排离线验证。当前截断／稀疏下载文件不能作为已完成权重。
离线入口支持 `yizhu-recognize ... --solution bioclip-multiview-plantclef`，不会切换 API。

在 `server/.env` 中选择一项，已有进程环境变量优先：

```dotenv
YIZHU_RECOGNITION_SOLUTION=target285-paraguayense-bioclip2
```

重启后端后生效，本机使用 `systemctl --user restart yizhu-server`。
`GET /api/health` 的 `recognition_solution` 显示配置选中的 ID、版本及实验标记；
同时检查 `recognition=ready`，仅显示方案名不代表模型加载成功。
回滚可改回 `plantnet-china` 或 `bioclip-v1` 并重启，不需要改 `client/`、
重新构建或上传小程序。

- 选择在启动时固定，客户端请求不能指定方案，也没有公共切换接口。
- 不按某张照片的用户答案自动挑模型，不宣称能自动选出“最准”的方案。
- 新执行的任务使用当前方案；重启时仍在排队/处理中的任务也会使用新进程的方案。
  已完成记录不重算、不补写历史元数据；切换期间注意队列中的任务。
- 新结果把实际方案 ID/版本保存在 `rawResult.original.solution`，
  保留模型 revision、原始召回及重排信息。客户端仍收到原有 source/version/status/candidates 字段。
- 组合方案对外仍默认最多 5 个候选；后台保留完整候选供核查，中文可见性规则继续生效。
- 配置错误、缺失权重或推理失败均明确失败，不偷偷运行其他方案。
- 旧 v2 方案仍只有极少样例验证，原有缓存保持内存实现；新两级方案增加版本化磁盘
  文本缓存，但未命中的文本仍需编码。新方案接受门控是未校准的一致性规则，
  三模型实际峰值／时延／准确率尚未验收，不应直接当作生产升级。

自有模型研究见 `../docs/research/plant-model-training.md`：
先建立授权明确的图像数据集，比较图像参考检索、冻结骨干分类头，再考虑有限微调；
该研究阶段未训练；后续公开图片训练见下节。未使用私人照片或开通 GPU。

## 离线训练自己的植物分类器

训练数据缓存已迁移到独立 `/data` 文件系统，并通过 bind mount 保持 `server/.cache/training` 路径兼容；挂载、校验和恢复步骤见 `../docs/training-storage.md`。


共享物种名录已拆为 `species_core`（稳定 ID / 学名 / 分类等）与 `species_zh`
（中文名 / 别名 / 简介 / 来源），分类节点同样按稳定 ID 关联中文翻译。
兼容视图保留现有接口，迁移和回滚说明见 `../docs/catalog-schema.md`。
基于新结构的有界中国植物训练见 `../docs/research/china-species-training.md`；
不等于批准恢复旧观察频次排名或完成“中国常见 1,000 种”。

每个方案的植物集合、来源和版本可通过 `yizhu_server.solution_metadata` 查询。
训练头会自动保存旁路 `.metadata.json`，全方案索引从真实标签/名录生成：

```sh
server/.venv/bin/python -m yizhu_server.solution_metadata refresh
server/.venv/bin/python -m yizhu_server.solution_metadata list
```

名录或模型变化后需刷新快照。输出范围与当前可见植物分开记录，二分类器不声称
具有物种名单；详见 `../docs/solution-metadata.md`。此操作不加载模型或切换线上方案。

植物存在二分类独立于物种识别：使用冻结的 ResNet50 和两类线性头，
训练数据改用 Open Images 的人工核验 Plant 正/负标签，而不是把动物照片
自动当作非植物。数据准备、训练/评估/预测命令与限制见
`../docs/research/plant-presence-training.md`。目前不注册为线上拒识开关。
扩展到 5,990 张后，两种骨干训练/测试已完成，按验证集规则选择 ImageNet。
其 999 张封存测试结果为植物召回率 92.79%、非植物误放行率 21.80%，
尚未达到 95% 测试召回目标，不作为线上硬拦截器。完整对照见上述文档。

模型对照已明确采用**本地推理，不使用在线识别 API**，Pl@ntNet 分支先评估官方
预训练权重，再根据实测决定是否微调；与所选 V2 数据的对应关系仍待核实。
见 `../docs/research/plant-base-models-top5.md`。
`plantnet-300k-pretrained` 仍不是运行时 Provider，不要填入 `YIZHU_RECOGNITION_SOLUTION`。
官方 **SqueezeNet 1.0 和 ResNet50 / 1,081 类（非 V2）** 已有本地真实试跑结果及独立 CLI，
不是已启用的新线上模型；见 `../docs/research/plantnet-pretrained-trial.md`。
同一组 52 张公开测试照片，两者 Top-1/Top-5 均为 5/52；
仅直接覆盖 11 个测试物种中的 1 个，不能视为总体准确率。

```sh
server/.venv/bin/python -m yizhu_server.plantnet_pretrained predict \
  --architecture squeezenet1_0 --photo /path/to/plant.jpg \
  --output server/.artifacts/plantnet-pretrained/new-photo.json
```

将 `--architecture` 改为 `resnet50` 可使用已下载并校验的 ResNet50 权重。

来源提取和候选池已落地，见 `../docs/research/china-everyday-candidate-pool.md`。
独立提取命令（只读公开资料和物种目录，不运行模型）：

```sh
uv run --locked --script server/scripts/build-plant-candidates.py \
  --output server/.cache/research/everyday-candidate-pool-next
```

**当前已按用户要求暂停旧清单训练。** 旧数据集的 `paused.json` 会阻止以下旧流程
误续跑；不删除图片/模型。新场景候选和来源核验缺口见
`../docs/research/china-everyday-plants-selection.md`，未确认新范围前不恢复训练。

中国区域观察频次筛选的 1,000 种中文清单及训练说明：
[`../docs/research/china-common-plants-training.md`](../docs/research/china-common-plants-training.md)。
只采集公开 CC0 / CC BY 图片，冻结已有 BioCLIP 2 并训练线性分类头；不读取私人照片、
不替换线上模型。命令从项目根目录执行：

```sh
bash server/scripts/train-common-plants.sh server/.artifacts/models/china-common-v1
```

采集可续跑；数据不足的物种不会硬凑训练，产物如实记录实际类别数。
模型目录必须是新路径。数据/特征/模型都在 Git 忽略的本地目录。

### 固定 972 个候选的独立图片采集

用户已确认可从其他图片来源补齐这批候选。新任务不恢复上述旧排名清单，
只采集 iNaturalist 公开 CC0 / CC BY 图片，目标每种 70/15/15 张：

```sh
# 项目根目录运行；collect 可续跑，不会自动训练或部署
server/.venv/bin/python -m yizhu_server.everyday_training_data prepare
server/.venv/bin/python -m yizhu_server.everyday_training_data collect
```

数据与进度位于 `server/.cache/training/china-everyday-972/`。
分类匹配失败、样本不足分别记录，不将少于 972 类说成完整覆盖。
详见 `../docs/research/china-everyday-images.md`。

### Pl@ntNet 新分类头＋二分类拦截（离线）

实现见 `src/yizhu_server/plantnet_transfer.py`。从公开采集结果创建固定快照，
以原版 Pl@ntNet ResNet50 冻结骨干训练新分类头，验证集选定保守二分类阈值，
再单独测试完整流程。实际支持的物种集合随模型保存；部分覆盖不能冒充 972 类。

命令、产物和验收边界见 `../docs/research/plantnet-cascade.md`。
运行入口为 `scripts/train-plantnet-cascade.sh`；不会自动部署或修改客户端。

用户确认仅 B 的 883 种作为新增目标后，当前采集入口改为：

```sh
server/.venv/bin/python -m yizhu_server.plantnet_training_data collect
```

旧 972 类服务已停止并保留文件，已有合格照片已复用；
40 个已支持的重叠类别不进入新增任务，支持元数据仍为 420。
详见 `../docs/research/plantnet-additions-data.md`。

保留原 A 的分类参数、仅训练新增 B 参数的命令使用
`train-plantnet-cascade.sh ... --allow-partial --incremental`；
必须先生成包含 A 验证数据的专用快照。
见 `../docs/research/plantnet-incremental.md`。权重保持不变不等于排名表现不变，
新增模型采用独立离线版本，当前支持合同不会被训练计划自动扩张。

线上接入、核对及回退说明：`../docs/plantnet-online.md`。
