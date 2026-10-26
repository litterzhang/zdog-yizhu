# 私有植物手账与识别服务

## 结构与命名

`JournalStore` 保存手账，`RecognitionRuntime` 管理单个后台推理实例，
`RecognitionProvider` 隔离算法，客户端 `RemoteJournal` 负责加载和操作。
这些是正式业务模块，不使用阶段名称作为类名、接口名或表名。
中文物种名集中在 `server/src/yizhu_server/data/bioclip-labels.json` 的 `common_name`；
缺少中文名时显示学名。展示映射与模型预测、物种知识分别管理。

算法沿用 BioCLIP v1 权重，现使用完整发布向量替代 24 项小集合：
108,372 个植物参考条目，另以全生物检索首选辅助不确定性提示。完整来源与边界见 `plant-catalog.md`。
候选可能错误，`confidence` 保持 null；原始分数和全生物首选完整保留，不被用户选择覆盖。
物种知识尚无已核实来源，明确留空。不据此判断食用、药用、安全或树龄。

## 接口

以下私有接口均使用 `Authorization: Bearer …`，由 zdog-auth 返回的 `sub` 决定所有者。
不接受请求中的 ownerId，不从 URL 获取 token。

| 方法/路径 | 内容 |
| --- | --- |
| GET `/api/journal` | 当前账号的足迹、档案、观察、候选名录及模型状态 |
| POST `/api/photos` | multipart 单图：`file`、`requestId` |
| GET `/api/photos/content` | 私有图片；额外使用 `X-Photo-Id` 请求头，不把私人图片 ID 写入代理日志 URL |
| POST `/api/recognitions` | `photoId`、`requestId`；返回 202，任务持久化后排队 |
| POST `/api/recognitions/{id}/retry` | 重试失败任务，保留同一记录 ID |
| POST `/api/recognitions/{id}/confirm` | `speciesId` 或 null；更新用户确认字段并同步关联株记的物种，保留原始识别结果 |
| POST `/api/plants` | `recognitionId`、`nickname`、可选 `location` / `story` / `description`；同时建立首次观察 |
| POST `/api/plants/{id}/edit` | 更新昵称、地点、故事、物种及私人描述，不修改个体 ID；省略 description 保留旧值 |
| POST `/api/plants/{id}/observations` | `photoId`、`note`、`requestId`；或识别记录 ID 与明确的同株确认 |
| POST `/api/observations/{id}/edit` | 更新观察笔记 |
| POST `/api/journal/{kind}/{id}/delete` | 删除单条足迹、档案或观察 |
| POST `/api/journal/clear` | `confirm:true`，删除自己的全部手账与图片 |

私有图片响应 `Cache-Control: no-store`。客户端鉴权下载后使用本机临时路径显示，
不会将 token 放到 Image 的 URL。退出/过期清空内存缓存，小程序还会尽力删除下载的临时文件；
浏览器临时资源由浏览器生命周期管理。退出不删除服务器数据。
Taro H5 上传和下载必须显式设置 `withCredentials:false`，仍传递 Authorization 请求头。
不要为修复浏览器拦截而开启跨域 Cookie 或通配 Origin；服务端刻意不返回
`Access-Control-Allow-Credentials`。

手账快照只携带其记录引用的物种详情，不再返回全量名录；公开名录分页 API
见 `plant-catalog.md`，不需要登录，也不接触私人记录。

`Plant.description` 最多 1,000 字，可修改或清空；只属于当前用户的这个体。
公共 `Species.knowledge` 由有出处的版本化资料导入，不允许客户端写入。

## SQLite 与文件

默认 `server/.data/journal.sqlite3`、`server/.data/photos/`，可用 `YIZHU_DATA_DIR` 调整。
目录权限 700、数据库及图片 600，禁止放入 H5 静态根目录；Git 忽略整个 `.data/`。
照片目录由程序管理，不要手工放入其他资料。

- `docs`：记录 kind、UUID、owner、JSON、时间；索引 `(owner,kind,created)`。
- `photos`：照片 UUID、owner、上传请求键、文件摘要、时间；唯一约束 `(owner,request_key)`，
  以及 owner 索引。文件名由服务器生成，不使用用户提供的文件名。
- `usage`：按 owner 和北京时间日期累计识别请求；删除足迹不会恢复当天额度。
- 个体 ID 使用服务器 UUID，与昵称、学名、照片哈希和坐标无关。
- 识别与上传有重复请求键；建档按识别记录避免重复创建；关联与追加避免重复观察。
- SQLite 事务串行保护写入。当前仅支持一个应用进程，不要部署多个推理 worker 竞争同一队列。

### 足迹与株记的唯一关联

- Recognition 的 `linkedPlantId` 是持久关联，同一足迹只能归入一个株记。
  建档与关联在写事务中校验：重复建档返回已有株记，跨株记关联返回 409，同株重复关联不重复追加观察。
- 已关联足迹显示“查看株记”，不再提供建档或关联其他植物的按钮。
- 确认、更改或清空足迹物种，同时更新所关联株记；不改写原始候选、个体 ID、其他足迹。
  多条足迹属于同株时，以最近一次手动确认/档案编辑为准。
  首次关联时已有确认物种会同步；未确认足迹不会覆盖株记现有物种。
- 删除单条观察不解除足迹归属；删除株记后解除归属，保留足迹可再次建档。
- 旧数据按首次建档来源和观察引用自动补齐关联。若已关联多个株记，返回
  `associationConflict:true`，禁止确认/再次关联，不自动合并或删除用户记录。
  用户需先整理重复观察或删除不需要的档案；首次建档来源仍保留其归属。

重启会恢复等待中的识别；用户删除等待/运行中的任务后，后台完成结果不会把它重新插回。
模型一次预热后复用。预热可能需数分钟，但不阻塞普通 API。客户端每 5 秒查询尚未完成的任务，
隐藏应用时暂停，重新进入/登录后从服务器恢复，不要求停留在结果页。

## 图片、删除和限额

- 每次上传原文件不超过 900 KiB，请求总大小不超过 1 MiB；JPEG/PNG/WebP、静态、最多 20MP。
- 应用 EXIF 方向后只保留 RGB 像素，最长边缩至 1600，保存为 JPEG。不保留原文件或 EXIF。
- 小程序尝试压缩大照片；H5 当前需自行选取不超过 900 KiB 的照片。
- 每账号每天 30 次识别、最多一个等待/运行任务；全局最多 8 个等待/运行任务。
- 每账号最多 200 张照片、200 个档案、1000 条足迹和 1000 条观察；全局最多 2000 张照片。
- 删除足迹会清除观察对它的关联，但保留档案/观察仍引用的照片。
- 删除档案同时删除其观察；仍被足迹引用的图片保留。删除最后一条观察时，档案封面仍可保留图片。
- 实际文件仅在数据库事务提交后删除。未被引用的新上传满 24 小时后，在后续上传/服务启动时清理，并非实时定时删除；
  启动时也清理失去数据库引用的文件和过期上传临时文件。
- “删除我的全部数据”只作用于当前账号，不会删除认证中心账号或手机相册。

## 运行与验证

使用 `uv sync --project server --extra recognition --locked` 安装 CPU 推理依赖。
普通服务用 `uv run --project server --extra recognition --locked yizhu-server`；
避免多 worker 和开发自动重载重复加载模型。权重下载及校验见 `recognition-baseline.md`。

H5：`https://yizhuh5-h.942295.xyz`，使用已有认证中心账号密码。
API：`https://yizhu-h.942295.xyz/api`。H5 CORS 放行明确的 Origin 及 `X-Photo-Id`，
微信需配置 request、uploadFile、downloadFile 三类合法域名。

建议手动验证：登录 → 选图识别 → 离开再从足迹找回 → 手动选物种/保留待确认 →
建档 → 追加观察 → 关联另一条足迹 → 编辑 → 退出重登 → 分别删除足迹、观察与档案。
最终在“我的”测试数据删除。请使用测试照片，不上传身份资料等无关敏感内容。

## 当前限制与后续演进

SQLite 和磁盘保存不等于高可用云服务；主机、隧道离线会影响访问，目前没有自动异地备份。
完整账号注销、跨端绑定界面、大列表分页、专用缩略图、算法拒识与候选范围扩展仍待完善。
后续更换识别模型或存储实现时，应保留现有个体 ID、用户确认、原始服务结果和接口约定，
通过显式数据迁移/模型标签映射演进，不重新生成用户植物身份。
