> 当前更新（2026-09-20）：使用默认 Taro 构建及官方 `@tarojs/plugin-mini-ci@4.2.1`；其 peer 要求 `miniprogram-ci@^1.9.15`，因此使用兼容的 `1.9.15`，未强制忽略依赖冲突。前端已移入 `client/`，H5 为正式构建目标，自定义 review/verify 和上传包装器均已移除。
>
> 最近一次联网安装审计：112 项（1 low、38 moderate、27 high、46 critical），包含构建/上传工具传递依赖。未将这些工具打包进前端，也未据此宣称生产安全。下文为此前记录，当前命令见根目录 README。

# 工程选择与安全待办

## 官方脚手架与版本

2026-09-20，网页浏览工具未返回正文；经正常网络授权后，用 `curl` 成功读取 Taro 官方入门文档：

`https://docs.taro.zone/docs/GETTING-STARTED`

文档说明 `taro init`、本地 CLI、微信构建命令，以及 CLI/平台依赖必须保持一致。文档通用 Node 要求与具体包可能不同；本轮同时检查 npm 官方 registry：`@tarojs/cli` stable tag 为 `4.2.1`，包 engines 为 `node >=18`。

实际使用官方 CLI `@tarojs/cli@4.2.1`，在 `/tmp/yizhu-scaffold-20260920` 生成默认 React + TypeScript + Sass + Webpack5 模板，再只复制新文件，未覆盖既有文档、设计资源或 AGENTS.md。

初次 `--npm npm` 被 CLI 拒绝；检查该版本 CLI 自身代码后，使用区分大小写的 `--npm Npm --css Sass` 成功。模板自动安装遇到 npm 12 的 `EALLOWSCRIPTS`，随后在项目根目录单独执行 `npm install` 成功。没有全局安装 CLI。

锁定实际安装组合：

- 所有 `@tarojs/*` 直接依赖与 `babel-preset-taro`：`4.2.1`。
- React：`18.3.1`（模板 React 18 范围）。
- TypeScript：`5.9.3`（模板 TypeScript 5 范围）。
- Webpack：`5.91.0`（模板固定版本，runner 的 peer 也精确要求此版本）。
- Node：实际验证 `24.19.0`；没有宣称其他 Node/操作系统已通过验证。

移除了未使用平台的插件、H5/RN 构建脚本、提交钩子等；保留模板需要的依赖，不强行覆盖 Taro 的内部依赖树。静态校样单独使用 esbuild，不为产品添加 H5 目标。

## 依赖审计：发布前阻塞

联网 `npm audit --json` 报告 **47 项：21 moderate、22 high、4 critical**。安装静态校样工具后的 npm 审计数量相同。

关键链包括：

- Taro CLI 引入的 `decompress`：压缩包解压路径/链接问题。
- Taro components 引入的 `swiper`：原型污染问题，影响被传递标记到上层包。
- 模板固定 Webpack 版本：公开路径、可选 HTTP 构建加载路径等告警。

这不是说所有告警路径都会进入微信产物，也**不能由此认定小程序不受影响**。需要后续逐条可达性核查、上游修复或经过验证的兼容替代。当前不使用不可信模板/压缩包，不启用 Webpack HTTP 构建，不对公网开放开发服务器。

已核查 registry 中更新的 Webpack 5，但 Taro runner peer 精确固定 `5.91.0`，所以没有未经验证强制替换。`npm audit fix --force` 建议包含跨大版本 Taro 降级，不执行。此依赖树仅用于本地 mock 样板，不能据此上线。

离线 audit 曾返回空漏洞集合，但缺少联网公告，**不将它作为安全通过结果**。

## 后端和生产安全尚未实现

- 不创建云函数、数据库或收费资源；先评估当前价格、识别耗时、网络限制。
- `RecognitionProvider` 是真实后端扩展边界；样板明确拒绝非 mock provider。
- 后端 owner 必须来自可信登录上下文；不得直接采用本地 mock owner 或客户端输入。
- 每个实体读写、图片访问、批量删除都必须验证所有权。
- 计划索引：Plant `(ownerId, updatedAt)`；Recognition `(ownerId, createdAt)`；Observation `(ownerId, plantId, observedAt)`。具体数据库语法与索引要在选型后核实，不把本表当作已部署配置。
- 生产需要单图格式/大小验证、服务端解码检查、缩略图、EXIF 处理、识别配额、频率限制、幂等键、可恢复任务、引用计数/清理队列和删除重试。
- 当前只限制本地选图 10 MB，没有上传链路、没有定位请求、没有秘密值。
- 后续第三方识别的数据流向、许可、保存期限与费用必须先说明并由用户确认。

## Git 与平台边界

当前环境可见 `.git` 目录，但 `git status` / `git log` 报告不是可用仓库；没有重建、覆盖或提交 Git 元数据。`package-lock.json` 已生成，仍需在正常 Git 环境中由后续提交纳入版本控制。

未修改 `/home/zhang/workspace/xdog`，没有加入任何其他 workspace。没有开通资源、提交备案、审核、发布或上传预览包。
