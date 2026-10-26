# 一株

个人植物识别与观察记录项目。名称待定；已接入 BioCLIP 实际推理、zdog-auth 登录及私有手账存储。算法候选范围有限，结果需人工确认，不用于食用、用药或安全判断。

```text
client/       Taro + React + TypeScript + SCSS，微信与 H5 共用源码
server/       Python + uv + Flask，接入 zdog-auth（保留本地 H5 预览能力）
nginx.conf.d/yizhu.conf  H5 与 API 的域名配置
docs/         产品说明与平台准备记录
```

## 本地开发

```sh
cd client
npm ci
npm run dev:h5       # Taro 默认 H5 开发流程，127.0.0.1:10086
npm run build:h5     # 输出 dist/h5/
npm run build:weapp  # 输出 dist/weapp/
```

不使用自定义打包、静态评审或产物校验脚本。普通检查按需运行 `npm run typecheck`、`npm run lint`、`npm test`。

## 本机植物识别实验

已完成 BioCLIP v1 的 CPU 推理基线，并通过后台识别任务接入客户端。
在线候选已扩展为 **108,372 个植物参考条目**，不是世界植物全集；
发现页提供植物名录入口，支持“目 → 科 → 属 → 种”浏览，详情保留已有生长形态资料，不再提供形态筛选。
完整参考文件安装、来源、实测和中文名/形态资料限制见 `docs/plant-catalog.md`。
中文显示现覆盖 11,278 个植物参考条目，支持全国中文名与学名搜索；
其余 97,094 条在数据库标记为不可见，不出现在公开名录和展示候选中，但保留识别向量与历史引用。
目／科／属也优先显示有来源的中文名称。来源和重建说明见 `docs/chinese-names.md`。
每株植物还可保存最多 1,000 字的私人描述，与故事/观察笔记分开。
带出处的公共物种简介单独显示；全国数据与描述说明见 `docs/national-names-and-descriptions.md`。
安装可选依赖使用 `uv sync --project server --extra recognition --locked`；
运行入口为 `yizhu-recognize`。官方权重、具体命令、实测耗时与负例误判见
`docs/recognition-baseline.md`。模型缓存与实验报告不提交 Git，不从零训练模型。
`yizhu-recognize` 默认使用新的完整参考；旧小集合需显式传入 `--labels`。
新增离线 `--rerank` 对照：v1 自动召回前 50 个候选，再用本地 BioCLIP 2 重排。
需要额外的已校验权重，不会自动下载或替换线上模型；方法与限制见
`docs/recognition-reranking.md`。

## 部署 H5 与 API

H5 直接由 Nginx 服务构建目录，不依赖 Python：

```sh
npm --prefix client run build:h5
# 本机已配置 SELinux 标签规则；重建后恢复静态文件标签：
sudo restorecon -R client/dist/h5
```

Nginx 的 H5 `root` 为 `/home/zhang/workspace/zdog-yizhu/client/dist/h5`，`try_files` 支持前端页面地址刷新；缺失 JS/图片返回 404。

API 才需要启动 Python：

```sh
cd server
uv sync --extra recognition --locked
uv run --extra recognition --locked yizhu-server  # 默认监听 127.0.0.1:8091
```

本地开发优先用 `npm --prefix client run dev:h5`。Python 仍保留 `http://127.0.0.1:8091` 的本地静态预览，但公网 H5 不再经过它。

`nginx.conf.d/yizhu.conf` 使用 HTTP → HTTPS 跳转、共享 `/etc/nginx/ssl/942295/` 证书；仅 API 代理到 `127.0.0.1:8091`：

- 公网 H5：`https://yizhuh5-h.942295.xyz`
- 公网 API：`https://yizhu-h.942295.xyz/api/`，健康检查为 `/api/health`
- 本机没有公网 IP，复用 `host.942295.xyz` 的现有 `*-h` 代理：去掉 `-h`，通过 `127.0.0.1:8014` SSH 隧道转发到本机 HTTPS。**客户端和微信 request 合法域名使用带 `-h` 的公网域名**；本机 Nginx 的 `server_name` 仍为 `yizhu.942295.xyz` / `yizhuh5.942295.xyz`，不另建公网站点。
- H5 域名不代理 `/api/`；API 域名不服务前端页面。后端 CORS 允许明确配置的 H5 origin 的 GET/HEAD/POST/OPTIONS，未开启 Cookie 凭据。

```sh
sudo install -m 644 nginx.conf.d/yizhu.conf /etc/nginx/conf.d/yizhu.conf
sudo nginx -t
sudo systemctl reload nginx
```

构建目录必须允许 Nginx worker 读取，父目录必须可遍历。如果 home 是私有目录，不要直接 `chmod 755`；可经确认仅给 `nginx` 用户添加 home 的 traverse ACL（`setfacl -m u:nginx:--x /home/zhang`），不授予 home 列表读取权限。

本机已为 `/home/zhang/workspace/zdog-yizhu/client/dist/h5(/.*)?` 配置持久的 `httpd_sys_content_t` 文件标签规则。仅 H5 构建目录可被 Nginx 读取；SELinux 保持 Enforcing，`httpd_enable_homedirs` 保持关闭。若在其他机器部署，需按实际路径配置对应标签规则。

部署与权限命令不自动执行。H5 只需构建文件与 Nginx，API 另外需要 Python 服务持续运行。

## 微信小程序预览与上传

使用官方 `@tarojs/plugin-mini-ci`，直接调用 Taro，不写自定义上传器：

```sh
cd client
cp -n .env.example .env.local
# 在本机编辑 .env.local；不要把密钥内容发到聊天。
npm run preview:weapp  # 构建并向微信请求预览
npm run upload:weapp   # 构建并上传开发版本，不提交审核或发布
```

普通构建不需要凭据。预览/上传前，在 `.env.local` 配置：

- `TARO_APP_ID`：真实小程序 AppID。
- `MINIPROGRAM_PRIVATE_KEY_PATH`：代码上传私钥的绝对路径，或相对于 `client/` 的路径，例如 `.secrets/private.key`。`.secrets/`、`*.key`、`*.pem` 已加入 Git 忽略；不要放进 `src/` 或 `dist/`，不要提交密钥，权限建议 `600`。
- `MINIPROGRAM_VERSION` / `MINIPROGRAM_DESCRIPTION`：上传版本和说明。

所需的是微信公众平台「管理 → 开发管理 → 开发设置 → 小程序代码上传」中的**代码上传密钥**，不是 AppSecret。该入口来自本轮读取的官方文档；后台不同请以实际页面为准。建议开启 IP 白名单并添加运行上传命令机器的公网出口 IP。不要关闭白名单来绕过配置。

上传命令本身就是发送代码的操作，不会在普通构建时触发。此前经用户确认已成功生成过预览；本轮业务接入只构建 H5，不自动上传新微信版本、提交审核或发布。

## H5 验证

访问 `https://yizhuh5-h.942295.xyz`，使用已有 zdog-auth 账号密码登录。
按「发现选图 → 上传并识别 → 足迹查看结果 → 记住这一株 → 记录新变化」验证；
也可测试关联已有植物、编辑/删除、退出重登后找回记录。

- 网页版当前选图需 **小于 900 KiB**，支持静态 JPEG、PNG、WebP；小程序会尝试压缩较大图片。
- 上传前明确确认，照片只发送到自己的服务器。完整模型启动可能需数分钟，任务会保留并排队。
- H5 与微信账号是否为同一个身份由认证中心绑定关系决定，昵称相同不代表同一账号。
- 微信真机后续需同时配置 **request、uploadFile、downloadFile** 合法域名 `https://yizhu-h.942295.xyz`。
- 数据、接口、索引、限额和删除规则见 `docs/journal-service.md`。

## 当前边界

已实现「发现 / 足迹 / 株记 / 我的」四个 Tab，以及识别结果、植物档案、真实登录页面。「发现」不含历史；识别历史集中在「足迹」，记住的植物集中在「株记」。

前端已接入 Flask / `zdog-auth`：小程序使用 `Taro.login()`，H5 使用已有账号的密码登录；不会在失败时回退到演示账号。认证配置见 `server/README.md`，前端 API 地址集中在 `client/src/config/api.ts`。本机 HTTPS 健康检查已确认 `auth.configured=true`，但这不代表真实认证联调通过。

取得 token 后必须通过 `/api/auth/me` 验证才显示已登录。token 保存在 Taro storage，启动和回到前台时再次验证；不保存密码或客户端密钥。401 清除会话，网络/服务故障保留 token 供重试但不显示已登录。退出只清理本机凭据、页面草稿及缓存状态，不删除服务器记录，也不等于撤销服务端令牌或注销账号。

微信重新预览：在 `client/` 运行 `npm run preview:weapp`（会上传预览代码），扫码后从「我的」进入登录。需在小程序后台配置 request 合法域名 `https://yizhu-h.942295.xyz`，并确认认证中心的客户端与这个小程序 AppID 对应；不要通过关闭域名检查绕过配置。真实微信 code 交换、账号绑定关系和真机登录仍待验证。H5 只提供已有账号登录，暂未提供注册/绑定 UI。

足迹、植物档案和观察保存在 SQLite，照片私有存储在服务器磁盘，所有请求按可信登录身份校验所有权。默认页面不混入示例记录；Mock Provider 仅保留供测试。知识资料源、跨端账号绑定界面、账号注销、异地备份及算法改善仍待完成。H5 构建成功不代表浏览器业务验收或微信真机通过。

依赖仍有已知审计告警（包含官方构建/上传工具的传递依赖），不要当作已完成安全验收的正式产品。当前环境的 `.git` 不可用，没有重建或提交 Git 元数据。

## Repository location

The repository root is `/home/zhang/workspace/zdog-yizhu`. Host configuration and storage changes made during the 2026-10-26 rename are recorded in `docs/repository-rename.md`.
