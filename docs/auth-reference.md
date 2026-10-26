# zdog-mp 登录参考

2026-09-20：已克隆到 `/home/zhang/workspace/zdog-mp`，阅读版本 `a1c604c`。仅做源码分析，未修改参考仓库、安装其依赖、运行其服务或调用真实认证接口。

## 当前实现，不是旧文档中的本地 JWT 方案

`zdog-server/src/authclient.py` 把认证委托给独立的 `zdog-auth`。服务端读取 `ZDOG_AUTH_ISSUER`、`ZDOG_AUTH_CLIENT_ID`、`ZDOG_AUTH_CLIENT_SECRET`；客户端密钥不下发前端。

### 小程序

`Taro.login()` → 临时 `code` → `POST /login_wx` → 服务端调用认证中心 `/machine/wechat/miniprogram` → 返回 `access_token` → Taro storage 与前端状态。

主要文件：
- `zdog-client/src/components/auth/login.jsx`
- `zdog-server/src/routes/auth.py`
- `zdog-server/src/authclient.py`

### H5

设计路径为用户名/密码登录或注册，经 `/auth/login`、`/auth/register` 转发到认证中心 `/machine/password/login`、`/machine/register`。不是在浏览器调用 `Taro.login()`。

**当前 checkout 存在缺文件问题：**`CredentialForm.jsx` 导入 `./credentials`，但仓库中没有该模块。`git check-ignore` 确认根 `.gitignore` 的 `credentials.*` 会匹配 `credentials.js`。因此不能把该 H5 登录当作当前源码可直接运行的完整参考。

### 身份与退出

- 请求用 `Authorization: Bearer …` 传递令牌。
- 服务端调用 `/machine/userinfo` 校验，使用返回的 `sub` 作为业务用户 ID，不信任客户端提交的 userId。
- 单次请求内缓存身份；认证服务故障返回 503，和令牌无效的 401 分开处理。
- `/auth/bind/wechat` 用当前账号令牌与微信 code 绑定账号，避免直接微信登录另建一个不关联的账号。
- 客户端退出清理 storage 与 Redux 状态；本次未确认认证中心是否提供令牌撤销/刷新机制。

## 对一株的接入结论

按用户后续要求，一株后端已改为 Flask，通过自己的 `/api/auth/*` 路由调用统一认证中心。保留轻量前端状态管理，不引入股票业务或 Redux。

接入前需要为一株确认独立认证客户端配置，以及认证中心对一株小程序 AppID 的支持方式；仅凭此仓库不能确认认证中心的多 AppID 实现。所需凭据不是小程序代码上传私钥，秘密值只在服务端本地配置。

一株的公网 API 为 `yizhu-h.942295.xyz/api/`，H5 为 `yizhuh5-h.942295.xyz`；`host.942295.xyz` 的现有 `*-h` 规则去掉后缀后，通过 SSH 隧道访问本机不带 `-h` 的同名站点。后端 POST 路由、严格 CORS 和基于 `sub` 的身份校验已实现。前端已接入微信 code 登录、H5 密码登录与 `/me` 校验；原演示账号已移除。用户已配置服务端凭据，重启后本机 HTTPS health 返回 `auth.configured=true`，不等于已验证认证中心连通或 AppID 对应关系。

参考代码中的 URL query token 兼容入口、`scope.userInfo` 前置申请，以及“只要有本地 token 就算已登录”的判断未照搬。前端 token 仅经 Authorization header 发往 API，恢复时必须重新验证，不保留密码。用户已反馈微信登录成功；跨端账号关系与注销流程仍待完善。植物数据持久化现已实现，详见 `journal-service.md`。
