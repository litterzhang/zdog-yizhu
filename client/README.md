# Client

Taro + React + TypeScript + SCSS。开发流程使用官方 CLI，不使用自定义打包或 review/verify 脚本。

```sh
npm ci
npm run dev:h5
npm run build:h5
npm run dev:weapp
npm run build:weapp
```

产物分别在 `dist/h5/` 和 `dist/weapp/`，互不覆盖。`project.config.json` 指向微信产物目录。

- `src/config/brand.ts`：名称与文案。
- `src/config/dialogs.ts`：统一弹框的标题、说明和操作文案。
- `src/styles/`：设计 token。
- `src/services/`：会话内 mock 识别与观察记录。
- `src/assets/`、`design/source/`：照片及线稿。
- `tests/`：领域单元测试。

按需运行 `npm run typecheck`、`npm run lint`、`npm test`。

## 物种样图字段

物种对象 `Species` 顶层预留 `sampleImageUrl?: string | null`，由服务端返回完整的 HTTPS 图片地址。名录详情、名录搜索及手账中的物种对象使用同一字段；它不属于 `knowledge`，也不是用户上传的植物封面或观察照片。

```json
{
  "id": "species-id",
  "sampleImageUrl": "https://images.example.org/plants/reference.jpg"
}
```

上述地址仅说明字段格式，客户端不会请求这个示意地址。服务端未补充该字段前，界面不显示样图区；省略、`null`、空字符串或非 HTTPS 地址同样不显示。有效地址在物种简介下方展示，保持图片比例，加载失败可重试，不使用内置照片替代。服务端应提供有合法使用权限的公开物种样图，并补齐必要来源署名；不要返回其他用户的私有照片。微信端的真实图片加载仍需在服务端提供地址后做真机验证。

## 统一弹框

应用内说明、关联确认、保存、退出和删除操作使用 `useDialog` 与 `components/Dialog`，不直接调用原生 `showModal`。白底、细线、草木绿沿用现有 SCSS token；危险操作使用 `tone: 'danger'`，保留明确的取消和确认按钮。页面隐藏或卸载时取消待确认操作，重复打开不会重复授权；系统权限授权、选图器和全屏照片预览仍由平台提供。

页面将 hook 返回的 `dialog` 放在页面内容之前；Tab 页使用 `useDialog(true)`，弹框期间隐藏原生 Tab 栏，关闭后恢复。H5 支持键盘切换按钮和 Escape 取消，长内容内部滚动。验证沿用现有类型检查、Lint、单测与构建，不增加审查脚本或 UI 依赖；构建通过不代表微信真机验收。

上传使用官方 `@tarojs/plugin-mini-ci`：配置 `.env.local` 后运行 `npm run preview:weapp` 或 `npm run upload:weapp`。这两个命令会联系微信；普通 build 不上传。私钥可放在 Git 忽略的 `.secrets/` 或仓库外，不要放进 `src/`、`dist/` 或提交 Git，不需要 AppSecret。详细配置见根目录 README。

旧静态设计截图仅保留为历史产物，不再维护生成工具；请直接使用 H5 查看应用。Mock 数据在刷新/重启后重置，照片不上传。

CI 兼容性：`package.json` 仅将 `miniprogram-ci` 使用的 Less 固定为 `4.2.0`，避免新版 Less 的模块导出规则导致 `less/lib/less` 加载失败；Taro 的 Less 版本不受影响。更新 CI SDK 时再评估是否移除此 override。

微信语法兼容性：`babel.config.js` 为 `weapp` 单独使用 Taro 的保守 targets（iOS 9 / Android 5），避免 H5 的现代 browserslist 让 `?.`、`??` 残留在上传文件中；H5 配置不变。不要手动修改 `dist/`。

导航采用标准 Taro tabBar：发现（无历史）、足迹（识别历史）、株记（已记住的个体）、我的（账号与数据）。登录当前仅为有明确标记的演示状态，不是真实微信认证。
