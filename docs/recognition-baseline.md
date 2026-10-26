# BioCLIP v1 本机基线

> 历史记录：下文描述 2026-09-20 的 24 项植物小集合实验。
> 2026-09-21 在线服务已切换为完整参考检索，现状见 `plant-catalog.md`；
> 复现本页旧实验必须显式传入下面的 `--labels`。

## 范围

这是一项**真实模型的离线候选排序实验**，不是 mock 输出，也不是已经上线的识别服务。
不改变小程序当前流程，不向公网注册接口，不上传照片、不获取定位。
API 进程不会导入 Torch 或加载模型；推理由独立命令按需启动。

## 已固定的来源

- 官方模型：`https://huggingface.co/imageomics/bioclip`
- revision：`ce901ab3c6a913f9e9ef94ce6d27761069f4f01c`
- 文件：`open_clip_pytorch_model.bin`，598,599,013 bytes。
- 官方 LFS SHA-256：`e380384f0c30d425d8c6c40f24471f9dd497fbdfa734a89c461a94aee95f0ef4`。
- 官方该 revision 的模型卡声明 MIT；代码来源和许可证见 `https://github.com/Imageomics/bioclip`。
- pybioclip：`https://github.com/Imageomics/pybioclip`，MIT，实际依赖锁在 `server/uv.lock`。
- 作者与许可说明见 `recognition-options.md`；不要把模型许可当成训练照片的再分发授权。

模型文件放在 Git 忽略的 `server/.cache/models/bioclip-v1/`，不进入前端包、Git 或 Web 静态目录。
加载前检查完整文件大小及 SHA-256，并使用 `weights_only=True`；不会加载未校验的 checkpoint。

## 安装与运行

在项目根目录：

```sh
uv sync --project server --extra recognition --locked
```

安装只增加可选 CPU 推理依赖。Torch 和 torchvision 绑定官方 CPU index，
不自动安装 CUDA；没有 `recognition` extra 的普通服务器仍能运行 Flask。
锁文件包含的实测组合：

| 包 | 版本 |
| --- | --- |
| pybioclip | 2.1.6 |
| open-clip-torch | 3.3.0 |
| torch | 2.14.0+cpu |
| torchvision | 0.29.0+cpu |
| pillow | 12.3.0 |

如需重新下载权重，可使用安装后的官方 Hugging Face CLI，在 `server/` 下执行：

```sh
uv run --extra recognition --locked hf download imageomics/bioclip \
  open_clip_pytorch_model.bin \
  --revision ce901ab3c6a913f9e9ef94ce6d27761069f4f01c \
  --local-dir .cache/models/bioclip-v1
```

此下载命令是官方工具用法，当前网络直连 Hub 存在超时；本轮实际下载采用官方
CDN，公网主机只用于取得官方下载跳转，不承载推理。不要将权重下载失败误报为模型不可用。

在项目根目录运行基线：

```sh
uv run --project server --extra recognition --locked yizhu-recognize \
  client/src/assets/photos/ginkgo-leaf.jpg \
  client/src/assets/photos/ginkgo-tree.jpg \
  --labels server/src/yizhu_server/data/bioclip-labels.json \
  --threads 2 --repeat 3 --include-controls \
  --output .artifacts/recognition/bioclip-v1.json
```

输出文件已存在时拒绝覆盖，重跑请换文件名。命令输出 JSON，不写图片路径或元数据。
`HF_HUB_OFFLINE=1` 保证推理过程不自动访问 Hub；不会默认切换到 BioCLIP 2。

> 2026-09-21 更新：标签文件已升为 `baseline-labels-v2`，删除所有手工 `common_name`。
> 学名、ID 与非植物对照保留；历史实验记录不改写。线上中文名只来自全国名录。

## 实验边界与输出解释

- 标签文件：`server/src/yizhu_server/data/bioclip-labels.json`，
  24 个手工选择的植物学名和 5 个非植物对照；不是完整植物名录或中文物种数据库。
- 使用 pybioclip 的多文本模板及 BioCLIP v1 图像预处理。只覆盖本地模型加载方法，
  禁用默认 JIT 编译以降低本机冷启动成本；不是量化或重新训练模型。
- 两张现有银杏照片来自项目已有授权资源；并非独立物种测试集。
  额外的纯白图和固定种子噪声为合成负例，不代表真实非植物图片分布。
- `score` 是这 29 个标签内部的 softmax 分数，候选列表变化会改变它。
  `confidence` 始终为 null，结果始终 `requires_review`。不自动确认物种，不建立植物个体关联。
- 最高分是植物也不能证明输入包含该植物；非植物标签不是经过验证的拒识机制。
- 报告记录初始化、每次图像推理耗时和进程峰值 RSS。峰值包含模型初始化与标签向量生成；
  推理耗时不包含联网上传、排队和持久化，也不代表微信端端到端延迟。
- 输入限制：单图最多 8 MiB / 20 MP，仅静态 JPEG、PNG、WebP；应用 EXIF 方向后只保留 RGB 像素。

## 2026-09-20 实测结果

本机 Intel Xeon E3-1225 V2，4 个可用 CPU、7.6 GiB 内存；
本轮使用 2 个 CPU 线程，每个输入重复 3 次，**无 GPU 推理**。
权重大小和 SHA-256 均已验证。完整报告在 Git 忽略的
`.artifacts/recognition/bioclip-v1.json`，不是线上监控日志。

| 输入 | 首选候选 | 单次推理耗时中位数 |
| --- | --- | --- |
| 现有 `ginkgo-leaf.jpg` | Ginkgo biloba（银杏） | 0.8334 秒 |
| 现有 `ginkgo-tree.jpg` | Ginkgo biloba（银杏） | 0.8313 秒 |
| 合成纯白图 | Pinus massoniana（植物，负例错误） | 0.8245 秒 |
| 固定种子随机噪声 | Oxalis corniculata（植物，负例错误） | 0.8298 秒 |

- 初始化：**346.1615 秒**，含依赖导入、权重校验/加载和标签向量生成；
  不能把这个合并计时单独称为“模型加载时间”。
- 总运行：356.1548 秒；进程峰值 RSS：**1563.32 MiB**，约 1.53 GiB。
- 纯白图首选植物的原始分数约 0.355，噪声图约 0.759。
  这直接说明 softmax 分数不能当成正确概率，简单保留几个非植物标签也不足以拒识。
- 这是两个同物种样例加两个合成负例，**不计算、不宣称产品准确率**。
- Ruff 和全部 84 项后端测试通过。新增测试覆盖模型完整性检查、标签校验、
  不伪造置信度、非植物标签、图片格式/大小/元数据及 Flask 不加载推理依赖。
  自动化单测没有下载或加载模型；上表来自另外执行的真实离线命令。

## 下一步门槛

先缓存版本化标签向量，降低每次启动重复计算的开销；再设计图片质量检查和未知输入处理。
需补充有可靠标签的不同物种照片与真实非植物负例，比较候选命中和资源消耗。
在候选范围、未知输入行为和延迟经过验证前，不直接替换线上 mock，也不宣称识别准确率。
生产接入还需要鉴权上传、请求额度、任务状态、文件清理、中文名称来源与数据库。
