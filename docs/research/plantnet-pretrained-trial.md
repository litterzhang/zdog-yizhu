# Pl@ntNet 官方预训练权重：本地试跑

日期：2026-09-22。用户要求实际尝试官方预训练模型。
本轮完成真实本地推理，不使用在线识别服务、不恢复训练、不切换线上模型。

## 已取得并运行的模型

- 官方仓库公开共享目录实际列出 27 个 checkpoint；不能把“训练代码支持某架构”
  当作“对应权重已下载”，本次已经取得目录文件名和大小。
- 最初选择 ResNet50（206,149,183 bytes），但传输很慢。
  为先完成一次真实试跑，改用目录中的 **SqueezeNet 1.0** 小模型，
  文件 `squeezenet_weights_best_acc.tar`，**10,354,289 bytes**，完整下载耗时约 283 秒。
- 标签使用同一官方入口的 `class_idx_to_species_id.json` 和
  `plantnet300K_species_id_2_name.json`，均为 **1,081 项**。
- SqueezeNet checkpoint 的分类层形状为 `[1081, 512, 1, 1]`，
  在 torchvision SqueezeNet 1.0 上严格加载成功。
  **这是原 1,081 类预训练模型，不是 Pl@ntNet-300K V2 模型。**
- 不把 V2 的 1,000 类 CSV 套到这些权重上，也不推断其他未加载 checkpoint 的类别数或成绩。

已完成权重 SHA-256：
`3644b96450ad01af55d294a9a7179a4f160ebbaf9c2afa6aed8c304ace5d5507`。
这是本地下载后校验值；文件大小与官方目录一致，但没有冒充发布方签名或发布方独立哈希。

来源：

```text
https://github.com/plantnet/PlantNet-300K
https://seafile.plantnet.org/d/01ab6658dad6447c95ae/
https://seafile.plantnet.org/d/bed81bc15e8944969cf6/
```

列目录使用公开文件服务接口，仅下载权重/标签，没有提交任何图片做远程识别。
代码 BSD-2-Clause 不自动代表独立 checkpoint/照片的授权；独立权重许可未另行确认，
本次不作为公开部署/再分发许可已完成的声明。

## 推理配置

按官方 `utils.py` 和 README 的预训练测试路径：
短边缩放 256、中心裁剪 224、RGB tensor、ImageNet mean/std。
没有运行下载的 Python；自己构建本地 torchvision 模型，
`torch.load(weights_only=True, map_location="cpu")` 安全读取，严格检查输出层并加载。
未使用 `weights_only=False`，未下载 ImageNet 初始化，也没有随机分类头回退。

实际版本：torch `2.14.0+cpu`、torchvision `0.29.0+cpu`、Pillow `12.3.0`。
CPU 单线程、低优先级；本机 NNPACK 无法初始化，因此在离线预测上下文中显式关闭。
checkpoint 没有完整训练参数记录，这不是对论文原环境的逐位复现。

## 实测结果

下表先保留首轮 SqueezeNet 结果；后续 ResNet50 实测对照见下节。

使用已经下载、来源/许可已记录的公开图片中 **test 分组的全部 52 张，11 个物种**。
没有使用 train 分组、没有重拍挑图、没有按模型是否覆盖该物种筛选测试集，
也没有访问私人识别历史或株记图片。
每次预测保留完整 1,081 类分类头，不用这 11 个答案反向缩小模型候选范围。

| 指标 | 本次结果 |
| --- | ---: |
| 按保守学名直接匹配，测试物种被原配标签表覆盖 | **1 / 11** |
| Top-1 命中 | **5 / 52（9.62%）** |
| Top-5 命中 | **5 / 52（9.62%）** |
| 逐物种宏平均 Top-5 | **9.09%** |
| 技术推理成功 | **52 / 52** |
| 单张中位耗时（含解码、预处理和候选整理） | **0.08865 秒** |
| 单张 P95 耗时 | **0.0922 秒** |
| 进程峰值 RSS（不是纯权重大小） | **352.71 MiB** |
| 初始化耗时 | **3.4531 秒** |

唯一直接对齐的测试物种是白车轴草 *Trifolium repens*，对应 5 张均 Top-1 命中。
构、鸭跖草、萝藦、莲、诸葛菜、酢浆草、红花酢浆草、垂序商陆、马齿苋、
阿拉伯婆婆纳的当前学名未直接对齐到标签表。
**未做完整学名异名审查，因此这些是直接名称覆盖/命中统计，不是最终物种概念对齐后的基准。**

另试随包公开银杏照片：未命中；原配表中没有 `Ginkgo` 属名。
实际读取 V2 物种 CSV 后，严格名称交集同样仅含这组测试中的白车轴草，也没有 `Ginkgo biloba`。
这只说明该小样本的标签范围问题，不能据此断言所有中国植物都不支持。

### 不能把这个结果解释成什么

- **不能说“Pl@ntNet 总体准确率只有 9.62%”。**
  这里同时受标签覆盖、未统一异名和小样本选择限制。
- 白车轴草 5/5 也不能推导出该物种实际准确率为 100%。
- SqueezeNet 的成绩不能当作 ResNet50、PlantCLEF 或 Pl@ntNet 在线服务的成绩。
- iNaturalist 研究级标签不是专家复核金标准，基础模型训练集重合也未排除。
- 不能与尚未在同一测试集上实测的 BioCLIP 方案作准确率高低结论。

### ResNet50 后续实测（2026-09-22）

经 `host.942295.xyz` 下载并传回完整官方权重，文件大小 **206,149,183 bytes**，
两端 SHA-256 一致：
`4bdaa41d62d0fb9640f6ba137812560d2f2d587ecabe810e3c3571625f72305c`。
这不是发布方独立签名校验。使用 `weights_only=True` 安全读取，
ResNet50 分类层 `[1081, 2048]` 检查及完整 state dict 严格加载通过。
**同样是原版 1,081 类，不是 V2。**

复用完全相同的 52 张公开 test 图片；测试清单摘要、逐张 sample ID 和图片哈希
均已与 SqueezeNet 报告比对一致。两轮均为 CPU 单线程；不是同时进行的性能基准。

| 指标 | SqueezeNet 1.0 | ResNet50 |
| --- | ---: | ---: |
| 直接名称覆盖 | 1 / 11 种 | 1 / 11 种 |
| Top-1 | 5 / 52（9.62%） | 5 / 52（9.62%） |
| Top-5 | 5 / 52（9.62%） | 5 / 52（9.62%） |
| 覆盖范围内命中（仅白车轴草） | 5 / 5 | 5 / 5 |
| 技术推理成功 | 52 / 52 | 52 / 52 |
| 单张中位耗时 | 0.08865 秒 | 0.32835 秒 |
| 单张 P95 耗时 | 0.0922 秒 | 0.3321 秒 |
| 初始化 | 3.4531 秒 | 5.47 秒 |
| 进程峰值 RSS | 352.71 MiB | 605.79 MiB |

ResNet50 这轮总耗时 23.5541 秒（含初始化和两个合成对照）。
空白图、随机噪声仍分别输出植物候选；未实现经过验证的非植物拒识。
离线模块 26 项测试本轮通过（3.72 秒），未重跑整套后端或客户端检查。

结论：在这个标签覆盖受限的小样本上，更大的原分类网络没有增加命中。
这不是两个骨干的总体能力排序；覆盖内只有一个物种，不能据此证明真实准确率。
下一步需先解决目标标签覆盖/异名核对，再决定是否用授权目标图片训练新分类头；
本轮没有训练、调整阈值或选择更容易的测试照片。

## 未知/非植物问题

空白图、随机噪声两个合成对照仍会输出植物类别。
这只是闭集分类器的技术检查，不是完整非植物测试集或误报率评测。
没有宣称已实现可靠的“不是植物”或未知物种拒识。
softmax 仅记为未校准 score，`confidence=null`，不把它包装成正确概率。

## 文件与复现命令

新增离线模块：`server/src/yizhu_server/plantnet_pretrained.py`。
它不注册到服务器 `YIZHU_RECOGNITION_SOLUTION`，也没有修改 `.env`。

```sh
# 已有准备好的 manifest，无需重复 prepare；输出路径必须是新文件
server/.venv/bin/python -m yizhu_server.plantnet_pretrained predict \
  --architecture squeezenet1_0 --photo /path/to/plant.jpg \
  --output server/.artifacts/plantnet-pretrained/new-photo.json

server/.venv/bin/python -m yizhu_server.plantnet_pretrained evaluate \
  --architecture squeezenet1_0 \
  --data server/.cache/training/china-common-1000 \
  --include-controls \
  --output server/.artifacts/plantnet-pretrained/new-test.json
```

- 实际完整结果：`server/.artifacts/plantnet-pretrained/official-squeezenet-1081-test.json`
- ResNet50 完整结果：`server/.artifacts/plantnet-pretrained/official-resnet50-1081-test.json`
- ResNet50 简要结果：`docs/research/plantnet-resnet50-trial.json`
- 简要结果与校验信息：`docs/research/plantnet-pretrained-trial.json`
- 权重和原配标签：`server/.cache/models/plantnet-300k-pretrained/`（Git 忽略）

复现 ResNet50 对照（项目根目录运行，输出文件必须不存在）：

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 nice -n 10 \
server/.venv/bin/python -m yizhu_server.plantnet_pretrained evaluate \
  --architecture resnet50 --data server/.cache/training/china-common-1000 \
  --include-controls \
  --output server/.artifacts/plantnet-pretrained/new-resnet50-test.json
```

## 验证、下载状态与结论

Ruff 通过；后端 **319 项测试通过**，其中新增 26 项离线模型/映射/评估规则测试。
真实 SqueezeNet checkpoint 加载、单图预测、52 张批量预测及合成对照都实际完成。
没有将合成权重测试或文件下载成功当作模型推理成功。

首轮 ResNet50 慢速下载曾停止并保留 **17,825,792 bytes** 的 `.part`；
后续经中转下载已完成，完整权重已严格加载并完成上述真实推理，旧部分文件仍保留。
训练暂停标记保留，未提交照片给外部识别 API、未重启生产服务。

结论：**官方预训练模型确实可以本地运行，但这份原分类头不能直接作为当前目标的完整覆盖方案。**
如继续这条路线，下一步应核对实际目标类别/异名和权重许可，
再考虑以预训练骨干训练目标分类头并校准拒识；不是继续把原标签换成中文就算支持了新物种。
这属于后续选择，本轮没有自行开始训练或切换线上模型。
