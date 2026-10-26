# Pl@ntNet 新物种分类头与植物存在二分类

日期：2026-09-23。目标仍是中国常见植物候选集的**端到端 Top-5 ≥90%**，
不是将实验中已收齐图片的小类别集合当成目标完成。

## 离线方案

1. 第一级复用按验证集规则选出的 ImageNet ResNet50 植物存在二分类头。
2. 第二级加载本地原版 **1,081 类** Pl@ntNet ResNet50 预训练权重，
   移除旧 `fc`，冻结骨干，使用归一化 2,048 维特征训练新的物种线性分类头。
   **不是 Pl@ntNet-300K V2 权重，也不调用 Pl@ntNet 识别 API。**
3. 物种编码沿用官方 resize256 / center-crop224 / ImageNet normalization；
   二分类沿用全图 letterbox224。不同预处理与权重分别写入元数据和特征缓存键。
4. 分类头先训练，验证损失早停。部分骨干微调尚未实现／执行；
   先根据这条基线的验证表现判断后续需要，不依据最终测试集选参数。

分类输出是候选物种，不是具体植物个体 ID；归一化 softmax 得分不作为校准概率，
返回 `confidence=null`。范围外植物拒识尚未解决。

## 保守拦截与评测

训练结束后，使用**二分类验证集及当前物种快照验证集**选阈值，规则为：

```text
min(旧阈值,
    二分类验证集达到 99% 植物召回的阈值,
    物种验证集达到 99% 植物召回的阈值)
```

这是有限验证集上的经验条件，不承诺真实用户召回达到 99%。
模型、阈值、参数方案和验证案例在测试评分前写入固定文件。
测试阶段不重新选择模型或阈值。原二分类头文件不改写。

报告包含：
- 单独物种分类 Top-1 / Top-5、端到端 Top-1 / Top-5、按物种平均值；
- 已训练物种集合及数量、请求目标数、无测试图片的目标清单；
- 未训练类别的测试图片、二分类拒绝、技术失败均保留在分母；
- 仅覆盖类别的成绩另列，不替代总体指标；
- 固定新阈值在既有 Open Images 二分类 test 集上的误放行率和召回率。

原二分类测试集曾用于旧实验；当前公开图片快照也不等于独立中国现场验收集。
**即便小样本 Top-5 超过 90%，仍不能宣称目标达成。** 尚需完整目标覆盖、
更丰富的图片和独立的中国拍照场景测试。跨来源检查照片哈希与近重复，
但不同平台的摄影者身份对应、基础模型预训练数据重叠仍未知。

## 固定快照，不干扰采图

`snapshot` 从采图器原子写入的逐类文件读取，复制公开图片并核对哈希。
快照保留全部 972 个目标及当时已有照片；只有满足 70/15/15 的物种训练分类头。
不修改实时采集目录，不删除旧清单的暂停标记，不读取私人照片。
少于六类时拒绝运行 Top-5 实验；部分覆盖必须显式 `--allow-partial`。

从项目根目录执行：

```sh
server/.venv/bin/python -m yizhu_server.plantnet_transfer snapshot \
  --source server/.cache/training/china-everyday-972 \
  --data server/.cache/training/china-everyday-972-snapshot-20260923

bash server/scripts/train-plantnet-cascade.sh \
  server/.cache/training/china-everyday-972-snapshot-20260923 \
  server/.artifacts/models/plantnet-china-v1 --allow-partial

# 对自己有权使用的图片执行本地完整串联预测
server/.venv/bin/python -m yizhu_server.plantnet_transfer predict \
  --model server/.artifacts/models/plantnet-china-v1 --photo /path/to/photo.jpg
```

新版本应使用新的快照／输出路径。既有快照和训练／测试结果不覆盖。
脚本不访问网络、不重启 Flask、不设置新的线上 Provider。

产物：
- `plan.json`：运行前固定参数、数据与二分类头哈希。
- `classifier.safetensors`：新的安全格式分类头。
- `gate-validation.json`：阈值选择依据及验证案例。
- `model.json`、`model.metadata.json`：权重、预处理、实际物种集合与限制。
- `test.json`：固定模型的测试结果，逐图案例可复算。
- `progress.json`：训练阶段。systemd 状态才是进程是否存活的依据。

`solution_metadata refresh` 可登记新架构，不会错误标成 BioCLIP 分类头，
也不会把它加入线上识别方案。

## 本次实际运行

13:41 冻结快照：862 张照片、972 个请求目标、6 类满足本轮训练最低数量。
因此本次仅验证真实预训练权重、新分类头与二分类组合链路，
不是 972 类最终模型。已经启动临时用户任务：

`yizhu-plantnet-china-20260923.service`

CPUQuota=100%、Nice=10、MemoryMax=1500M、MemorySwapMax=512M。
日志：`server/.artifacts/plantnet-china-20260923/run.log`。
已读取状态为 active/running。当前尚无最终评估结果。
新增合成测试验证泄漏防护、冻结快照、保存重载和指标分母，
**不把合成测试分数报告为植物识别准确率**。Ruff 与 418 项后端测试通过。

### 已完成的链路实验结果

后续已确认任务 inactive/dead、Result=success、ExecMainStatus=0。
六类分类头训练、固定阈值两级评估和原始照片推理重放均完成：

| 范围 | Top-1 | Top-5 |
| --- | ---: | ---: |
| 已训练六类的 90 张 test 照片 | 88/90（97.78%） | 90/90（100%） |
| 包含未覆盖类别的全部 118 张 test 照片 | 88/118（74.58%） | 90/118（76.27%） |

全部 118 张植物照片都通过该门控，因此此批物种和端到端命中率相同。
28 张未覆盖物种照片没有从分母移除；按有测试照片的物种平均的 Top-5 为 54.55%。
测试仅覆盖 11 种，既不是 972 类验收，也未实现用户后来明确的 A 保留+B 扩展。

验证集选定门控阈值 `0.0683201402425766`；原二分类 test 的
植物召回 494/499（99.00%），非植物误放行 248/500（49.60%）。
这表明保守阈值减少误拒植物的同时放过了很多非植物，不能包装成高质量上线结果。
测试结果未用于重新选择阈值；原二分类权重与合同未改写。

结果：`server/.artifacts/models/plantnet-china-v1/test.json`。
保存重载后，已对一张公开紫薇原图进行完整串联推理，
Top-5 ID 顺序与缓存评估一致，二分类得分差小于 1e-6；
证据为 `server/.artifacts/plantnet-china-20260923/raw-photo-replay.json`。
支持合同仍是原模型 A 的 420 种，没有因该实验头而扩张，未上线。
