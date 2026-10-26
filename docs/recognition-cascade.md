# 多视图 BioCLIP + PlantCLEF 两级方案

实现日期：2026-09-21。新增可选方案 `bioclip-multiview-plantclef`，版本 `1`，
标记 `experimental=true`。**现有三个方案保留，默认与线上选择没有自动变更。**

## 实际流程

1. v1 对整图召回前 50 个不同物种。
2. 整图低细节、非植物或未解析分类首选导致拒识时，直接返回不确定，
   不靠局部背景叶片或第二级模型强行变成成功。
3. 否则生成两个局部方形视图：短边的 0.8 倍，长轴中心在 1/3、2/3，
   短轴居中、框限制在原图内。短边不足 64 像素不加局部，重复框不重复计算。
   常规比例下局部有重叠；极端长宽比不保证重叠。整图始终保留。
4. 每个局部各召回最多 50 个候选；拒识局部不贡献新候选或支持票。
   原整图前 50 全部保留，按 RRF（k=60）最多加入 50 个新的物种 ID。
5. v2 对至多 100 个候选重排，仍使用原图的官方预处理，不添加手工答案。
6. 缺少跨视图支持时调用本地 PlantCLEF；没有第三层 LLM／Jev 或外部图像 API。

## 门控是实验规则，不是已校准正确概率

第一级给出正常待确认候选的条件：

- 重排首选在目录中可见，分数为正且与第二名不是数值平局；
- 该物种出现在至少两个有效视图的前五中，并至少在一个视图中排名第一；
- 原始召回与重排均没有拒识。

未满足时调用 PlantCLEF。第二级仅在其原始首选已精确映射、可见、有唯一领先，
同时位于 v2 前五并得到上述多视图支持时采用它的候选顺序。
不跳过 PlantCLEF 未映射的首选后把后续类别冒充可靠首选，
也不让隐藏的 v2 首选被悄悄转换为成功。

若第二级没有这些支持，返回 `uncertain`：
原候选与第二级已映射候选交替去重展示，仍允许用户手动确认。
**第二级发现的新增物种可以作为参考候选，不会仅因高 softmax 自动变成成功。**
因此该策略偏保守，可能增加不确定比例；后续须依据独立样本调整并升级策略版本。

所有 `confidence` 为 null。不同模型的分数不相加、不跨模型比较；
混合参考列表明确标注 `stage_scores_not_comparable_or_calibrated`。
`requires_review` 仍是等待用户确认，不是系统保证鉴定正确。
裁剪视图并非独立样本，一致性也不等于正确率。

`rawResult.original.cascade` 保存策略版本、`calibrated=false`、调用／接受原因、
各阶段实际耗时、各路原始召回和第二级类别结果。
用户纠正不覆盖这些原始数据。正常仍对外最多 5 个候选。
模型或文件的技术异常明确失败，不伪装成分类不确定，也不悄悄回退到旧方案。

## PlantCLEF 权重准备

固定来源为 DOI `10.5281/zenodo.10848263` 中的
`vit_base_patch14_reg4_dinov2_lvd142m_pc24_onlyclassifier_then_all`。
使用 EMA 权重，7,806 类；不是对一株全部物种的独立分类器。

官方包：
`https://zenodo.org/api/records/10848263/files/PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar/content`

- 大小：2,271,216,640 字节。
- 发布方 MD5：`42bc1bf3725ebe0bc93a2167c6b3deda`。
- 模型记录许可：CC BY 4.0，保留作者、DOI、许可和 EMA 格式转换说明。
  原始训练照片许可单独处理。

取得完整文件后，在仓库根目录执行：

```sh
server/.venv/bin/python -m yizhu_server.recognition_plantclef \
  /path/to/PlantNet_PlantCLEF2024_pretrained_models_on_the_flora_of_south-western_europe.tar
```

默认输出 `server/.cache/models/plantclef-2024/prepared/`，拒绝覆盖已有目录。
先验证完整包，再只读取允许的 checkpoint、类别及学名表；
不整包解压、不执行原示例。checkpoint 仅以 `weights_only=True` 和 CPU mmap 读取，
只允许额外解析标准库 `argparse.Namespace` 这一数据容器，且完全忽略其参数，
架构和预处理由代码固定；其他未知类型仍拒绝。只保留 `state_dict_ema`，
转换为 safetensors；不允许 `weights_only=False` 等不安全回退。
本地 manifest 保存来源及转换后文件校验值，它不是发布方数字签名。

选择新方案时先验证资源，再加载 v1/v2；PlantCLEF 本体按需加载，使用
`pretrained=False`、严格 state_dict 检查和 518×518 预处理。
服务启动不会连接模型站点自动下载。`ready` 不代表按需第二级已做过真实推理验收。

**当前限制：两次整包下载截断，分段下载也在完成 15/542 块后因不完整响应退出，
已保留断点与文件，当前没有继续运行的下载任务。稀疏 `.part` 文件显示完整大小
不表示下载完成，必须通过完整包校验。此时不能将新方案视为已具备生产启用条件。**

断点记录：`server/.cache/models/plantclef-2024/range-progress.json`；
需要在网络可用时续取，或从其他网络取得完整官方包后执行上面的准备命令。

2026-09-21 后续启用检查：用户已要求启用，但预检仍缺准备好的正式资产，故未改
实际 `.env`、未重启。两个官方入口的 512 KiB Range 探测分别耗时
53.81／43.13 秒；另核查微软 SPARROW 的较小转换包入口，
相同大小探测耗时 54.66 秒，没有解决传输问题，未下载该转换模型或改变推理后端。
当前本机 API 实际返回 `bioclip-v2-rerank`、`recognition=ready`，
不是新方案已启用。没有删掉 PlantCLEF 后冒称完整两级方案，也没有加载未校验的部分文件。

## 文本缓存与资源

只有新方案启用持久 v2 文本缓存：
`server/.cache/models/bioclip-v2/text-vectors.sqlite3`。
键包含模型权重、模板、tokenizer 配置与库版本，值是归一化 float32 文本向量。
最多 10,000 条，旧记录淘汰；损坏向量删除并重新编码，不使用 pickle。
不保存图片特征、私人图片地址或用户内容。首次遇到大量新候选仍需编码，
不能把缓存称为准确率改进或承诺冷启动秒级响应。

每张图最多三次 v1、一次 v2，第二级最多一次调用；
没有新增 worker、LLM、GPU 服务或请求并行。未知实际峰值和延迟仍须测量。

## 离线运行和显式切换

资源准备完成后可先离线运行，不改变正在运行的 API：

```sh
server/.venv/bin/python -m yizhu_server.recognition_cli \
  client/src/assets/photos/ginkgo-leaf.jpg \
  --solution bioclip-multiview-plantclef --repeat 1 \
  --output server/.artifacts/cascade-sample.json
```

报告拒绝覆盖，权限 0600，不写私人路径。
新方案的多个模型会同时使用内存，不要在当前约 8 GiB 主机上与线上双模型进程
盲目并行加载一整套副本。应另安排资源窗口，先获得停服／资源变更确认。

只有验收后才修改实际 `server/.env`：

```dotenv
YIZHU_RECOGNITION_SOLUTION=bioclip-multiview-plantclef
```

再重启 API。回滚可选回原 `bioclip-v2-rerank`；本次未执行切换或重启。

## 验证状态

Ruff 通过，完整后端 **234 项测试通过**，包括：
候选保留／去重／上限、拒识短路、第二级调用、未映射首选、隐藏物种、
两级均不确定、分数错误、文件完整性、EMA 安全转换、缓存持久化与失效、
方案选择、离线入口、手账持久化及用户纠正不覆盖原始结果。

测试使用合成模型输出、合成小 checkpoint、临时图库和账号。
另外在现有 timm 中实际构造了随机初始化架构：92,584,830 个参数，
head 为 7,806×768、输入为 518×518；没有加载正式训练权重，
此项只验证架构构造，不是识别效果或权重兼容验收。
尚未完成正式 PlantCLEF 权重加载、全流程真实图片推理、准确率／并发／延迟评估；
没有以单测通过替代这些验收。客户端、数据库和实际线上配置未修改。
