# 开源植物识别选型

调研日期：2026-09-20。后续已获用户同意，完成 BioCLIP v1 的 CPU 安装、
权重校验和小规模离线实测；记录见 `recognition-baseline.md`。尚未接入小程序真实识别流程。

## 建议

采用 **BioCLIP 系列 + pybioclip**，先以 BioCLIP v1 建立资源较小的基线，再比较 BioCLIP 2。
这是最初根据当前机器与项目需求提出的工程选择。后续实测表明 v1 可在本机运行，
但初始化慢、合成负例会被强行分类，尚不满足直接上线条件。

| 方案 | 已核实内容 | 对一株的判断 |
| --- | --- | --- |
| BioCLIP v1 | ViT-B/16；代码及官方模型卡标注 MIT；基于带分类标签的生物图像训练，可做零样本物种分类 [1][2] | 优先做本机基线，不从零训练 |
| BioCLIP 2 | ViT-L/14；代码及模型卡标注 MIT；官方提供预训练权重和物种分类用法 [3][4] | 作为质量比较对象，不能把论文指标当作中国植物实拍准确率 |
| Pl@ntNet-300K | 306,146 张图、1,081 种植物，公开 ResNet 等预训练模型；代码 BSD-2-Clause [6][7] | 可作植物专用对照；候选类别受训练标签限制，不是完整 Pl@ntNet 产品的开源替身 |

BioCLIP 2 仓库还列出 2026 年 2 月发布的 BioCLIP 2.5 Huge（ViT-H/14）[3]。
本轮未单独审查 2.5 模型卡、权重许可与部署开销，不作为首个落地目标。

pybioclip 提供 Python/CLI 接口，许可 MIT，当前 README 指明默认模型为 BioCLIP 2；
部署基线时必须显式选择 v1，不能依赖默认值。其 pyproject 声明 Python >=3.10，
列出 3.13，依赖 torch、torchvision、open_clip_torch 等 [5]。
本项目 Python >=3.13 的实际依赖组合仍需解析、锁定和测试。

## 许可与限制

- 模型卡和代码许可均需固定到实际采用的 revision，保存许可与来源；训练集图片另有来源和许可，不能据此打包搬运训练照片。
- BioCLIP v1 模型卡还表达希望衍生产品开源的作者请求；不要遗漏这一说明，也不要把它误写为另一套已核实的许可证。
- Pl@ntNet 仓库链接了公开 checkpoint，但本轮未核实下载站上的独立权重许可；BSD 代码许可不能代替该检查。
- 两代 BioCLIP 不是专门针对中文植物的完整知识库。中文名拟通过有出处的学名映射补充，查不到就保留学名。
- 不把相似度或候选集合内的分数写成“识别正确概率”，也不宣称固定准确率。不提供食用、药用、安全或树龄保证。

## 本机与成本

本轮实际运行 `nproc`、`free -h`：4 个可用 CPU、7.6 GiB 总内存。
未安装 `nvidia-smi`；这不足以证明没有 GPU，需另行确认。
后续单进程实测：图片推理约 0.83 秒，初始化约 346 秒，进程峰值 RSS 约 1.53 GiB。
这不包括上传、排队和存储，未测并发能力，不承诺小程序端到端秒级响应。

建议先单进程、单任务推理，复用模型，预计算经过筛选的植物标签向量；
不要直接加载所有生物标签或在每个 Web worker 复制模型。
自托管仍消耗内存、CPU/电力、磁盘和带宽，不等于零运营成本。
本机直连 Hugging Face 模型卡超时，最终经现有公网主机读取官方文本；
后续权重下载连通性也要验证，不临时依赖第三方推理 API。

## 最小验证与接入顺序

1. 固定官方包与模型 revision，核对具体下载量；不下载训练集、不从头训练。
2. 用 30–50 张有可靠标签、许可明确的本地植物照片做初筛，包含叶、花、树干、模糊图及非植物负例。记录候选命中、无法确定场景、首次加载/热推理耗时与峰值内存；该小样本不是正式准确率证明。
3. 达到可接受的质量和延迟后，接入既有 Flask / zdog-auth；不新增复杂微服务。照片只发给自己的服务，需在上传前明示，去除元数据并限制大小、像素与格式。
4. 计划实现后端 RecognitionProvider 与鉴权上传接口，输出候选、模型 revision 和原始分数；未经校准的 `confidence` 保持 null。低确定性结果保留建档路径。
5. 现有 `MockJournal` 会拒绝非演示 Provider，不能只替换识别类就声称真实链路完成。需同步调整照片 URI、物种数据、任务错误/重试、上传隐私文案和结果存储；物种判断仍不能自动关联植物个体。

## 官方来源

网页检索工具未返回可用正文；以下通过直接读取官方仓库和官方模型卡核实，未依据第三方推荐文章。

1. BioCLIP 模型卡：`https://huggingface.co/imageomics/bioclip`
2. BioCLIP README / LICENSE：`https://github.com/Imageomics/bioclip`
3. BioCLIP 2 README / LICENSE：`https://github.com/Imageomics/bioclip-2`
4. BioCLIP 2 模型卡：`https://huggingface.co/imageomics/bioclip-2`
5. pybioclip README / pyproject：`https://github.com/Imageomics/pybioclip`
6. Pl@ntNet-300K README：`https://github.com/plantnet/PlantNet-300K`
7. Pl@ntNet-300K LICENSE：`https://github.com/plantnet/PlantNet-300K/blob/main/LICENSE`
