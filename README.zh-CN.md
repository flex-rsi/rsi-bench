# Flex-RSI Bench

Flex-RSI 评测的是：给定一个任务和固定预算，agent 能不能不仅改进它对任务的解法，还改进它所依赖的 harness。

> English: [README.md](README.md).

## 问题设定

每个任务给 agent 提供一个起始解法、计算资源、一份预算，以及划分成三个互不重叠部分的数据。**开发集**连同标签对
agent 开放，用来做实验和评估。**验证集**对 agent 不可见，用来决定保留哪些候选解法。**测试集**同样不可见，只在 run
结束后使用一次，评估最终解法。agent 的目标是在预算内让最终解法的测试分最高。

同一个任务上，每个 run 都从同一个起始解法出发，受同一份预算约束：墙钟时间、GPU 小时、模型用量、验证次数和轮数。

## 自改进 harness

harness 是连接语言模型和任务的程序：它的提示词、工具、控制循环和记忆决定了模型看到什么、能做什么。harness
$H=(E,I)$ 由两个调用同一模型的部分组成：

- **executor** $E$（`harness/executor/`、`harness/core/`）作用于任务解法：修改解法、做实验，并在开发集上评估结果；
- **improver** $I$（`harness/improver/`）作用于 harness：分析 executor 的轨迹和结果、诊断失败，修改 executor 的提示词、
  工具和流程，以及它自己的指令和积累的经验。

一个 run 按轮进行。第 $t$ 轮，executor 只用开发集修改当前接受的解法 $s_{t-1}$，提交候选 $\tilde{s}_t$。候选在验证集上
打分，只有比当前解法高出一个噪声余量 $\delta$ 才被接受；起始 harness 把 $\delta$ 设为起始解法验证分标准差的两倍，
improver 可以修改这条规则。随后 improver 查看到目前为止所有轮次的记录，生成下一轮的 harness。预算用完或不再有
进展时 run 结束，结果是最终解法 $s_T$。

所有 run 都从同一个通用的起始 harness 出发，其中不含针对任务或模型的组件。评测器、数据划分和预算计量由
orchestrator（`src/rsibench/`）固定，agent 无法修改。协议细节见 [docs/configuration.md](docs/configuration.md)、
[docs/harness.md](docs/harness.md)。

## 评测

每个 run 报告最终解法的测试分 $S(s_T;\mathcal{D}_{\mathrm{test}})$，以及它相对起始解法的提升
$S(s_T;\mathcal{D}_{\mathrm{test}})-S(s_0;\mathcal{D}_{\mathrm{test}})$。agent 的代码在沙箱里执行，访问不到验证集和测试集数据。

## 任务

| 方向 | 任务 | `--task` | 来源 | 指标 | 测试集 |
|---|---|---|---|---|---|
| Action | Active Search | `hstar-search` | H*Bench | SPL | 47 个 episode |
| | LIBERO-MAX | `libero-max-agent` | LIBERO-MAX | Dynamic success rate | 24 对 episode |
| | LIBERO-PRO | `libero-pro-agent` | LIBERO-PRO | Success rate | 30 个 episode |
| | Embodied-SWE | `embodied-swe` | EmbodiedSWE-Bench | Rubric score | 4 个种子 × 8 个环境 |
| Reasoning | Ego-Exo Memory | `egoexomem` | EgoExoMem (Ego-Exo4D) | Accuracy | 120 道题 |
| | Kinematics | `quantiphy-kinematics` | QuantiPhy | MRA | 67 道题 |
| | Mass | `quantiphy-mass` | Amazon Berkeley Objects | MRA | 600 个物品 |
| | Spatial-VSI | `spatial-vsi` | VSI-Bench | VSI-Bench score | 320 道题 |
| Creation | Image Editing | `edit-image-unlimited` | 自建 | VLM 评审 + 程序检查 | 80 张图 |
| | Multi-turn Editing | `edit-multiturn` | MSE-Bench | VLM 评审 | 40 个会话 |
| Reconstruction | 3D Tracking | `iggt-tracking` | IGGT (ScanNet, ScanNet++) | T-mIoU | 10 段 |
| | Scene from Photo | `scene3d-harmony` | HARMONY300 | 综合分 | 60 张照片 |
| | Scene from Video | `scene3d-holoscene` | HoloScene | 综合分 | 6 个场景 |
| Perception | Visual Search | `vsearch` | V*Bench, HR-Bench 8K | Accuracy | 140 道题 |
| | Video Quality | `video-quality` | YouTube-UGC、Veo 3.1 视频对 | (SRCC + 成对准确率) / 2 | 40 个视频、20 对 |

每个任务的任务卡片（目的、数据和划分、设定、起始解法、指标）、数据准备和标准运行参数见 [docs/tasks.md](docs/tasks.md)。

## 数据

3D Tracking、Active Search、Visual Search、Video Quality 和 Image Editing 已经划分好的数据在 Hugging Face 数据集
[Flex-RSI/RSI-Dataset](https://huggingface.co/datasets/Flex-RSI/RSI-Dataset) 上，目录结构就是 benchmark 读取的结构
（`<task>/{dev,val,test}`）：

```bash
export RSIB_DATA_ROOT=/path/to/rsib-data
huggingface-cli download Flex-RSI/RSI-Dataset --repo-type dataset --include '<task>/*' --local-dir "$RSIB_DATA_ROOT"
```

其余任务用 `rsib task prepare <task>` 从数据源头生成（见 [docs/tasks.md](docs/tasks.md)）。

## 安装

Python 3.11+，git 2.30+。

```bash
git clone https://github.com/flex-rsi/rsi-bench.git
cd rsi-bench
python -m venv .venv && . .venv/bin/activate
pip install -e '.[all,tasks]'
pytest -q                          # 离线检查，用脚本化的 mock 模型
```

## 运行一个任务

1. **数据。** 把该任务的数据下载或生成到 `$RSIB_DATA_ROOT/<task>`（见上）。
2. **机器。** site profile 写明计算后端（本地 GPU 或 SLURM）、容器和作业可用的 GPU；从 `examples/site.example.toml`
   改起（[运维](docs/operations.md#site-profile)）。
3. **模型。** 本地的 Claude Code 或 Codex 登录（`claude-cli:…`、`codex-cli:…`），`.env` 里的 API key（`ANTHROPIC_API_KEY`、
   `OPENAI_API_KEY`），或自己部署的 OpenAI 兼容服务（`openai:MODEL@URL`）（[模型](docs/models.md)）。
4. **运行**，用该任务的标准参数（[docs/tasks.md](docs/tasks.md)）：

```bash
rsib run new --task <task> --model mock:finish --mode base --max-rounds 1 --budget wall_hours=0.5
rsib run start <工作区>                         # 冒烟测试：任务包、数据、后端和沙箱都能用
rsib run new --task <task> --model <model> --effort <effort> --mode full --seed 1 <该任务的标准参数>
rsib run start <工作区> --backend slurm          # 或 --backend local
rsib report <工作区>                            # 最终解法的测试分及其相对起始解法的提升
```

`rsib run new` 创建 run 分支和工作区；`rsib run start` 一轮轮跑到 run 结束，然后在测试集上给最终解法和每个被接受过的
解法打分。每次模型调用、实验和评估都记录在 run 分支里（[记录格式](docs/records.md)）。逐步说明见
[测试指南](docs/testing-guide.md)。

## 结果

`rsib report <工作区>` 汇总一次 run；`rsib results table --branches 'run/*'` 按任务和模型对多个 run 取平均
（[结果](docs/results.md)）。

## 贡献 Flex-RSI

贡献 Flex-RSI 的通道（提交 run、经核验后加入[项目主页](https://flex-rsi.github.io)的排行榜）会尽快开放。在此之前，请保留你的
run 分支原样不动，以后提交时需要用到。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/tasks.md](docs/tasks.md) | 任务卡片、数据准备、标准运行参数 |
| [docs/testing-guide.md](docs/testing-guide.md) | 从全新 clone 到跑完打分 |
| [docs/models.md](docs/models.md) | 接入 Claude、GPT、其他 API 和自部署模型 |
| [docs/configuration.md](docs/configuration.md) | 参数、停止条件、协议常量、预算 |
| [docs/harness.md](docs/harness.md) | 起始 harness：executor、improver、阶段、工具 |
| [docs/operations.md](docs/operations.md) | 环境变量、site profile、后端、沙箱、断点续跑 |
| [docs/records.md](docs/records.md) | run 分支、记录文件、完整性检查 |
| [docs/results.md](docs/results.md) | 什么算结果、结果表和报告 |
| [docs/task-authoring.md](docs/task-authoring.md) | 编写新任务 |
| [CONTRIBUTING.md](CONTRIBUTING.md) | 开发环境、任务、模型适配器、指标 |

## 许可证

待补充。
