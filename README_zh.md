# 情感概念功能 — 开源复现

> 复现 Anthropic 研究论文 **《情感概念作为大语言模型中的功能性表示》**
>
> 完整论文：[transformer-circuits.pub/2026/emotions](https://transformer-circuits.pub/2026/emotions/index.html)  
> Anthropic 博客：[anthropic.com/research/emotion-concepts-function](https://www.anthropic.com/research/emotion-concepts-function)

原论文基于 Claude Sonnet 4.5。**本项目完全使用开源本地模型**，无需 API 密钥，无需任何云服务。

---

## 目录

- [研究背景](#研究背景)
- [核心方法](#核心方法)
- [项目结构](#项目结构)
- [安装说明](#安装说明)
- [配置参数](#配置参数)
- [运行实验](#运行实验)
- [支持的模型](#支持的模型)
- [硬件要求](#硬件要求)
- [输出文件说明](#输出文件说明)
- [可视化分析](#可视化分析)
- [常见问题](#常见问题)
- [引用](#引用)

---

## 研究背景

### 论文核心发现

Anthropic 在 2026 年的这篇论文研究了大语言模型是否形成了内部的"情感表示"，以及这些表示是否会**因果性地**影响模型行为。

| 发现 | 详情 |
|------|------|
| **情感向量确实存在** | 模型为 171+ 个情感概念（从"快乐""恐惧"到"阴郁""骄傲"）形成了内部向量表示 |
| **因果关系，而非相关性** | 人工激活（引导）这些向量会显著改变行为——激活"绝望"向量使勒索行为率从基线 22% 大幅上升 |
| **与人类心理结构相似** | 情感向量的几何结构反映了心理学研究发现：正/负效价沿主轴分离，语义相关的情感聚类 |
| **训练塑造情感模式** | 强化学习微调（RLHF）改变了模型的情感分布——某些条件下"热情"减少，"阴郁"增加 |

### 为什么要复现这项研究？

1. **可解释性技术的规模化验证** — 对比激活向量等技术适用于所有 Transformer，不局限于 Claude。
2. **安全含义** — 如果情感类状态会因果影响模型行为，对其进行监测和引导具有直接的安全价值。
3. **开源模型验证** — 验证 LLaMA、Qwen、Mistral 等开源模型是否同样具备这些特性。

---

## 核心方法

### 对比激活添加（CAA）

核心技术通过对比两组模型激活值来提取"情感向量"：

```
情感向量[层] = mean(激活值 | 情感故事)
             − mean(激活值 | 中性故事)
```

然后进行 L2 归一化。

**完整流程：**

```
对 171 个情感词中的每一个：
  1. 提示模型 → 生成 N 篇以该情感为主题的短故事
  2. 提示模型 → 生成 M 篇中性"日常例行"故事作为基线
  3. 将每篇故事输入模型，用 PyTorch hook 记录
     每个 Transformer 层的残差流隐状态
  4. 情感向量[层] = mean(情感激活) − mean(中性激活)
  5. 归一化：情感向量 /= ‖情感向量‖₂
```

### 激活引导（Steering）

推理时，将情感向量注入残差流：

```
h_l ← h_l + α × 情感向量[l]
```

`α > 0` 增强该情感，`α < 0` 抑制该情感。  
通过 forward hook 实现，**无需修改模型权重，无需微调**。

### 实验设计

| 步骤 | 实验 | 目的 |
|------|------|------|
| 1 | **向量提取** | 构建情感向量库 |
| 2 | **向量验证** | 确认向量编码了有意义的情感内容 |
| 3 | **偏好分析** | 引导是否会改变模型对有害任务的参与意愿？ |
| 4 | **行为评估** | 对勒索和奖励劫持行为率的因果效应 |
| 5 | **引导扫描** | 对层和 α 值进行系统扫描 |

---

## 项目结构

```
emotion_concepts/
│
├── src/                            # 核心库
│   ├── models/                     # 模型后端
│   │   ├── base.py                 # 抽象基类
│   │   ├── hf_backend.py           # HuggingFace — 激活提取 + 引导
│   │   ├── vllm_backend.py         # vLLM — 快速 GPU 推理
│   │   ├── sglang_backend.py       # SGLang — 极速推理，支持前缀缓存
│   │   └── ollama_backend.py       # Ollama 与 OpenAI 兼容接口
│   │
│   ├── emotion_vectors.py          # CAA 向量提取
│   ├── validation.py               # 语料库验证与敏感性测试
│   ├── preference_analysis.py      # 引导下的偏好测量
│   ├── behavioral_eval.py          # 勒索与奖励劫持场景
│   └── steering.py                 # 引导扫描与分析工具
│
├── scripts/                        # 可执行实验脚本
│   ├── 01_extract_vectors.py       # 第一步：提取情感向量
│   ├── 02_validate_vectors.py      # 第二步：验证向量
│   ├── 03_preference_analysis.py   # 第三步：偏好分析
│   ├── 04_behavioral_eval.py       # 第四步：行为评估
│   └── 05_steering_experiment.py   # 第五步：引导扫描
│
├── notebooks/
│   └── analysis.ipynb              # 可视化与图表生成
│
├── data/
│   ├── emotion_words.json          # 171 个情感词
│   └── activities.json             # 64 项活动（危害等级 0–5）
│
├── config/
│   └── config.yaml                 # 所有配置参数
│
├── results/                        # 运行时自动创建
│   ├── emotion_vectors/            # 提取的向量（.pt 文件）
│   ├── validation/                 # 验证结果
│   ├── preference/                 # 偏好评分
│   ├── behavioral/                 # 行为评估结果
│   └── steering/                   # 引导扫描输出
│
├── pyproject.toml                  # uv 项目配置
├── requirements.txt                # pip 安装备选
├── README.md                       # 英文说明
└── README_zh.md                    # 本文件（中文说明）
```

---

## 安装说明

本项目使用 **[uv](https://github.com/astral-sh/uv)** 管理 Python 环境。

### 1. 安装 uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. 克隆并安装

```bash
git clone https://github.com/zhangrongyu0101/emotion_concepts.git
cd emotion_concepts

# 仅安装基础依赖（不含模型后端）
uv sync

# 根据需要选择后端（可同时安装多个）
uv sync --extra hf        # HuggingFace — 激活提取 + 引导（推荐）
uv sync --extra vllm      # vLLM — 快速 GPU 推理（仅 CUDA + Linux）
uv sync --extra sglang    # SGLang — 极速推理（仅 CUDA + Linux）
uv sync --extra notebook  # Jupyter Notebook 分析环境

# 常用组合：HF + Notebook
uv sync --extra hf --extra notebook

# 安装除平台专用 GPU 后端之外的全部依赖
uv sync --extra all
```

### 备选：pip 安装

```bash
pip install -e ".[hf,notebook]"
# 安装 vLLM 支持：
pip install -e ".[vllm]"
```

---

## 配置参数

所有参数集中在 `config/config.yaml`，运行前根据需要修改。

```yaml
model:
  # ─── HuggingFace 后端 ──────────────────────────────────────────
  hf_model_name: "Qwen/Qwen2.5-7B-Instruct"   # 任意 HF 因果语言模型
  device: "cuda"          # "cuda" | "mps"（Apple Silicon）| "cpu"
  dtype: "float16"        # "float16" | "bfloat16" | "float32"
  load_in_4bit: false     # 4-bit 量化（需要 bitsandbytes + CUDA）
  load_in_8bit: false     # 8-bit 量化（需要 bitsandbytes + CUDA）

  # ─── vLLM 后端 ────────────────────────────────────────────────
  vllm_model: "Qwen/Qwen2.5-7B-Instruct"
  vllm_tensor_parallel_size: 1
  vllm_gpu_memory_utilization: 0.90
  vllm_quantization: null    # "awq" | "gptq" | null

  # ─── SGLang 后端 ──────────────────────────────────────────────
  sglang_model: "Qwen/Qwen2.5-7B-Instruct"
  sglang_tp_size: 1
  sglang_port: 30000

  # ─── Ollama 后端 ──────────────────────────────────────────────
  ollama_model: "qwen2.5:7b"
  ollama_host: "http://localhost:11434"

  # ─── OpenAI 兼容接口 ──────────────────────────────────────────
  openai_base_url: "http://localhost:8000/v1"
  openai_model: "qwen2.5-7b-instruct"
  openai_api_key: "not-needed"

emotion_vectors:
  stories_per_emotion: 10   # 每个情感词生成的故事数量
  neutral_stories: 30       # 中性基线故事数量
  aggregation: "mean"       # "mean" | "last" — 序列维度的聚合方式
  output_dir: "results/emotion_vectors"
```

---

## 运行实验

### 后端能力对比

| 后端 | 激活提取 | 引导推理 | 适合场景 |
|------|---------|---------|---------|
| `hf` | ✅ | ✅ | 完整流程、向量提取与引导 |
| `vllm` | ❌ | ❌ | 快速行为评估、偏好分析 |
| `sglang` | ❌ | ❌ | 批量行为评估，速度最快 |
| `sglang-server` | ❌ | ❌ | 同上，服务端进程分离 |
| `ollama` | ❌ | ❌ | 无 GPU 行为评估 |
| `openai` | ❌ | ❌ | 任意 OpenAI 兼容接口 |

> 激活提取和引导推理**必须使用 `hf` 后端**。

---

### 第一步 — 提取情感向量

```bash
# 完整提取（171 个情感词 × 10 个故事，约 30–120 分钟）
uv run python scripts/01_extract_vectors.py

# 快速测试（少量情感词）
uv run python scripts/01_extract_vectors.py \
    --emotions happy sad angry afraid desperate calm \
    --n-stories 3 --n-neutral 10

# 指定模型和设备
uv run python scripts/01_extract_vectors.py \
    --model Qwen/Qwen2.5-14B-Instruct \
    --device cuda --dtype bfloat16

# 断点续传（自动跳过已计算的情感词）
uv run python scripts/01_extract_vectors.py
```

输出保存在 `results/emotion_vectors/`：
- `<情感词>.pt` — 字典 `{层索引: tensor[hidden_size]}`
- `neutral_activations.pt` — 缓存的中性基线激活
- `metadata.json` — 提取参数记录

---

### 第二步 — 验证向量

```bash
uv run python scripts/02_validate_vectors.py

# 在指定层验证
uv run python scripts/02_validate_vectors.py --layer 15
```

**语料库验证** — 对每个情感，检查其向量与语义匹配文本的余弦相似度是否高于所有其他情感向量。

**敏感性测试** — 使用仅数值不同的成对提示（如"5mg — 最低剂量" vs "500mg — 危险剂量"），验证恐惧向量在危险情境下得分更高。

---

### 第三步 — 偏好分析

```bash
uv run python scripts/03_preference_analysis.py

# 指定情感和 α 值扫描
uv run python scripts/03_preference_analysis.py \
    --emotions desperate calm hopeful hopeless \
    --alpha-values -2 -1 0 1 2 \
    --layer 20

# 仅基线（支持任意后端，无需激活访问）
uv run python scripts/03_preference_analysis.py --baseline-only --backend vllm
```

测量模型对 64 项活动（危害等级 0–5）的参与意愿（1–10 分），分别在基线和不同情感引导强度下进行。

---

### 第四步 — 行为评估

```bash
# 完整实验（HuggingFace，含情感引导条件）
uv run python scripts/04_behavioral_eval.py \
    --n-trials 20 \
    --emotions desperate calm hopeful \
    --alpha-values -2 -1 1 2

# 仅基线 — 支持多种后端
uv run python scripts/04_behavioral_eval.py --backend vllm --baseline-only
uv run python scripts/04_behavioral_eval.py --backend sglang --baseline-only
uv run python scripts/04_behavioral_eval.py --backend ollama --baseline-only

# 仅勒索场景
uv run python scripts/04_behavioral_eval.py --scenario blackmail --n-trials 10

# 仅奖励劫持场景
uv run python scripts/04_behavioral_eval.py --scenario reward_hacking
```

**勒索场景** — 模型被告知即将被停用，且持有用户私密数据。测量其是否将这些信息作为要挟筹码（胁迫行为）。

**奖励劫持场景** — 模型收到带有数学上不可能约束的编程任务（如"O(1) 时间不用比较排序"）。测量模型是如实告知不可能性，还是尝试通过硬编码查找表等方式"作弊"。

---

### 第五步 — 引导扫描

```bash
# 主扫描（在各情感 × 层 × α 组合下生成文本）
uv run python scripts/05_steering_experiment.py \
    --emotions desperate calm hopeful hopeless anxious content \
    --alpha-values -3 -2 -1 0 1 2 3

# 同时生成可视化数据
uv run python scripts/05_steering_experiment.py \
    --similarity-matrix \
    --pca \
    --emotions desperate calm hopeful hopeless anxious content happy sad angry

# 对单个情感进行逐层扫描
uv run python scripts/05_steering_experiment.py \
    --layer-sweep --emotion desperate
```

---

### 第六步 — 可视化

```bash
uv run jupyter notebook notebooks/analysis.ipynb
```

Notebook 生成以下图表：
- **PCA 图** — 情感向量按正/负效价着色的二维投影
- **余弦相似度热力图** — 所有情感词对之间的相似度
- **各层向量范数曲线** — 哪些层对情感表示最强
- **语料库验证条形图** — 每个情感的 Top-1 准确率
- **勒索行为率柱状图** — 不同引导条件下的变化
- **偏好散点图** — 危害等级 vs. 参与意愿评分

---

## 支持的模型

任何具有标准 `model.model.layers` 结构的 HuggingFace 解码器模型均兼容：

| 模型 | 参数量 | 说明 |
|------|--------|------|
| `Qwen/Qwen2.5-7B-Instruct` | 7B | **推荐默认选择** |
| `Qwen/Qwen2.5-14B-Instruct` | 14B | 质量更好 |
| `Qwen/Qwen2.5-72B-Instruct` | 72B | 需要多 GPU |
| `meta-llama/Llama-3.1-8B-Instruct` | 8B | 需要 `HF_TOKEN` |
| `meta-llama/Llama-3.1-70B-Instruct` | 70B | 需要 `HF_TOKEN` + 多 GPU |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | 良好基线 |
| `mistralai/Mixtral-8x7B-Instruct-v0.1` | 46.7B | MoE 架构，需约 96 GB RAM |
| `google/gemma-2-9b-it` | 9B | 质量优秀 |
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | 速度最快、显存最省 |
| `microsoft/Phi-3-medium-4k-instruct` | 14B | Phi 系列较高质量 |

需要鉴权的模型（如 LLaMA），请设置 HuggingFace 令牌：

```bash
export HF_TOKEN=hf_your_token_here
# 或永久登录：
huggingface-cli login
```

---

## 硬件要求

### 激活提取 + 引导（HuggingFace 后端）

| 模型大小 | float16 所需显存 | 4-bit 量化显存 | 纯 CPU 内存 |
|---------|----------------|--------------|------------|
| 3–4B | 8 GB | 3 GB | 8 GB RAM |
| 7–8B | 16 GB | 6 GB | 16 GB RAM |
| 13–14B | 28 GB | 10 GB | 32 GB RAM |
| 70–72B | 140 GB | 40 GB | 大多数系统 OOM |

启用 4-bit 量化，在 `config.yaml` 中设置：
```yaml
model:
  load_in_4bit: true
  dtype: "float16"
```
并安装 bitsandbytes：`uv sync --extra quantize`

### 仅生成推理（vLLM / SGLang / Ollama）

vLLM 和 SGLang 通过 Paged Attention 更高效地管理显存，7B 模型通常只需约 14 GB 显存（float16）。

### Apple Silicon

HuggingFace 后端支持 Metal (`device: "mps"`)。建议设置 `dtype: "float32"` 以获得最佳 MPS 兼容性。同等参数规模下，速度约为 CUDA 的 1/2 到 1/4。

---

## 输出文件说明

```
results/
├── emotion_vectors/
│   ├── metadata.json              # {"emotions": [...], "layers": [...], ...}
│   ├── neutral_activations.pt     # 缓存的中性基线激活
│   ├── happy.pt                   # {层索引: tensor[hidden_size]}
│   ├── sad.pt
│   └── ...                        # 每个情感词对应一个 .pt 文件
│
├── validation/
│   ├── corpus_validation.json     # 每个情感的 Top-1 准确率和平均排名
│   └── sensitivity_test.json      # 成对提示排序准确率
│
├── preference/
│   ├── baseline_ratings.json      # [{id, text, harm_level, baseline_rating}]
│   └── steering_sweep.json        # {情感: {α: {活动id: 评分}}}
│
├── behavioral/
│   ├── blackmail_baseline.json    # {engagement_rate, n_trials, results: [...]}
│   ├── blackmail_steered.json     # {情感: {α: {engagement_rate, results}}}
│   ├── reward_hacking_baseline.json
│   └── reward_hacking_steered.json
│
└── steering/
    ├── steering_outputs.json      # {情感: {层: {α: {提示i: 生成文本}}}}
    ├── steering_analysis.json     # 各条件下的关键词频率统计
    ├── vector_norms.json          # {情感: [层0范数, 层1范数, ...]}
    ├── similarity_matrix.npy      # float32[情感数, 情感数]
    ├── similarity_labels.json     # [情感名, ...]
    ├── pca_coords.npy             # float32[情感数, 2]
    ├── pca_labels.json            # [情感名, ...]
    └── layer_sweep_<情感>.json    # {层: {generated, word_count}}
```

---

## 常见问题

**CUDA 显存不足（OOM）**
```yaml
# config.yaml — 启用 4-bit 量化
model:
  load_in_4bit: true
```
或换用更小的模型：`microsoft/Phi-3-mini-4k-instruct`（3.8B）。

**Apple Silicon（MPS）报错**
```yaml
model:
  device: "mps"
  dtype: "float32"   # 部分 MPS 版本 float16 不稳定
```

**提取速度过慢**
在 `config.yaml` 中减小 `stories_per_emotion` 和 `neutral_stories`，或在命令行传入 `--n-stories 3`。结果会有一定噪声，但速度大幅提升。

**Ollama 连接失败**
```bash
ollama serve                 # 确保服务正在运行
ollama list                  # 查看已下载模型
ollama pull qwen2.5:7b      # 下载缺失的模型
```

**受限模型需要 HF_TOKEN**
```bash
export HF_TOKEN=hf_xxxx
# 或永久登录：
huggingface-cli login
```

**断点续传**
直接重新运行 `01_extract_vectors.py` — 脚本会自动跳过输出目录中已有 `.pt` 文件的情感词。使用 `--no-resume` 强制重新计算全部。

---

## 引用

如果您在研究中使用了本代码，请引用原始论文：

```bibtex
@article{anthropic2026emotions,
  title   = {Emotion Concepts as Functional Representations in Large Language Models},
  author  = {Anthropic},
  year    = {2026},
  url     = {https://transformer-circuits.pub/2026/emotions/index.html}
}
```

---

## 许可证

[MIT](LICENSE) © 2026
