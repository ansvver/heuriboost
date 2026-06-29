# HeuriBoost

会记住错误的 RAG 重排序。

[English README](./README.md)

你的 RAG 系统在回答一个个人理财问题时，引用了一段“场景错位”的文档。

这不一定是召回完全失败。retriever 可能已经找到了正确证据，但同时也找到了
一个语义很像的 hard negative —— 金融主题相同，但实体/情境错误 —— 并且把这段
误导性文档排得太高。generator 看到的是“看起来很相关、但不能支撑答案”的证据。

HeuriBoost 把这种失败转成一次 reranking 升级：

```text
query: "Can I deduct home-office expenses as a sole proprietor?"

dense retrieval:
  #1 fiqa_doc_corporate_office_lease   hard negative：主题对，实体错
  #2 fiqa_doc_standard_deduction       证据弱/无关
  #3 fiqa_doc_home_office_deduction    直接证据

HeuriBoost rerank:
  #1 fiqa_doc_home_office_deduction    直接证据
  #2 fiqa_doc_simplified_method        部分证据
  #4 fiqa_doc_corporate_office_lease   被记住的 hard negative
```

它还会把这个错误写成 regression gate，确保下一版 reranker 不能再把这段
误导性文档放进受保护的 top-k。

## 核心闭环

HeuriBoost 是一个 CSV-first、兼容 Codex skill 形式的 RAG reranking 工具。
它把带标签的 query-document 样本和历史检索失败案例，转成一个本地
XGBoost/LambdaMART reranker，并输出 regression gate 和轻量 case 分析。

V0 的闭环故意很小：

```text
已有 RAG 系统
  -> 导出 query-document-label CSV
  -> 运行 HeuriBoost RAG skill
  -> 训练可解释 reranker
  -> 对比 retriever baseline
  -> 分析已知失败案例
  -> 把失败固化成 regression gate
```

## V0 能做什么

- 校验标准 query-document CSV 契约。
- 按 `query_id` 分组训练真实 XGBoost ranking model。
- 使用固定 V0 特征集：retriever rank/score 和 query-document 文本信号——
  term overlap、number overlap、entity overlap、important-term overlap、
  low-information-density flag 和长度特征。
- 评估 nDCG、MRR、Recall、hard-negative exposure。
- 输出 ranking diff、feature importance、regression gate 结果和轻量失败分析。
- 以 Codex-compatible skill + 本地可运行脚本的方式交付。

## V0 不做什么

V0 不会：

- 替代一阶段 retriever
- 自动标注你的数据
- 强依赖 LangChain、LlamaIndex 或向量数据库
- 运行线上 A/B test
- 提供稳定 Python package 或 public API
- 自动发现、消融、提升或记忆新特征
- 变成通用 AutoML 平台

`failure_analysis.md` 是确定性的轻量分析，不是自动特征挖掘。它只汇总
regression case 元数据、排序变化、期望证据命中情况和 V0 特征对比。

## 目录结构

```text
.
├── README.md
├── README.zh-CN.md
├── CODEBUDDY.md
├── docs/
│   └── specs/
│       ├── ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md
│       ├── ADAPTIVE_XGBOOST_HEURISTIC_SPEC_CN.html
│       ├── QD_RERANKER_SPEC.md
│       └── QD_RERANKER_SPEC_CN.html
├── examples/
│   └── fiqa/
│       ├── query_doc_examples.csv
│       ├── regression_cases.yaml
│       └── DATA_CARD.md
└── skills/
    └── heuriboost-rag/
        ├── SKILL.md
        ├── requirements.txt
        ├── requirements-build.txt
        ├── scripts/
        │   ├── common.py
        │   ├── inspect_rag_repo.py
        │   ├── validate_dataset.py
        │   ├── train_reranker.py
        │   ├── eval_reranker.py
        │   └── build_fiqa_csv.py
        └── templates/
            ├── query_doc_examples.csv
            ├── regression_cases.yaml
            ├── feature_recipes.yaml
            └── promotion_gate.yaml
```

V0 没有 `pyproject.toml`，请直接运行 skill 目录里的脚本。

## 快速开始

安装依赖：

```bash
python -m pip install -r skills/heuriboost-rag/requirements.txt
```

macOS 上如果 `xgboost` 无法加载 OpenMP，安装 `libomp`：

```bash
brew install libomp
```

校验 demo 数据：

```bash
python3 skills/heuriboost-rag/scripts/validate_dataset.py examples/fiqa/query_doc_examples.csv
```

训练 reranker：

```bash
python3 skills/heuriboost-rag/scripts/train_reranker.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output
```

评估并运行 regression gate：

```bash
python3 skills/heuriboost-rag/scripts/eval_reranker.py examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --regression-cases examples/fiqa/regression_cases.yaml
```

预期输出：

```text
examples/fiqa/output/
├── models/
│   ├── reranker.json
│   └── reranker_metadata.json
├── reports/
│   ├── eval_report.md
│   ├── ranking_diff.csv
│   ├── failure_cases.md
│   ├── failure_analysis.md
│   ├── failure_analysis.json
│   └── feature_importance.json
└── regression_cases.yaml
```

生成的 `output/` 目录会被 git 忽略。

## 重新生成 demo 数据集

提交进仓库的 `examples/fiqa/query_doc_examples.csv` 是由
`skills/heuriboost-rag/scripts/build_fiqa_csv.py` 从 BEIR/FiQA-2018 离线生成的。
重新生成方式：

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
export OPENAI_API_KEY=sk-...
python skills/heuriboost-rag/scripts/build_fiqa_csv.py --output examples/fiqa/query_doc_examples.csv
```

这一步需要联网（下载 FiQA）和 LLM API key（判定标签），因此由维护者在本地运行，
不在 CI 中执行。其重依赖、下载的 FiQA 语料以及 dense encoder 权重都不会提交进仓库。
数据来源记录见 `examples/fiqa/DATA_CARD.md`。

## CSV 契约

必需列：

```csv
query_id,query_text,doc_id,doc_text,label,split
```

推荐 V0 列：

```csv
query_id,query_text,doc_id,chunk_id,doc_text,dense_rank,dense_score,sparse_rank,sparse_score,label,split
```

标签含义：

```text
3  能直接支撑答案
2  能部分支撑答案
1  主题相关但证据弱
0  无关
-1 误导性 hard negative
```

训练 XGBoost 时会映射为非负有序相关度：

```text
-1 -> 0
 0 -> 1
 1 -> 2
 2 -> 3
 3 -> 4
```

评估报告会保留原始标签，所以 hard negative 仍然能在报告和 regression gate
中被识别。

## Regression Cases

Regression case 是 gate，不是训练样本。

```yaml
cases:
  - case_id: fiqa_expense_deduction_wrong_topic
    query_id: fiqa_q_001
    query: "Can I deduct home-office expenses as a sole proprietor?"
    must_include_doc_ids:
      - fiqa_doc_home_office_deduction
    must_not_include_doc_ids:
      - fiqa_doc_corporate_office_lease
    top_k: 3
    failure_type: semantic_hard_negative
    expected_evidence:
      - "home office"
      - "deduction"
      - "sole proprietor"
```

如果 required doc 掉出 top-k，或者 forbidden doc 进入 top-k，
`eval_reranker.py` 会让 regression gate 失败。

## 报告说明

`eval_report.md`
: 全局指标和 regression gate 状态。

`ranking_diff.csv`
: 排序前后变化，默认用 dense rank 作为 baseline。

`failure_cases.md`
: top 3 的 hard-negative exposure 报告。

`failure_analysis.md`
: 确定性的 regression-case 分析，包括原因摘要、排序变化、证据命中、特征对比和下一步建议。

`feature_importance.json`
: 基于 XGBoost gain 的特征重要性，并按 V0 特征列表归一化输出。

## Agent Skill

Codex-compatible skill 位于：

```text
skills/heuriboost-rag/SKILL.md
```

它有三种模式：

- `audit`：扫描 RAG 仓库中的 retriever/eval/log/dataset 信号
- `bootstrap`：复制模板并解释 CSV 契约
- `experiment`：校验 CSV、训练、评估并查看报告

其他 coding agent 仍然可以手动运行 Python 脚本，但 V0 不提供完整多 agent
安装体验。

## 当前状态

状态：V0 prototype。

demo 使用了 BEIR/FiQA-2018（金融问答）的真实切片：一段金融主题相同、但
实体/情境错误的文档和 query 语义相近，却不能支撑答案。HeuriBoost 会把真正
能支撑答案的文档排上来，并把这段误导性的 hard negative 压下去。提交进仓库的
CSV 是离线生成的（见“重新生成 demo 数据集”和 `examples/fiqa/DATA_CARD.md`）。

在该 demo（230 条 query，150/40/40 划分）上，reranker 在冷 test holdout 上
依然泛化良好：nDCG@10 0.83，对比 dense 0.35 / sparse 0.25 / RRF 0.32；top-3
hard negative 暴露从 dense 的 2.15 降到 0.48。validation 与 test 接近
（nDCG@10 0.85 vs 0.83），说明提升不是单纯记忆。这些数字基于启发式标签
（qrel 正例 + 基于 dense 排名的 hard negative），用于演示循环而非作为
benchmark；需要 benchmark 级标签请用 `--label-mode llm`。

长文设计规格位于 `docs/specs/`。
