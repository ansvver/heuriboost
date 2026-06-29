# HeuriBoost

会记住错误的 RAG 重排序。

[English README](./README.md)

你的 RAG 系统在回答 2024 Q3 问题时，引用了 2023 Q3 文档。

这不一定是召回完全失败。retriever 可能已经找到了正确证据，但同时也找到了
一个语义很像的 hard negative，并且把这个错误年份的文档排得太高。generator
看到的是“看起来很相关、但不能支撑答案”的证据。

HeuriBoost 把这种失败转成一次 reranking 升级：

```text
query: "What caused 2024 Q3 gross margin decline?"

dense retrieval:
  #1 doc_2023_q3_margin   hard negative：主题对，年份错
  #2 doc_2024_q3_ops      部分证据
  #3 doc_2024_q3_margin   直接证据

HeuriBoost rerank:
  #1 doc_2024_q3_margin   直接证据
  #2 doc_2024_q3_ops      部分证据
  #4 doc_2023_q3_margin   被记住的 hard negative
```

它还会把这个错误写成 regression gate，确保下一版 reranker 不能再把这个错误年份
文档放进受保护的 top-k。

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
- 使用固定 V0 特征集：retriever rank/score 和 query-document 文本重叠。
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
│   └── financial_rag/
│       ├── query_doc_examples.csv
│       └── regression_cases.yaml
└── skills/
    └── heuriboost-rag/
        ├── SKILL.md
        ├── requirements.txt
        ├── scripts/
        │   ├── common.py
        │   ├── inspect_rag_repo.py
        │   ├── validate_dataset.py
        │   ├── train_reranker.py
        │   └── eval_reranker.py
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
python3 skills/heuriboost-rag/scripts/validate_dataset.py examples/financial_rag/query_doc_examples.csv
```

训练 reranker：

```bash
python3 skills/heuriboost-rag/scripts/train_reranker.py examples/financial_rag/query_doc_examples.csv --output-dir examples/financial_rag/output
```

评估并运行 regression gate：

```bash
python3 skills/heuriboost-rag/scripts/eval_reranker.py examples/financial_rag/query_doc_examples.csv --output-dir examples/financial_rag/output --regression-cases examples/financial_rag/regression_cases.yaml
```

预期输出：

```text
examples/financial_rag/output/
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
  - case_id: q_val_margin_2024_q3_wrong_year
    query_id: q_val_margin_2024_q3
    query: "What caused 2024 Q3 gross margin decline?"
    must_include_doc_ids:
      - doc_2024_q3_margin
    must_not_include_doc_ids:
      - doc_2023_q3_margin
    top_k: 3
    failure_type: temporal_hard_negative
    expected_evidence:
      - "2024 Q3"
      - "gross margin"
      - "raw material costs"
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

demo 使用了一个小型 financial RAG 场景：2023 Q3 文档和 2024 Q3 问题语义相近，
但不能支撑 2024 Q3 的答案。HeuriBoost 会把真正的 2024 Q3 证据文档排上来，
并把 wrong-year hard negative 压下去。

长文设计规格位于 `docs/specs/`。
