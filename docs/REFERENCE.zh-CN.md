# HeuriBoost 参考手册

HeuriBoost Q-D reranker 的操作参考。[README](../README.zh-CN.md) 讲故事、概念
和 demo 效果；本文承接维护者日常需要的契约与命令细节。

- [CSV 契约](#csv-契约)
- [标签含义](#标签含义)
- [Regression cases](#regression-cases)
- [跨轮 ledger](#跨轮-ledger)
- [闭环：case_sets 挖掘](#闭环case_sets-挖掘)
- [报告说明](#报告说明)
- [Agent skill](#agent-skill)
- [重新生成 demo 数据集](#重新生成-demo-数据集)

## CSV 契约

必需列：

```csv
query_id,query_text,doc_id,doc_text,label,split
```

推荐列（启用更丰富的特征）：

```csv
query_id,query_text,doc_id,chunk_id,doc_text,dense_rank,dense_score,sparse_rank,sparse_score,label,split
```

行按 `query_id` 分组，绝不跨组打乱 query-document 对。`split` 取
`train` / `validation` / `test` 之一。

## 标签含义

| 标签 | 含义 |
|---:|---|
| `3` | 能直接支撑答案 |
| `2` | 能部分支撑答案 |
| `1` | 主题相关但证据弱 |
| `0` | 无关 |
| `-1` | 误导性 hard negative |

训练 XGBoost 时映射为非负有序相关度（`-1→0, 0→1, 1→2, 2→3, 3→4`）。评估保留
原始标签，hard negative 仍能在报告和 gate 中被识别。

## Regression cases

Regression case 是考题，不是训练样本。每个 case 带 `status`：

| 状态 | 行为 |
|---|---|
| `gate` | 已攻克并冻结。失败即阻断（exit 非零）。 |
| `pending` | 已知待攻击的 gap。评估并报告，失败不阻断。 |
| `retired` | 因漂移失效。不评估，仅留作历史。 |

缺省 `status` 默认为 `gate`（向后兼容）。

可选的 per-case 本地检查：

- `require_rank`（int）：第一个 `must_include` 文档须达到 rank <= 该值。
- `min_ndcg10`（float）：该 query 的 nDCG@10 须 >= 该值。

一个 case **通过** 当且仅当：所有 `must_include` 在 `top_k` 内（且设了
`require_rank` 时第一个达到该名次）且没有 `must_not_include` 进入 `top_k`，
且设了 `min_ndcg10` 时满足。

```yaml
cases:
  - case_id: fiqa_expense_deduction_wrong_topic
    query_id: fiqa_q_001
    status: gate
    require_rank: 3
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

`gate` case 失败时 `eval_reranker.py` exit 非零；`pending` 失败只报告，不改
exit code。

## 跨轮 ledger

`regression_ledger.py` 在提交进仓库的 `examples/fiqa/ledger.json` 中维护跨轮
记忆（受版本管理，不被 gitignore，不自动提交）。每轮评估追加一份快照（全局
指标、per-case 通过/失败、与锚定基线的对比）。锚点是某轮冻结的全局指标，确认
有提升后手动刷新。

```bash
# 评估后设置锚点（手动，一次性或在确认提升时）：
python skills/heuriboost-rag/scripts/regression_ledger.py set-anchor --ledger examples/fiqa/ledger.json

# 打印进度摘要（gate/pending 计数、晋级候选、基线对比行）：
python skills/heuriboost-rag/scripts/regression_ledger.py summary --ledger examples/fiqa/ledger.json

# 把 pending case 晋级为 gate（交互式确认，不自动晋级）：
python skills/heuriboost-rag/scripts/regression_ledger.py promote examples/fiqa/regression_cases.yaml <case_id> --ledger examples/fiqa/ledger.json
```

与锚定基线的对比是**报告，不自动阻断** —— 晋级永远是人工决定。`eval_reranker.py`
加 `--no-ledger` 可跳过 ledger 写入（临时评估用）。

## 闭环：case_sets 挖掘

Pending case 是已知待攻击的 gap。教科书路径是：从语料中挖掘同模式训练样本，
折叠进 train，再评估。Case 本身仍是考题 —— 只有与 case 隔离开的挖掘样本进入
训练。

四步闭环（由维护者手动运行，不自动 promote）：

```bash
# 1. 为所有 pending case 挖掘同模式样本（需要 build 依赖）
python skills/heuriboost-rag/scripts/mine_case_sets.py \
  --dataset examples/fiqa/query_doc_examples.csv \
  --cases examples/fiqa/regression_cases.yaml \
  --out-dir examples/fiqa/case_sets

# 2. 把挖掘样本折叠进 train 重新训练
python skills/heuriboost-rag/scripts/train_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --case-sets examples/fiqa/case_sets \
  --regression-cases examples/fiqa/regression_cases.yaml

# 3. 评估 + 记账（标记该轮使用了 case_sets）
python skills/heuriboost-rag/scripts/eval_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --split validation \
  --regression-cases examples/fiqa/regression_cases.yaml \
  --case-sets-used

# 4. （手动）若 pending case 通过且基线检查 OK，手动 promote
python skills/heuriboost-rag/scripts/regression_ledger.py promote \
  examples/fiqa/regression_cases.yaml <case_id> --ledger examples/fiqa/ledger.json
```

**挖掘规则** = 三个信号取交集：与 case query 的语义相似度
（`all-MiniLM-L6-v2`，top-K）、相同失败形状（hard negative 在
`dense_rank <= --shape-rank`，positive 在 `dense_rank >= --shape-pos-gap`）、
相同 `failure_type`。

**隔离**：挖掘样本的 `query_id` 不能等于任何 case 的 `query_id`，`doc_id` 不能
等于任何 case 的 `must_include`/`must_not_include` doc_id。训练加载时还会再做一次
防御性复检。

`sentence-transformers` 是 build 依赖（`requirements-build.txt`），不是运行时
依赖。挖掘会复用 `examples/fiqa/.cache/query_embeddings.npz`。

> **流水线验证说明**：启发式标签下的攻击结果是流水线验证级别，不是 benchmark。
> 它验证闭环机制是否工作（挖掘 → 训练 → 评估 → promote），而非攻击是否真正让
> pending case 通过。可信的攻击质量需要 LLM 模式标签
>（`build_fiqa_csv.py --label-mode llm`）。

## 报告说明

`eval_reranker.py` 写入 `examples/fiqa/output/reports/`：

| 文件 | 内容 |
|---|---|
| `eval_report.md` | 全局指标 + regression gate 状态（Gates + Pending）。 |
| `ranking_diff.csv` | 排序前后变化（默认以 dense rank 为 baseline）。 |
| `failure_cases.md` | top 3 的 hard-negative exposure 报告。 |
| `failure_analysis.md` | 确定性 regression-case 分析：原因摘要、排序变化、证据命中、特征对比、下一步建议。 |
| `feature_importance.json` | 基于 XGBoost gain 的特征重要性，按特征列表归一化。 |

`failure_analysis.md` 是确定性轻量分析，不是自动特征挖掘。

## Agent skill

Codex-compatible skill 位于 `skills/heuriboost-rag/SKILL.md`，有三种模式：

- `audit` —— 扫描 RAG 仓库的 retriever/eval/log/dataset 信号
- `bootstrap` —— 复制模板并解释 CSV 契约
- `experiment` —— 校验 CSV、训练、评估并查看报告

其他 coding agent 可手动运行 Python 脚本。

## 重新生成 demo 数据集

提交进仓库的 `examples/fiqa/query_doc_examples.csv` 由 `build_fiqa_csv.py` 从
BEIR/FiQA-2018 离线生成。脚本先跑 BM25 + `all-MiniLM-L6-v2` + RRF 检索
（FiQA 不自带候选），再用两种模式之一打标签。

启发式模式 —— 零成本、确定性、无需 LLM（仓库里的 CSV 就是这样生成的）：

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
python skills/heuriboost-rag/scripts/build_fiqa_csv.py \
  --label-mode heuristic --output examples/fiqa/query_doc_examples.csv
```

LLM 模式 —— 通过 OpenAI 兼容 judge 打完整 5 级标签（默认 DeepSeek）：

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
export DEEPSEEK_API_KEY=sk-...   # 或用 OPENAI_API_KEY 并加 --base-url ""
python skills/heuriboost-rag/scripts/build_fiqa_csv.py \
  --label-mode llm --output examples/fiqa/query_doc_examples.csv
```

两种模式都需联网（下载 FiQA）；只有 LLM 模式需要 API key。该步骤在本地运行，
不在 CI 中。重依赖、FiQA 语料、dense encoder 权重都不提交。来源见
`examples/fiqa/DATA_CARD.md`。
