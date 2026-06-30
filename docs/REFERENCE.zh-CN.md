# HeuriBoost 参考手册

HeuriBoost Q-D reranker 的操作参考。[README](../README.zh-CN.md) 讲故事、概念
和 demo 效果；本文承接维护者日常需要的契约与命令细节。

- [特征 registry](#特征-registry)
- [HPO](#hpo)
- [消融](#消融)
- [候选发现](#候选发现)
- [CSV 契约](#csv-契约)
- [标签含义](#标签含义)
- [Regression cases](#regression-cases)
- [跨轮 ledger](#跨轮-ledger)
- [生产 case 修复](#生产-case-修复)
- [闭环：case_sets 挖掘](#闭环case_sets-挖掘)
- [报告说明](#报告说明)
- [Agent skill](#agent-skill)
- [重新生成 demo 数据集](#重新生成-demo-数据集)

## 特征 registry

每个特征都是声明式的 `FeatureRecipe`，不是散落代码。元数据真源是
`skills/heuriboost-rag/templates/feature_recipes.yaml`；Python 实现在
`skills/heuriboost-rag/scripts/features/`（`registry.py`、`primitives.py`、
`recipes.py`）。

每个 recipe 携带规格 §6.4 必填字段：

| 字段 | 含义 |
|---|---|
| `name`、`version` | 特征标识 + 单特征版本 |
| `description` | 人类可读说明 |
| `task_profiles` | 使用它的 profile（V0：`qd_reranker`） |
| `inputs` | 特征读取的输入列（须在 `ALLOWED_INPUTS` 内） |
| `impl` | 实现引用（V0：`extract_all`，共享函数） |
| `type`、`default_value` | V0 全为 `numeric` / 0.0 |
| `cost_tier` | `L0`..`L3` |
| `online_safe` | 活跃 profile 下须为 true |
| `leakage_risk` | `low`/`medium`/`high` |
| `expected_slices` | 前瞻声明；可为空 |
| `owner` | 归属团队 |

`ALLOWED_INPUTS = {query_text, doc_text, dense_rank, dense_score, sparse_rank,
sparse_score}`。任何其他列（尤其 `label`、`split`、`query_id`、`doc_id`）
在加载时作为泄漏/标识符向量被拒。

加载是 eager 的：`import common` 即触发 `FeatureRegistry.validate()`，遇到
impl 缺失、inputs 越界、`online_safe: false` 或必填字段为空即硬失败
（SystemExit）。这把"FEATURE_NAMES 必须等于 feature_recipes.yaml"契约变成
load-time 检查。

训练出的模型 `reranker_metadata.json` 记录 `feature_set_name`、
`feature_set_version` 与 per-feature 的 `feature_versions` 字典。

## HPO

`scripts/run_hpo.py` 通过 `HPOEngine` adapter（`scripts/hpo/`，Optuna 后端）
搜索 XGBoost 参数。属于**构建/实验**依赖（`optuna` 在
`requirements-build.txt`），非运行时。

```bash
python -m pip install -r skills/heuriboost-rag/requirements-build.txt
python3 skills/heuriboost-rag/scripts/run_hpo.py examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output --n-trials 20 --seed 42 [--timeout-sec 120]
```

输出写入 `examples/fiqa/output/hpo/`（被 git 忽略）：`hpo_report.md`
（val + test nDCG@10 + val−test gap + trial 表）、`best_params.json`
（params + `best_iteration` + scores + feature_set 归属）、`trials.json`
（完整 trial 历史）。

关键契约（见 `.trellis/spec/backend/hpo-contracts.md`）：

- **防泄漏**：HPO 搜索只见 train+valid 快照（签名上 case-blind + test-blind）。
  post-hoc test 评估是单次前向，非优化。
- **确定性**：`nthread=1` + `TPESampler(seed=...)` → 同种子运行产生字节级
  一致的 `trials.json`。
- **可复现**：用 `best_params` + `num_boost_round = best_iteration + 1` 重训
  可精确复现 HPO-best 模型。
- **nDCG 尺度**：HPO 分数用与 shipped baseline（0.853）相同的原始 label
  `ndcg_at_k`，直接可比。
- **过拟合说明**：在 40 query 的 FiQA validation 上，HPO 会过拟合（val 比 test
  高约 0.08，test 可能低于 0.83 基线）。报告通过 val−test gap 诚实暴露这一点。

## 消融

`scripts/run_ablation.py` 跑规格 §15.3 的 A/B/C/D 特征消融：给定候选特征，
测试它在公平 HPO 调参后是否真的有帮助。

```bash
python3 skills/heuriboost-rag/scripts/run_ablation.py examples/fiqa/query_doc_examples.csv \
  --candidate-recipe candidate_recipe.yaml \
  --candidate-impl candidate_impl.py:candidate \
  --output-dir examples/fiqa/output --n-trials 5 --seed 42 \
  --regression-cases examples/fiqa/regression_cases.yaml
```

候选 = recipe YAML（规格 §6.4 字段，`inputs` ⊆ `ALLOWED_INPUTS`）+ impl 函数
`(row) -> float`（`--candidate-impl pyfile:func`）。框架把它叠加到 shipped
`extract_all` 上，不改 registry（作为探针）。

4 个 cell（基线±候选 × 固定/HPO 参数）用相同训练过程；B 与 D 用相同 HPO 预算 +
种子。输出写入 `examples/fiqa/output/ablation/`（被 git 忽略）：
`ablation_report.md`（cell 表 + deltas + 推荐）+ `ablation_result.json`。

Deltas：B-A（参数增益）、C-A（纯特征增益）、**D-B（调参后候选增益——主要）**、
D-C（带候选的调参增益）。

推荐（仅报告——晋级永远手动）：
- `promote` iff D-B(val) > 阈值（默认 0.01）AND D-B(test) > 0 AND D gate 全通过。
- `reject` iff D-B(val) ≤ 0 或 D 退化 gate。
- `quarantine` 否则。

val+test+gate 三重检查避免 cherry-pick HPO 过拟合的 val 噪声。完整契约见
`.trellis/spec/backend/ablation-contracts.md`。

## 候选发现

`scripts/run_discover_candidates.py` 读取 pending regression case 的
`failure_analysis.md` + 现有特征集，调一次 LLM（DeepSeek，JSON 模式）提议 N 个
候选特征，静态校验后写入候选文件，供 `run_ablation.py` 消费。

```bash
export DEEPSEEK_API_KEY=sk-...
python3 skills/heuriboost-rag/scripts/run_discover_candidates.py \
  --out-dir examples/fiqa/output/discovery --n-candidates 5
```

输出写入 `examples/fiqa/output/discovery/`（被 git 忽略）：每个有效候选
`candidates/<name>/{recipe.yaml, impl.py}` + `candidates_report.md`（含"⚠ 运行
消融前请审阅 impl.py"警告）。

LLM 看到 pending cases 的 Feature Contrast + Suggested Actions + 现有特征名 +
primitives API + `ALLOWED_INPUTS` + recipe schema——**不看** label、不看 case 行、
不看 `extract_all` 源码。生成的 `impl_code` 仅静态校验（`ast.parse` + `def
candidate` + import 白名单），生成阶段**不 importlib 加载**（不可信）。用户审阅
`impl.py` 后，用 `run_ablation.py` 测试候选。

无效候选丢弃+告警（1 次 LLM 调用，无重试）。完整契约见
`.trellis/spec/backend/discovery-contracts.md`。

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
加 `--no-ledger` 可跳过 ledger 写入（临时评估用）。`--reckless` 是显式
变体：`train_reranker.py --reckless` 会在省略 `--case-sets` 时默认使用
`examples/fiqa/case_sets`；`eval_reranker.py --reckless --split test` 会硬失败，
除非 ledger anchor 存在、test split 存在、所有引用到的 `source_case_id`
仍按原始 regression rule 通过，并且 test 的 `nDCG@10` 与 `MRR@10` 都超过
anchor。

空的 `case_sets` 输入是允许的；在鲁莽模式下，这意味着没有 source case 需要
重新验收，但 test 的 anchor 对比仍然会执行。

默认流里的锚定基线对比仍然只是“报告，不自动阻断”；鲁莽模式则更严格，若
test 的 `nDCG@10` 或 `MRR@10` 没有超过 anchor，就直接失败。

## 生产 case 修复

用户侧的 reckless 修复流从两张表开始，旧的内部产物由编译器自动生成。

`base_dataset.csv` 是稳定的 train、validation 和指标级 test 验收集。最小列：

```csv
query,text,relevance
```

推荐列：

```csv
domain,query_id,query,doc_id,text,relevance,split,rank,score
```

`production_cases.csv` 是线上 incident / feedback 表。最小列：

```csv
query,shown_doc_text,user_verdict
```

推荐列：

```csv
domain,case_id,query,shown_doc_id,shown_doc_text,user_verdict,rank,score
```

`domain` 可省略，默认 `default`；但一旦提供，就是 synthetic id、候选补全、
promoted repair memory、历史 gates 和 touched-domain 检查的硬边界。`base_dataset`
若带 `split`，编译器会尊重；若不带，则按 query 确定性自动切分。某个 query 只有
一个 doc 时，普通编译只警告；但 strict repair 仍要求 validation/test 的 query
group 至少有两个 doc。

`base_dataset.relevance` 的 label alias：

| alias | 内部 label |
|---|---:|
| `good`, `positive` | `3` |
| `partial` | `2` |
| `weak` | `1` |
| `irrelevant`, `negative` | `0` |
| `bad`, `hard_negative` | `-1` |

`production_cases.user_verdict` 取值：

| verdict | 行为 |
|---|---|
| `good` | 正向 repair sample，也是 full 验收目标 |
| `bad` | hard-negative repair sample，也是压制目标 |
| `unknown` | 仅作为上下文；不进训练，也不参与验收 |

命令：

```bash
python3 skills/heuriboost-rag/scripts/compile_cases.py \
  --base-dataset examples/fiqa/repair/base_dataset_minimal.csv \
  --production-cases examples/fiqa/repair/production_cases_full.csv \
  --output-dir examples/fiqa/output \
  --strict

python3 skills/heuriboost-rag/scripts/repair_reranker.py \
  --base-dataset examples/fiqa/repair/base_dataset_minimal.csv \
  --production-cases examples/fiqa/repair/production_cases_full.csv \
  --output-dir examples/fiqa/output \
  --reckless

python3 skills/heuriboost-rag/scripts/promote_repair.py \
  --output-dir examples/fiqa/output
```

生成的审计产物在 `output/.heuriboost/compiled/`：`query_doc_examples.csv`、
`regression_cases.yaml`、`case_sets/`、`production_cases.json`。这些不是用户
手写前置条件。

strict repair 行为：

- 缺 anchor 时，从只使用 `base_dataset` 的基线自动初始化；
- 已有 anchor 会复用，除非显式 `--reset-anchor`；
- 只向用户暴露一个候选模型，写到 `output/models/`；
- 当前 production cases 会进入 repair 训练；
- base test 仍是指标级回归验收集，不会被 production cases 静默扩充；
- 历史 gates 是自包含 case snapshot，不是 base-test 行。

默认 full 验收要求：至少一个 good doc 进入 top-k，所有 bad doc 离开 top-k，
历史 gates 全通过，全局 base-test `nDCG@10` 与 `MRR@10` 都超过 anchor，且 touched
domain 指标不低于 domain anchor。`--acceptance-level weak` 允许 bad-only 压制
检查，但永远不能 promote。

`promote_repair.py` 会拒绝失败或 weak run。成功 promote 会刷新 repair anchor，
把 full production cases 冻结为历史 gates，追加 promoted repair samples，并写入
`output/.heuriboost/current_model.json`。它不会修改用户输入 CSV，也不会线上部署。

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

# 2b. 鲁莽变体：省略 --case-sets 时默认用 examples/fiqa/case_sets
python skills/heuriboost-rag/scripts/train_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --reckless

# 3. 评估 + 记账（标记该轮使用了 case_sets）
python skills/heuriboost-rag/scripts/eval_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --split validation \
  --regression-cases examples/fiqa/regression_cases.yaml \
  --case-sets-used

# 3b. 鲁莽验收：评估 test，并要求超过 anchor
python skills/heuriboost-rag/scripts/eval_reranker.py \
  examples/fiqa/query_doc_examples.csv \
  --output-dir examples/fiqa/output \
  --split test \
  --reckless

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
