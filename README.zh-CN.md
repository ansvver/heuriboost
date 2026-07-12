# HeuriBoost

会记住错误的 RAG 重排序。

[English README](./README.md) · [参考手册](./docs/REFERENCE.zh-CN.md) · [设计规格](./docs/specs/)

## 问题

你的 RAG 系统在回答一个个人理财问题时，引用了一段"场景错位"的文档。

这不一定是召回完全失败。retriever 可能已经找到了正确证据，但同时也找到了一个
语义很像的 hard negative（金融主题相同，但实体/情境错误），并把这段误导性文档
排得太高。generator 看到的是"看起来很相关、却不能支撑答案"的证据。

去改 retriever 的 embedding 模型代价高，还可能把别的都拖坏。于是这个错误下周
又悄悄回来——换一个略有不同的 query，没人察觉，直到用户踩到。

## 思路

HeuriBoost 是一个轻量 reranker，接在你的 retriever 之后，**从你已经见过的具体
失败中学习**，而且关键是：永不遗忘。

```text
query: "Can I deduct home-office expenses as a sole proprietor?"

retriever 输出：                            HeuriBoost 重排后：
  #1 corporate_office_lease  ✗ hard neg      #1 home_office_deduction  ✓ 直接
  #2 standard_deduction      ~ 弱            #2 simplified_method      ~ 部分
  #3 home_office_deduction   ✓ 直接          #4 corporate_office_lease ✗ 被记住
```

每修好一个错误，它就把这个错误写成一条 **regression gate**。之后每一版 reranker
都必须永远把这段误导性文档挡在受保护的 top-k 之外。跑得越久，被钉死的失败越多。

整个卖点就是：一个带记忆的 reranker，让同一个失败不会出现第二次。

## 工作原理

四个互相衔接的阶段。前三个把数据变成一个经过评估的模型；第四个从模型自己犯的
错误中学习，闭环。

```mermaid
flowchart TD
  DATA["准备数据<br/>检索候选 + 打标签"]
  LEARN["训练模型<br/>对每个 query 的文档排序"]
  EVAL["评估并把关<br/>指标 + 已知失败用例"]
  EVOLVE["从失败中学习<br/>挖掘相似样本，回填训练"]

  DATA --> LEARN --> EVAL --> EVOLVE
  EVOLVE -.下一轮.-> LEARN
```

| 阶段 | 做什么 |
|---|---|
| **准备数据** | 对语料跑检索（你的 retriever 不带标签），给每个 query-document 对打分。 |
| **训练模型** | 把每行转成特征，训练 XGBoost 排序模型，按 query 分组使同一 query 的候选在一起。 |
| **评估并把关** | 与检索器基线对比，并回放已知失败用例。一个 **gate** 用例失败就阻断本轮。 |
| **从失败中学习** | 对仍开放的失败，挖掘相似样本回填训练。被稳定修好的失败由人工晋级为 gate。 |

循环的伪代码：

```text
function run_round(dataset, failure_cases, history):
    train = load(dataset, split="train")

    # 可选：挖掘相似样本来攻击开放的失败
    for case in failure_cases.open:
        train += mine_similar(case, corpus)   # 与用例隔离开

    model = train_ranker(train)

    metrics = evaluate(model)
    results = replay(failure_cases, model)    # 冻结用例失败即终止
    history.record(metrics, results)

    suggest_promote(failures_that_now_pass)   # 晋级永远是手动
    return ok if 每个冻结用例仍通过
```

两条铁律绝对成立，它们正是这份记忆可信的根基：

1. **已知失败用例是考题，永不作为训练行。** 只有与用例隔离开的挖掘样本进入训练。
2. **把开放的失败晋级为冻结 gate 永远是人工决定。**

## Demo 效果

demo 使用 **BEIR/FiQA-2018**（金融问答）的真实切片：一段主题相同、但实体错误的
文档和 query 语义相近，却不能支撑答案——正是 HeuriBoost 针对的失败。

在 validation 划分（40 条 query）上，学习到的 reranker 全面压过原始检索器基线：

| Ranker | nDCG@10 | MRR@10 | Recall@5 | Hard-neg@3 |
|---|---:|---:|---:|---:|
| **HeuriBoost** | **0.853** | **0.874** | **0.797** | **0.63** |
| dense | 0.329 | 0.403 | 0.318 | 2.33 |
| sparse | 0.232 | 0.297 | 0.208 | 0.13 |
| RRF | 0.281 | 0.337 | 0.261 | 0.95 |

它在冷 test holdout 上依然泛化（nDCG@10 ≈ 0.83），与 validation 接近——说明提升
不是单纯记忆。top-3 hard negative 暴露从 dense 的 2.33 降到 0.63。

> 这些数字基于**启发式标签**（qrel 正例 + 基于 dense 排名的 hard negative），用于
> 端到端演示循环；需要 benchmark 级标签请用 `--label-mode llm` 重新生成。见
> [DATA_CARD](./examples/fiqa/DATA_CARD.md)。

## 快速开始

```bash
HEURIBOOST_RAG_SKILL_DIR=plugins/heuriboost/skills/heuriboost-rag

# 安装运行时依赖（macOS 还需：brew install libomp 供 xgboost 用）
python -m pip install -r "$HEURIBOOST_RAG_SKILL_DIR/requirements.txt"

# 安装不可变 Reckless CLI/API 所需的可复用 package
python -m pip install -e "$HEURIBOOST_RAG_SKILL_DIR"

# 校验 -> 训练 -> 评估 提交进仓库的 FiQA demo
python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/validate_dataset.py" examples/fiqa/query_doc_examples.csv
python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/train_reranker.py"  examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output
python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/eval_reranker.py"   examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --regression-cases examples/fiqa/regression_cases.yaml
```

## Codex plugin

本仓库在 `plugins/heuriboost/` 中提供一个 repo-local Codex plugin，内含两个
skill：

| Skill | 用途 |
|---|---|
| `$heuriboost:heuriboost-rag` | 审计 RAG 仓库、初始化模板、校验/训练/评估 query-document CSV。 |
| `$heuriboost:reckless-input-builder` | 把杂乱日志、表格、工单、标签、CSV/JSON/JSONL 导出和生产反馈整理成 reckless repair 需要的 `base_dataset` + `production_cases`。 |

从当前 checkout 安装到 Codex：

```bash
codex plugin marketplace add .
codex plugin add heuriboost@heuriboost-local
```

安装或重装后开一个新的 Codex 线程，让 namespaced skills 注入到 prompt。

典型用法：

```text
Use $heuriboost:heuriboost-rag to audit this RAG repo for HeuriBoost readiness.
Use $heuriboost:heuriboost-rag to run an experiment from examples/fiqa/query_doc_examples.csv.
Use $heuriboost:reckless-input-builder to turn these production feedback logs into reckless repair input files.
```

底层鲁莽闭环变体：

```bash
HEURIBOOST_RAG_SKILL_DIR=plugins/heuriboost/skills/heuriboost-rag

python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/train_reranker.py" examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --reckless
python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/eval_reranker.py" examples/fiqa/query_doc_examples.csv --output-dir examples/fiqa/output --split test --reckless
```

### 鲁莽模式：生产 case 快速修复通道

`--reckless` 可以看作一种在线学习式的生产 case 修复通道。用户侧只需要准备两张表：

- `base_dataset.csv`：稳定的 train / validation / test 数据，最小列为
  `query,text,relevance`。
- `production_cases.csv`：线上失败或反馈，最小列为
  `query,shown_doc_text,user_verdict`。

如果你手上的材料是杂乱的日志、表格、工单、标签，或 CSV/JSON/JSONL 导出，可以用
`$heuriboost:reckless-input-builder`。它告诉 agent 如何把检索候选、
accepted/rejected 文档、rank、score、用户反馈映射成这两张表，如何判断 full /
weak 验收，以及如何用 `compile_cases.py` 校验结果。

推荐使用不可变 Reckless autopilot：它按内容哈希登记两份输入，在一个 workspace 中
冻结 policy 与 backend 配置，自动执行校验、训练和评测，最终停在
`READY_FOR_PROMOTION` 或结构化的 `BLOCKED_*` 状态。

```bash
HEURIBOOST_RAG_SKILL_DIR=plugins/heuriboost/skills/heuriboost-rag

python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/reckless_autopilot.py" run \
  --base-dataset examples/fiqa/repair/base_dataset_minimal.csv \
  --production-cases examples/fiqa/repair/production_cases_full.csv \
  --output-dir examples/fiqa/output \
  --historical-gates /approved/history/gates.jsonl \
  --anchor-ledger /approved/history/anchor.json \
  --policy "$HEURIBOOST_RAG_SKILL_DIR/templates/reckless_policy.yml"

python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/reckless_autopilot.py" report \
  --run-id RUN_ID --output-dir examples/fiqa/output --locale zh-CN

python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/reckless_autopilot.py" promote \
  --run-id RUN_ID --output-dir examples/fiqa/output --approved-by maintainer
```

首次创建本地 workspace 时，必须提供已经批准且非空的历史 gate 文件与 anchor ledger。
系统不会创建空 gate，也不会自动重置 anchor。不可变配置写在
`output/.reckless/workspace.json`；之后的 `run` 必须使用相同的 policy、训练、切分、
阈值和 pinned input 配置，否则应使用新的 output 目录。只有冻结输入和 backend identity
仍一致时，`resume --run-id RUN_ID` 才能恢复中断 run。

但鲁莽不等于放松验收。默认 full 验收要求：至少一个 good 生产文档进入 top-k，
所有 bad 文档离开 top-k，历史 gates 全通过，全局 test 的 `nDCG@10` 和 `MRR@10`
都超过 anchor，且所有 touched domain 不退化。只有 bad 的 case 可以显式使用
`--acceptance-level weak`，但 weak run 永远不能 promote。

```mermaid
flowchart LR
    A["base_dataset + production_cases"] --> B["编译生成内部产物"]
    B --> C["repair --reckless 训练一个候选模型"]
    C --> D{"case、gate、test、domain 都通过?"}
    D -- 是 --> E["显式 promote"]
    D -- 否 --> F["硬失败并继续迭代"]
```

Pre Promote 报告不可修改，位于
`output/.reckless/runs/<run-id>/reports/`。Promote 会重新校验报告、decision、候选
产物和当前指针；先发布 `output/.reckless/releases/<run-id>/` 下的不可变 release，最后
原子替换 `output/.reckless/current_model.json`。默认 CLI idempotency key 对同一 run 稳定，
因此中断后的 Promote 可以安全重试。

已有的 `repair_reranker.py --reckless` 和 `promote_repair.py` 保留为兼容包装器：它们保留
旧参数和 `10/3` 的最小 test-query 默认值，但只调用 package API，不再直接写旧 ledger、
gate 或 current pointer。`--reset-anchor` 与 `--keep-baseline-artifacts` 会明确拒绝；
Anchor reset 与 gate retirement 是独立、可审计的管理操作。weak run 永远非零退出，也不能
Promote。

若已有可变的 `output/.heuriboost/` 状态，任何新的 Promote 前先执行一次迁移。迁移会把旧
ledger、gates、promoted samples、current pointer 及其引用模型复制并哈希到不可变 bootstrap
release，不会改写旧文件：

```bash
python3 "$HEURIBOOST_RAG_SKILL_DIR/scripts/migrate_reckless_state.py" \
  --output-dir examples/fiqa/output
```

底层 `train_reranker.py --reckless` / `eval_reranker.py --reckless` 仍保留，
给已经直接维护 `regression_cases.yaml` 与挖掘 `case_sets` 的维护者使用。

要用自己的数据，参考 [CSV 契约](./docs/REFERENCE.zh-CN.md#csv-契约)；完整的失败攻击循环、ledger 和
skill 模式见[参考手册](./docs/REFERENCE.zh-CN.md)。

## 实现 Checklist

已完成：

- [x] 标准 query-document CSV 契约 + 校验器
- [x] 真实 XGBoost 排序模型，按 `query_id` 分组
- [x] 检索 + 文本信号特征集（overlap、hard-negative、长度信号）
- [x] FeatureRecipe registry / recipe DSL（声明式元数据 + load-time 泄漏/online-safe 校验）
- [x] 指标：nDCG、MRR、Recall、hard-negative exposure，对比基线
- [x] 报告：ranking diff、feature importance、确定性失败分析
- [x] HPO adapter（Optuna 后端，确定性，case/test-blind 搜索 + post-hoc test 评估）
- [x] A/B/C/D 消融框架（候选探针 + val/test/gate 三重晋级判定）
- [x] LLM 候选发现（单次 JSON 模式，静态校验，输出供消融消费）
- [x] Regression case 作为 gate，含三态状态机（gate / pending / retired）
- [x] per-case 检查（`require_rank`、`min_ndcg10`）+ 整体质量检查
- [x] 跨轮 ledger，含手动锚定的基线
- [x] `case_sets` 挖掘循环：挖掘相似失败、回填训练、与用例隔离
- [x] `--reckless` 闭环：把 case_sets 直接放进训练，并要求 test nDCG@10 + MRR@10 超过锚点
- [x] 两张表生产修复流：`base_dataset` + `production_cases` 编译、修复、显式 promote
- [x] AI 友好的 reckless 输入构建 skill：把用户原始材料整理成合规 repair 输入
- [x] 端到端 FiQA-2018 demo（提交的 CSV、离线构建器、两种标签模式）
- [x] Codex-compatible agent skill（`audit` / `bootstrap` / `experiment`）

未完成：

- [ ] 提交 demo 的 LLM 模式（benchmark 级）标签
- [ ] 特征晋级记忆（`FeatureMemory`；发现 + 消融已完成，决定的机构记忆待做）
- [ ] 其他 task profile（分类 / 回归 / …）
- [ ] 线上 serving、shadow/backtest、A/B 上线
- [ ] 稳定 Python package / public API（`pyproject.toml`）

"未完成"项背后的设计见 [`docs/specs/`](./docs/specs/)。

## 概念

HeuriBoost 是一个**自适应 XGBoost 框架**，从带标签样本和历史失败中学习。当前
交付的是 RAG query-document reranker 特化；同一架构可推广到分类、回归等监督
表格任务。

| 概念 | 含义 |
|---|---|
| **TaskProfile** | 绑定任务类型与其 objective、指标、gate、slice、serving 行为。Q-D reranker 是其中一个。 |
| **LearningExample** | 一条监督样本。ranking 下同组行共享 `group_id`（`query_id`）。 |
| **PredictionContextSnapshot** | 模型评估所依据的不可变候选集。比较模型须在同一 snapshot 上。 |
| **RegressionCase** | 以 gate 形式表达的历史失败。是 gate，不是训练数据。 |
| **FeatureRecipe** | 声明式、带版本的 feature（输入、成本、online safety、leakage risk）。住在 registry 里，不散落代码。 |
| **PromotionGate** | 候选模型替换当前模型前必须通过的门槛（全局指标、per-case、slice、latency）。 |
| **FeatureMemory** | 记录哪些 feature 被 promote/reject/quarantine 及原因。 |

完整定义见
[`docs/specs/ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md`](./docs/specs/ADAPTIVE_XGBOOST_HEURISTIC_SPEC.md)。

## 目录结构

```text
.
├── README.md / README.zh-CN.md      项目故事、概念、demo
├── .agents/plugins/marketplace.json repo-local Codex plugin marketplace
├── docs/
│   ├── REFERENCE.zh-CN.md            契约 + 命令（操作参考）
│   └── specs/                        长文设计规格
├── examples/fiqa/                    提交的 FiQA demo + cases + ledger
└── plugins/heuriboost/               打包两个 skills 和运行时的 Codex plugin
```

暂无 `pyproject.toml`——请直接运行 skill 目录里的脚本。
