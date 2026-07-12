# HeuriBoost Reckless Autopilot 设计规格

生成日期：2026-07-10
状态：设计已确认，等待实施计划
范围：HeuriBoost RAG 排序模型的生产 case 修复、自动验收、Pre Promote 报告、人工审批与安全晋级

## 1. 一句话目标

把现有 Reckless 模式中分散的编译、训练、评测、报告和晋级操作收敛为一个可恢复、可审计的状态机：正常路径自动运行到 `READY_FOR_PROMOTION`，维护人员只需根据完整报告执行一次明确的批准动作，系统随后自动 Promote。

## 2. 背景与问题

现有生产修复流程已经具备严格能力：

- `compile_cases.py` 把 `base_dataset` 与 `production_cases` 编译成内部训练与回归产物。
- `repair_reranker.py --reckless` 训练候选模型并检查当前生产 case、历史 gate、全局 test 指标和 touched domain 指标。
- `promote_repair.py` 显式晋级 full acceptance 且满足全部门禁的 run。
- `weak` acceptance 永远不允许 Promote。

问题不在验收规则不足，而在流程需要维护人员多次组织命令、检查中间产物、判断能否继续并手动串联下一步。人工成本应从“每个阶段参与”缩减为“只处理异常和最终高风险决策”。

## 3. 核心原则

1. 不自动创造或修改 `good` / `bad` 标签。输入标签必须来自用户已确认的数据。
2. Reckless 允许快速吸收当前生产 case，但不放松历史 gate、全局指标和领域非退化要求。
3. 正常路径没有中间人工审批；只有最终 Promote 需要一次明确的人类动作。
4. 浏览器或 CLI 展示的旧结论不能作为 Promote 的唯一依据，服务端必须在 Promote 时重新校验。
5. 训练、评测、报告与晋级均可审计、可重放、可恢复。
6. 所有 release 不可变，当前版本通过一个原子指针选择，失败不得形成半晋级状态。
7. 通用机制沉淀在 `~/coding/heuriboost`；消费项目只实现业务数据、特征和在线激活适配。

## 4. 目标与非目标

### 4.1 目标

- 提供稳定的 Python API，而不是继续堆叠互相调用的 CLI 主函数。
- 用策略驱动状态机编排输入校验、编译、训练、评测、报告和 Promote。
- 生成中文默认、可切换英文的完整 Pre Promote HTML。
- 提供幂等、安全、可审计的页面或 API Promote。
- 保持现有低层 CLI 与生产 repair 输入契约兼容。
- 允许消费项目注入特征、训练和模型激活适配器。

### 4.2 非目标

- 不在第一版实现在线样本标注系统。
- 不自动重置 Anchor，不自动删除或退役历史 gate。
- 不允许 `weak` acceptance 通过任何入口 Promote。
- 不把 `prod_recog` 等具体业务特征写入 HeuriBoost 通用库。
- 不在第一版实现多人 RBAC、分布式训练或云对象存储。

## 5. 包结构

在 `plugins/heuriboost/skills/heuriboost-rag/` 下增加正式库层：

```text
plugins/heuriboost/skills/heuriboost-rag/
├── pyproject.toml
├── src/heuriboost_rag/
│   ├── reckless/
│   │   ├── contracts.py
│   │   ├── policy.py
│   │   ├── orchestrator.py
│   │   ├── state.py
│   │   ├── report.py
│   │   ├── promotion.py
│   │   ├── release_store.py
│   │   └── errors.py
│   ├── backends/
│   │   ├── base.py
│   │   └── xgboost_rag.py
│   └── adapters/
│       └── workspace.py
└── scripts/
    ├── reckless_autopilot.py
    ├── repair_reranker.py
    └── promote_repair.py
```

现有脚本保留，但只负责参数解析和调用库 API。核心逻辑不得继续只存在于 CLI `main()` 中。

## 6. 公共接口

### 6.1 核心调用

```python
run = run_reckless_repair(request, backend, stores)
report = render_pre_promote_report(run, locale="zh-CN")
receipt = promote_repair(run.id, approval, promotion_target, stores)
```

### 6.2 RepairRequest

```python
@dataclass(frozen=True)
class RepairRequest:
    workspace_id: str
    base_dataset_id: str
    production_cases_id: str
    policy_version: str
    backend_name: str
    requested_by: str
    run_options: Mapping[str, object]
```

请求只引用已经登记并冻结的数据集版本，不能直接携带任意文件系统路径。

### 6.3 RepairBackend

```python
class RepairBackend(Protocol):
    name: str

    def validate(self, request: RepairRequest, context: RunContext) -> ValidationResult: ...
    def compile(self, request: RepairRequest, context: RunContext) -> CompiledInputs: ...
    def train(self, inputs: CompiledInputs, context: RunContext) -> CandidateModel: ...
    def evaluate(self, candidate: CandidateModel, context: RunContext) -> EvaluationResult: ...
    def verify_artifacts(self, candidate: CandidateModel, context: RunContext) -> ArtifactVerification: ...
```

HeuriBoost 提供默认 `XGBoostRagBackend`，封装当前 `repair_cases.py`、训练和评测能力。消费项目可以注册自己的 backend，但注册项只能来自服务启动配置，不能由浏览器动态提交 Python 路径。

### 6.4 PromotionTarget

```python
class PromotionTarget(Protocol):
    name: str

    def validate_target(self, expected_current: ModelRef | None) -> TargetValidation: ...
    def prepare_release(self, release: ReleaseSnapshot) -> PreparedActivation: ...
    def activate(self, prepared: PreparedActivation) -> ActivationResult: ...
    def rollback(self, receipt: PromotionReceipt) -> RollbackResult: ...
```

默认实现使用不可变 release 目录与 `current_model.json` 指针。`standard-vllm-server-to-b` 的 `prod_recog` 适配器负责其在线模型目录、Schema 兼容和加载目标。

## 7. 状态机

### 7.1 状态

```text
RECEIVED
VALIDATING
COMPILED
TRAINING
TRAINED
EVALUATING
REPORTING
READY_FOR_PROMOTION
PROMOTING
PROMOTED

BLOCKED_INPUT
BLOCKED_NOT_ELIGIBLE
BLOCKED_EVALUATION
PROMOTION_FAILED
INTERRUPTED
CANCELLED
FAILED_INTERNAL
```

### 7.2 正常迁移

```text
RECEIVED
  -> VALIDATING
  -> COMPILED
  -> TRAINING
  -> TRAINED
  -> EVALUATING
  -> REPORTING
  -> READY_FOR_PROMOTION
  -> PROMOTING
  -> PROMOTED
```

### 7.3 阻断规则

- Schema、标签、候选充分性、切分规模或隔离失败：`BLOCKED_INPUT`。
- `weak` acceptance、缺少 good target 或策略明确禁止晋级：`BLOCKED_NOT_ELIGIBLE`。
- 当前生产 case、历史 gate、全局指标或 touched domain 任一失败：`BLOCKED_EVALUATION`。
- 非预期异常：`FAILED_INTERNAL`，保留错误类型、阶段和恢复建议。
- 进程或机器中断：`INTERRUPTED`，只允许从已完成且指纹匹配的阶段续跑。
- Promote 前失败：`PROMOTION_FAILED`，当前模型保持不变，可重试同一 run。

硬阻断 run 不得通过修改状态字段继续运行；修复输入或策略后必须创建新的 run，并关联 `supersedes_run_id`。

## 8. 策略文件

新增版本化 `reckless_policy.yml`，至少包含：

```yaml
version: 1
acceptance_level: full
input:
  min_global_test_queries: 50
  min_domain_test_queries: 10
  min_docs_per_query: 2
  require_authoritative_labels: true
evaluation:
  require_all_current_cases: true
  require_all_historical_gates: true
  require_global_ndcg_improvement: true
  require_global_mrr_improvement: true
  allow_touched_domain_regression: false
promotion:
  allow_weak: false
  require_explicit_human_approval: true
  allow_anchor_reset: false
  allow_gate_retirement: false
```

策略在 run 创建时冻结并计算哈希。报告生成后策略变化不会改变该 run；如需使用新策略，必须新建 run。

## 9. 输入、指纹与恢复

每个 run 的指纹至少覆盖：

- 基础数据集内容哈希与 Schema 哈希
- 生产 case 内容哈希与 Schema 哈希
- backend 名称与版本
- 特征集合名称、版本和顺序
- Reckless 策略内容哈希
- HeuriBoost 代码提交
- 训练参数与随机种子

每个阶段写入独立 `stage_manifest.json`，包含输入指纹、输出产物、哈希、耗时和状态。恢复只能复用输入指纹完全一致的已完成阶段；否则从受影响的最早阶段重新执行。

## 10. 自动验收

`READY_FOR_PROMOTION` 必须同时满足：

1. `acceptance_level == full`。
2. 每个可执行生产 case 至少有一个 good 文档进入要求的 top-k。
3. 所有 bad 文档满足离开 top-k 的要求。
4. 所有历史 gate 通过。
5. 全局 test 的 `nDCG@10` 和 `MRR@10` 均严格优于 Anchor。
6. 每个 touched domain 均不低于对应 Anchor。
7. 特征 Schema 与候选模型元数据完整且一致。
8. 所有必需产物存在且哈希匹配。

提示项可以进入报告，但不能改变硬门禁结论。阻断理由必须结构化写入 `decision.json`。

## 11. Pre Promote 报告

### 11.1 产物

```text
reports/pre_promote_report.html
reports/pre_promote_report_data.json
reports/pre_promote_report_manifest.json
```

HTML 默认中文，支持页面内切换英文。两种语言共享同一份数据 JSON，只翻译标题、字段标签、结论叙述和帮助文本。

### 11.2 必含内容

- Run、代码、策略、backend、环境和时间信息
- 原始文件、格式、工作表、哈希、字段映射和标签来源
- 归一化、去重、隔离、切分、领域分布和告警
- 完整训练参数、轮数、样本/查询组数量、训练曲线和耗时
- 特征版本、特征重要度、模型和 Schema 哈希
- 当前生产 case、历史 gate、全局指标、领域指标及 Anchor 差异
- 每条硬门禁的阈值、输入值、结果和判定理由
- `READY` 或 `BLOCKED` 关键结论、阻断项和提示项
- 产物路径、哈希、精确重跑命令和回滚目标

HTML 内嵌 `<script type="application/json" id="heuriboost-pre-promote-data">`，内容必须与外部 JSON 哈希一致，供 AI Agent 和自动审计工具读取。

### 11.3 不可变性

报告生成后不可修改。Promote 后另生成：

```text
reports/promotion_receipt.html
reports/promotion_receipt.json
```

浏览器中的实时状态可以从 API 读取，但不能覆盖已生成的 Pre Promote 文件。

## 12. Promote 安全协议

### 12.1 审批请求

```python
@dataclass(frozen=True)
class PromotionApproval:
    run_id: str
    approved_by: str
    approved_at: datetime
    report_hash: str
    decision_hash: str
    expected_current_model: str | None
    idempotency_key: str
```

### 12.2 服务端复核

点击“批准并 Promote”后必须重新检查：

- run 仍处于 `READY_FOR_PROMOTION`
- full acceptance 与全部硬门禁仍成立
- 报告、decision、模型和 Schema 哈希未变化
- 当前模型仍等于报告生成时的 `expected_current_model`
- backend 与 PromotionTarget 版本未变化
- 同一 run 未被其他请求 Promote

### 12.3 Release 事务

1. 获取 workspace 级排他锁。
2. 在临时目录构建完整 release 快照。
3. 写入模型、Schema、gate 快照、ledger 快照、晋级样本、审批和 receipt。
4. 校验快照清单与全部哈希。
5. 原子重命名为 `.heuriboost/releases/<run_id>/`。
6. 最后原子替换 `current_model.json` 指针。
7. 写入审计事件并释放锁。

在第 6 步之前失败时，当前模型不变。第 6 步之后的重复请求返回已有 receipt，不重复追加 gate、样本或 ledger。

Anchor 重置、gate 删除和退役不属于本流程，必须使用独立管理命令和单独审计事件。

## 13. 错误模型

所有可预期错误继承稳定的 `HeuriBoostError`，至少包含：

- `code`
- `message`
- `stage`
- `run_id`
- `retryable`
- `details`
- `operator_action`

用户输入错误不得包装成内部异常；内部堆栈写日志，报告和 API 只暴露安全、可操作的摘要。

## 14. 兼容与迁移

- `compile_cases.py`、`repair_reranker.py`、`promote_repair.py` 保留现有参数，并迁移为薄包装。
- 已存在的 `.heuriboost/ledger.json`、`gates.jsonl` 和 promoted samples 通过一次迁移命令导入首个 release 快照。
- 未迁移时只允许读取旧状态，不允许 Web Promote，避免新旧状态同时写入。
- 下层 `train_reranker.py --reckless` / `eval_reranker.py --reckless` 继续供高级维护者使用，但不进入 Web 正常工作流。

## 15. 测试要求

### 15.1 单元测试

- 状态迁移与非法迁移
- 策略解析、冻结和哈希
- 输入指纹与阶段恢复判定
- full / weak 晋级资格
- 当前 case、历史 gate、全局指标和领域指标判定
- 报告数据完整性与中英文同源
- release 清单、哈希和原子指针更新
- Promote 幂等、并发、旧报告和产物篡改

### 15.2 集成测试

- Fake backend 完成正常、阻断、中断和恢复路径
- 小型真实 XGBoost 数据完成训练、评测、报告和 Promote
- Promote 在指针切换前故障时当前模型不变
- 重复 Promote 返回相同 receipt
- 旧状态迁移后可继续创建新 run

## 16. 消费项目边界

`standard-vllm-server-to-b` 只保留：

- `prod_recog` 样本导出和字段映射
- 产品识别专用特征抽取与 feature metadata
- `RepairBackend` / `PromotionTarget` 适配器
- 在线 XGBoost 模型加载、Schema 校验和降级逻辑
- 指向 HeuriBoost 的 submodule 版本

更新顺序固定为：

1. 在 `/home/ansvver/coding/heuriboost` 实现并验证。
2. 更新 HeuriBoost `main`。
3. 在 `standard-vllm-server-to-b` 重新拉取 `third_party/heuriboost` submodule。
4. 运行当前项目的训练集成测试与 `tests/test_product_search_service.py` 回归。

不得在消费项目复制状态机、门禁、报告或 Promote 事务实现。

## 17. 验收标准

- 一条 API/CLI 调用可自动运行到 `READY_FOR_PROMOTION` 或明确的 `BLOCKED_*`。
- 正常路径没有中间人工审批。
- 生成包含完整数据流、训练过程、评测证据和关键结论的中文 HTML。
- 人工一次批准后自动 Promote，并生成不可变 receipt。
- 任意 Promote 失败均不改变当前模型。
- 所有决策和产物可由 run 指纹与清单复现。
- 现有 Reckless CLI 兼容，历史 gate 和 regression 隔离规则不退化。
