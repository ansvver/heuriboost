# HeuriBoost Web Console 设计规格

生成日期：2026-07-10
状态：设计已确认，等待实施计划
范围：本地优先、可升级为团队服务的 HeuriBoost 浏览器管理控制台
依赖规格：[HeuriBoost Reckless Autopilot 设计规格](./2026-07-10_RECKLESS_AUTOPILOT_DESIGN.md)

## 1. 一句话目标

提供一个本地一条命令启动的 Web 控制台，把已确认标签的数据导入、校验、编译、训练、评测、Pre Promote 审阅、人工批准、自动 Promote、审计和回滚入口统一放到浏览器中。

## 2. 产品定位

第一版采用“单机优先、团队化可升级”的方式：

- 默认只监听维护人员本机 `127.0.0.1`。
- 单用户、单活动 workspace、单训练并发。
- SQLite 保存元数据，文件系统保存数据与模型产物。
- 使用明确接口隔离身份、数据库、产物存储和任务执行，后续可替换为共享服务组件。

第一版不是数据标注平台。CSV、JSONL、XLSX 中的 `good` / `bad` 结论必须已经由人工确认；页面只负责导入、预览、字段映射和校验。

## 3. 设计原则

1. 首页直接进入可操作的运行工作台，不做营销式首页。
2. Web 只调用 Reckless Core Python API，不拼接 shell 命令驱动业务逻辑。
3. 页面状态来自持久化 run 状态，关闭或刷新浏览器不影响训练。
4. 中间阶段自动执行；人工只负责导入并启动，以及最终一次批准。
5. 所有页面结论必须能追溯到结构化 JSON 与不可变产物。
6. 本地实现不能封死团队化升级路径，但第一版不提前引入分布式复杂度。

## 4. 非目标

- 不在第一版实现多人登录、RBAC、审批流编排或组织结构。
- 不支持浏览器内修改 `good` / `bad` 标签。
- 不支持 `.xls`、`.xlsm` 或带宏执行的工作簿。
- 不支持同时运行多个训练任务。
- 不允许浏览器输入任意本地路径、Python 模块或 adapter 名称。
- 不把训练、门禁或 Promote 规则复制到前端 JavaScript。

## 5. 技术方案

### 5.1 组件

```text
Browser
  -> FastAPI routes + Jinja2 templates + native JavaScript
  -> Application Services
  -> Reckless Core / BackendRegistry / PromotionTarget
  -> SQLite RunStore
  -> Local File ArtifactStore
  -> Local JobExecutor
```

### 5.2 依赖分层

HeuriBoost RAG skill 增加可选依赖：

```toml
[project.optional-dependencies]
web = [
  "fastapi",
  "uvicorn",
  "jinja2",
  "python-multipart",
  "openpyxl",
]
```

核心训练库不依赖 Web 组件。Web 测试依赖放入独立 test extra。

### 5.3 启动方式

```bash
python -m heuriboost_rag.web serve \
  --config /path/to/heuriboost-workspace.yml \
  --data-dir ~/.heuriboost \
  --host 127.0.0.1
```

启动配置决定 workspace、backend 和 PromotionTarget。浏览器只能选择已配置项，不能动态加载代码。

## 6. 包结构

```text
src/heuriboost_rag/
├── web/
│   ├── app.py
│   ├── config.py
│   ├── dependencies.py
│   ├── routes/
│   │   ├── pages.py
│   │   ├── imports.py
│   │   ├── datasets.py
│   │   ├── runs.py
│   │   ├── reports.py
│   │   ├── promotions.py
│   │   └── audit.py
│   ├── services/
│   │   ├── import_service.py
│   │   ├── run_service.py
│   │   ├── report_service.py
│   │   └── promotion_service.py
│   ├── stores/
│   │   ├── sqlite.py
│   │   └── filesystem.py
│   ├── jobs/
│   │   ├── executor.py
│   │   ├── worker.py
│   │   └── supervisor.py
│   ├── templates/
│   └── static/
└── importers/
    ├── base.py
    ├── csv.py
    ├── jsonl.py
    └── xlsx.py
```

## 7. 信息架构

主导航：

1. `运行工作台`
2. `数据集`
3. `训练运行`
4. `模型版本`
5. `回归门禁`
6. `审批审计`
7. `系统设置`

### 7.1 运行工作台

首页默认展示“新建 Reckless 修复”和最近运行：

- 选择已登记的基础数据集版本。
- 导入新的生产 case。
- 选择 Reckless 策略与 backend。
- 展示导入预检和阻断项。
- 点击“校验并启动完整流程”。
- 展示最近 run 的状态、关键指标和报告入口。

### 7.2 数据集

- 原始上传文件与归一化数据版本
- 文件格式、工作表、行列数、哈希、Schema 哈希
- 字段映射、校验结果和导入告警
- 基础数据集与生产 case 的角色区分
- 禁止直接覆盖已有版本

### 7.3 训练运行

- 阶段时间线和当前状态
- SSE 实时日志与阶段进度
- 每阶段输入、输出、耗时、指纹和产物
- 取消、重试、克隆 run
- Pre Promote 报告入口

### 7.4 模型版本

- 当前模型、候选模型和历史 release
- 特征 Schema、模型哈希、指标和晋级时间
- Promotion Receipt
- 明确的回滚目标和受控回滚操作

### 7.5 回归门禁

- 历史 gate 列表、来源 run、最近结果和失败详情
- 默认只读
- gate 删除、退役和 Anchor 重置使用独立受控管理操作，不出现在正常 Reckless run 页面

### 7.6 审批审计

- Run 创建、取消、重试、报告生成、审批、Promote、失败和回滚事件
- 操作人、时间、请求来源、run/model/report 哈希
- JSON 与 HTML Receipt 下载

## 8. 浏览器工作流

```text
选择基础数据集
  -> 上传 CSV / JSONL / XLSX 生产 case
  -> 预览与字段映射
  -> 校验并冻结数据集版本
  -> 创建 run
  -> 后台自动编译、训练、评测和生成报告
  -> READY_FOR_PROMOTION
  -> 查看中文 Pre Promote 报告
  -> 点击“批准并 Promote”
  -> 服务端复核并切换 release
  -> 展示 Promotion Receipt 与回滚点
```

浏览器关闭后任务继续运行。重新打开运行详情时，页面从 SQLite 恢复状态并重新订阅 SSE。

## 9. 数据导入

### 9.1 统一接口

```python
class DatasetImporter(Protocol):
    formats: tuple[str, ...]

    def inspect(self, upload: StoredUpload) -> ImportInspection: ...
    def preview(self, upload: StoredUpload, options: ImportOptions) -> PreviewPage: ...
    def normalize(self, upload: StoredUpload, mapping: FieldMapping) -> NormalizedDataset: ...
```

所有格式最终生成相同的内部表结构、Schema manifest 和内容哈希。下游校验、编译与训练不感知源格式。

### 9.2 CSV

- UTF-8 优先，编码失败时给出明确错误，不静默替换字符。
- 支持逗号分隔的标准 CSV；其他分隔符需显式选择。
- 保留原始文件，归一化后写 Parquet 或等价的类型稳定格式。

### 9.3 JSONL

- 每行必须是 JSON object。
- 错误报告包含准确行号。
- 不接受顶层数组冒充 JSONL。

### 9.4 XLSX

首版只支持 `.xlsx`：

- 上传后列出非隐藏工作表、行列数和首行预览。
- 用户选择工作表和表头行。
- 页面自动建议字段映射，并允许保存为 import profile。
- `openpyxl` 使用 `read_only=True`、`data_only=True` 和禁用外部链接的配置。
- 公式只读取已缓存结果；必需字段没有缓存值时阻断并列出单元格。
- 拒绝 `.xlsm`、宏、损坏 ZIP、超大解压体积和超限单元格数量。
- 限制由策略配置控制：上传字节数、工作表数、每表行数、列数、总单元格数和解压体积。

原始文件永远保留，归一化结果与映射独立版本化。

## 10. 任务执行

### 10.1 进程模型

- FastAPI 进程只处理 HTTP、模板与 SSE。
- Local JobExecutor 使用 SQLite 队列。
- 单个 worker 顺序领取任务。
- 每个 run 在独立子进程中调用 Reckless Core，隔离 XGBoost 内存和崩溃。
- worker 记录 PID、心跳、当前阶段和日志偏移。

### 10.2 任务状态

```text
QUEUED
CLAIMED
RUNNING
SUCCEEDED
BLOCKED
INTERRUPTED
CANCEL_REQUESTED
CANCELLED
FAILED
```

Run 状态使用核心规格中的状态机，job 状态只描述执行器生命周期，两者不得混用。

### 10.3 中断恢复

- Web 重启：QUEUED 任务保持，RUNNING 子进程继续由 supervisor 监控。
- worker 重启或机器重启：心跳超时的任务标记为 `INTERRUPTED`。
- 恢复时读取 stage manifest，只复用指纹一致的完成阶段。
- 临时文件写入 run 的 staging 目录，阶段完成后原子重命名。
- 取消请求在阶段边界生效；必要时终止训练子进程，但保留已完成阶段和日志。

## 11. 存储设计

默认 `--data-dir ~/.heuriboost`：

```text
~/.heuriboost/
├── heuriboost.db
├── uploads/<upload_id>/source.*
├── datasets/<dataset_id>/
│   ├── normalized.parquet
│   ├── schema.json
│   ├── mapping.json
│   └── manifest.json
├── runs/<run_id>/
│   ├── stages/
│   ├── models/
│   ├── reports/
│   ├── logs/
│   └── decision.json
├── releases/<run_id>/
└── current_model.json
```

数据目录可以配置到消费项目的受控路径，但默认不写入 HeuriBoost Git 仓库。

## 12. SQLite 数据模型

### 12.1 主要表

- `workspaces`：名称、adapter、配置哈希和活动状态
- `uploads`：原始文件、格式、大小、哈希和安全扫描结果
- `datasets`：角色、Schema、归一化路径、版本和校验状态
- `import_profiles`：格式、工作表、表头与字段映射
- `runs`：请求、核心状态、指纹、策略和关联数据集
- `run_stages`：阶段状态、输入输出哈希、耗时和错误
- `jobs`：执行器状态、PID、心跳、取消标记和重试计数
- `artifacts`：类型、路径、哈希、大小和生成阶段
- `approvals`：审批人、报告/decision 哈希和幂等键
- `promotions`：release、前后模型、结果和 receipt
- `audit_events`：不可变操作事件

### 12.2 数据规则

- 所有 ID 使用不可猜测的稳定标识。
- 时间统一存 UTC，页面按本地时区显示。
- JSON 扩展字段有显式 schema version。
- 数据库迁移使用版本化 migration，不允许运行时临时改表。
- 审计事件只能追加，不能更新或删除。

## 13. API

### 13.1 导入与数据集

```text
POST /api/imports
GET  /api/imports/{import_id}
GET  /api/imports/{import_id}/sheets
POST /api/imports/{import_id}/preview
POST /api/imports/{import_id}/normalize
POST /api/datasets/{dataset_id}/validate
GET  /api/datasets
```

### 13.2 Run

```text
POST /api/runs
GET  /api/runs/{run_id}
GET  /api/runs/{run_id}/events
POST /api/runs/{run_id}/cancel
POST /api/runs/{run_id}/retry
POST /api/runs/{run_id}/clone
```

### 13.3 报告与晋级

```text
GET  /api/runs/{run_id}/pre-promote-report
GET  /api/runs/{run_id}/report-data
POST /api/runs/{run_id}/promote
GET  /api/runs/{run_id}/promotion-receipt
GET  /api/releases
POST /api/releases/{run_id}/rollback
```

### 13.4 审计

```text
GET /api/audit-events
GET /api/audit-events/{event_id}
```

所有写 API 接收 `Idempotency-Key`。Promote API 额外要求一次性审批 token、report hash、decision hash 和 expected current model。

## 14. SSE 与日志

`GET /api/runs/{run_id}/events` 推送结构化事件：

```json
{
  "event_id": 123,
  "run_id": "RUN-20260710-01",
  "type": "stage_progress",
  "stage": "TRAINING",
  "status": "RUNNING",
  "progress": 0.65,
  "message": "Training round 26/40",
  "occurred_at": "2026-07-10T08:12:30Z"
}
```

事件 ID 单调递增，浏览器使用 `Last-Event-ID` 断线续传。日志显示需要限流和分页，不能把完整日志永久缓存在浏览器内存。

## 15. Pre Promote 页面与自动 Promote

页面采用已确认的“关键决策在顶部、完整过程按时间线展开”结构：

- 顶部：`READY/BLOCKED`、阻断项、提示项、全局指标、当前 case 和历史 gate。
- 正文：样本接入、校验隔离、数据编译、模型训练、全量评测、决策、产物索引。
- 中文默认，支持 English 切换。
- 页面读取不可变 report data，不能在前端重新计算晋级资格。

直接从文件系统离线打开的 `pre_promote_report.html` 是只读审计产物，不显示可用的 Promote 控件。只有本机 Web 服务基于同一份不可变报告快照渲染的交互页面，才提供审批操作区并调用服务端 Promote API。

只有 `READY_FOR_PROMOTION` 显示“批准并 Promote”。点击后：

1. 浏览器提交审批人、一次性 token、report/decision hash 和 expected current model。
2. 服务端执行核心规格中的全部复核与 release 事务。
3. 成功后页面展示 `PROMOTED`、新模型、receipt 和回滚点。
4. 失败时保留原当前模型，展示结构化失败原因和重试入口。

## 16. 本地安全

- 默认且建议只绑定 `127.0.0.1`。
- 启动时生成短期会话 token，通过浏览器启动链接进入。
- 所有写请求要求同源 CSRF token。
- 不配置宽松 CORS。
- 上传文件名只作为展示信息，磁盘路径使用生成 ID。
- 防止路径穿越、符号链接逃逸、ZIP bomb 和超大上传。
- 浏览器不能编辑 workspace adapter、PromotionTarget 或可执行命令。
- 日志和错误页面不得暴露敏感环境变量或任意文件内容。
- Promote 需要 workspace 锁和一次性审批 token。

团队版升级时使用 `IdentityProvider` 替换本地身份，不改变审批领域模型。

## 17. 扩展接口

```python
class IdentityProvider(Protocol): ...
class RunStore(Protocol): ...
class ArtifactStore(Protocol): ...
class JobExecutor(Protocol): ...
```

第一版实现：

- `LocalIdentityProvider`
- `SQLiteRunStore`
- `LocalFileArtifactStore`
- `LocalProcessJobExecutor`

团队版可替换为 SSO、PostgreSQL、对象存储和分布式任务队列，而页面与 Reckless Core 接口保持不变。

## 18. 测试要求

### 18.1 导入器

- CSV 编码、分隔符、空值和重复表头
- JSONL 非 object、错误行号和截断文件
- XLSX 多工作表、隐藏表、表头选择、字段映射和保存 profile
- XLSX 公式缓存缺失、损坏 ZIP、宏扩展名、外部链接和容量限制
- 三种格式归一化后得到相同的内部 Schema 和语义哈希

### 18.2 Web 与 API

- 上传、预览、归一化、校验和数据集版本化
- Run 创建、状态查询、SSE 断线续传、取消和重试
- CSRF、会话 token、路径穿越、非法 adapter 和超大上传
- 所有写 API 的幂等性
- 服务重启、worker 重启和 interrupted run 恢复

### 18.3 Promote

- 页面只在 READY 状态显示 Promote
- report/decision/model hash 变化时拒绝
- 当前模型变化时拒绝
- 重复点击和并发点击只产生一个 release
- Promote 失败不改变当前模型
- Receipt 与审计事件完整

### 18.4 UI

- Playwright 覆盖运行工作台、XLSX 导入、运行详情、报告和 Promote 闭环
- 常用桌面与移动视口无重叠、截断或无法操作的控件
- 状态不能只依赖颜色表达
- 键盘可完成主要表单和审批前浏览
- 中文与英文切换不改变数值、hash 或结论状态

### 18.5 真实集成

- 小型 XGBoost 数据完成浏览器端到端流程
- `standard-vllm-server-to-b` 的 `prod_recog` adapter 完成训练、报告、Promote 和在线加载
- 回归 `tests/test_product_search_service.py` 既有 case

## 19. 文档与运维

- README.zh-CN.md 增加 Web 启动、数据目录和常见故障入口。
- 参考手册记录 API、状态机、XLSX 限制和备份恢复。
- 现有维护 HTML 更新为 Web Console 与 CLI 两种运行路径。
- 数据目录提供显式备份命令，至少覆盖 SQLite、datasets、releases 和 current pointer。

## 20. 验收标准

1. 一条命令启动本地 Web Console，并给出可访问 URL。
2. 浏览器可导入 CSV、JSONL、XLSX；XLSX 支持工作表、表头和字段映射。
3. 用户可复用基础数据集，只上传新生产 case 创建 run。
4. 浏览器关闭或服务刷新不丢失任务和状态。
5. 中间校验、编译、训练、评测和报告无需人工确认。
6. 系统生成完整中文 Pre Promote HTML，并支持英文切换。
7. 人工一次“批准并 Promote”完成安全晋级并生成 Receipt。
8. 任意失败都不改变当前模型，并保留可操作错误和审计记录。
9. 当前项目通过 submodule 更新使用上游实现，不复制 Web 或状态机代码。
