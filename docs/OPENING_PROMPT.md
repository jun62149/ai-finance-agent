# 新会话开场白（直接整段粘贴给新对话框）

---

你是本项目唯一的开发实例（DeepSeek）。**本会话第一步只读**：读完下列文件后向我复述理解，**等我确认再动工**。

## 0. 项目与当前状态

- 项目：**AI 金融智能投研 Agent**（求职向二次开发；基底是已答辩本科毕设《基于 XGBoost 的 A 股选股分析系统》）
- 仓库：`C:\docm\projects\ai-finance-agent` ｜ 当前 HEAD：`4abaeb5` ｜ 工作区**干净** ｜ 审计 checkpoint tag：`review-batch1-before-fix`
- 环境：仓库内 `.venv`（Python 3.12）｜ 测试 **183 passed** ｜ GPU：RTX 4070 12GB（torch 为 CUDA 版）

## 1. 按顺序读这些（都是最新事实来源）

1. `README.md` —— 项目定位、架构图、启动方式、**统一预算口径的评测数字**（唯一对外门面）
2. `docs/TODO.md` —— 未完成项、记录规则（勾选必须等对应 commit 落地）
3. `docs/evaluation_before_fix.md` → `docs/evaluation_after_fix.md` —— 评测协议变更前后（**旧数字不可与新数字直接比较**）
4. 三份面试材料：`docs/bm25_latency_comparison.md`、`docs/bm25_index_consistency.md`、`docs/rrf_parameter_experiment.md`
5. `reports/2026-09-13_修复方案设计.md` —— 5 个 Issue 的修复方案（含**明确不做**的清单）
6. 审计与实施记录：`reports/2026-09-13_Batch1审计结果.md`、`reports/2026-09-13_Step2方案_BM25窗口.md`、`reports/2026-09-13_Step3方案_RRF参数实验.md`、`reports/2026-09-13_Step5方案_时区语义.md`
7. `docs/R1.md`、`docs/R2.md`、`docs/R3.md` —— **外部评审模型的评审协议与验收要求**（我们按它工作）
8. 历史背景（**已冻结，只读不改**）：`docs/CONTEXT_HANDOFF.md`、`docs/archive/*`

> 注：Hindsight 记忆服务在本机未配置 key（401），**项目记忆以 `reports/` + `docs/` 为准**。

## 2. 进度（评审驱动的修复批次）

| Step | 内容 | 状态 |
| --- | --- | --- |
| Step 0 | git checkpoint（tag `review-batch1-before-fix`） | ✅ `77a6acb` |
| Step 1 | **Issue 2**：评测预算统一（candidate/fusion/final-k 拆分 + 显式校验） | ✅ `0e0ee92` |
| Step 2 | **Issue 1**：BM25 取消候选截断（全库打分后过滤，`BM25_WINDOW` 默认 0） | ✅ `3738d85` |
| Step 3 | **Issue 3**：RRF 参数网格实验（12 格 + 2-fold + reranker 校验）→ **保持默认 k=60 等权** | ✅ `0f4c777`、README `1cb550f` |
| Step 4 | **Issue 5**：BM25 manifest + `IndexOutOfSync` 一致性校验 + 空正文丢弃 | ✅ `91fcc82`、`e28dbc7`、`3dad5ef` |
| **Step 5** | **Issue 4**：UTC+8 时间语义统一 | ✅ **已完成并封板**（5-0 ~ 5-8 全部 PASS，最终 commit `8ffec47`；方案 `reports/2026-09-13_Step5方案_时区语义.md`，`4abaeb5`） |
| 之后 | 最终 trace review（评审最后一轮） | 待定 |

Step 5 方案待确认的 4 个问题（**已全部结清**，2026-09-19 复核通过；结论见 `docs/timezone_semantics.md` 与 `reports/2026-09-13_Step5审计结果.md`，历史原文保留如下）：① 固定 UTC+8（不引入 `tzdata`/`ZoneInfo`）？② 审计时间戳（`created_at`/`built_at`/日志）是否一并统一？③ 若前置扫描显示 0 处不一致，是否同意**不回填** `publish_ts`、只改代码+文档？④ 是否接受"市场日历日"语义（跨 UTC 午夜时 `days` 按北京时间算）？

## 3. 铁律（不可违背）

1. **LLM 不负责量化计算**；只做理解、路由、工具调用、证据组织与解释。Agent 只有 5 个工具，**无 SQL/Shell/文件工具**。
2. **无数据不造数据**：结构化错误（`NoData`/`StockNotFound`/`ModelUnavailable`/`LLMUnavailable`/`IndexOutOfSync`…），空结果显式 `has_evidence=false`，**禁止 sample_data 回退**。
3. **证据可追溯**：检索结果必须带 `chunk_id/document_id/stock_code/doc_type/title/source/publish_time/url/content`，缺一即抛 `EvidenceMetadataMissing`——**LLM 读不到的引用不算证据**。
4. **`strong_prob` 是弱信号辅助字段**（分类 recall≈0.0005），任何位置都不得表述为可靠上涨概率。
5. 密钥只在 `.env`；DB / pkl / 向量库 / 模型权重 / 日志不进 Git。
6. 不动原毕设目录（`C:\docm\work\stock_xgboost_system`、`C:\docm\stock_analysis\stock_xgboost_system`）。
7. 小步提交、可回滚；每个 commit 后跑测试。

## 4. 文档纪律（评审明确要求）

- **区分"指标提升"与"正确性修复"**：例如 BM25 那次是"修复 metadata filter 下候选截断导致的漏召回"，**不是**"性能提升 N 倍"。
- **不隐藏旧结果**：旧口径数字全部留档在 `docs/evaluation_before_fix.md`。
- 参数结论表述为"**当前评测集下的经验参数**"，不写"理论最优"。
- 报告一律**新建文件**放 `reports/`（不改旧文档）；`docs/` 下的评测/一致性材料由评审要求新增，可继续维护。

## 5. 工作方式（评审驱动，必须遵守）

1. **先方案、后实现**：每个 Step 先出方案 → 我转给外部评审模型 → **评审通过后才动代码**。
2. **决策规则预先写死**（如"差距 <0.01 就保持简单默认"），避免事后挑参数；事后观察只能作为"探索性记录，不采纳"。
3. 审计发现用 **8 字段格式**：文件路径 / 函数类 / 当前实现 / 为什么有风险 / 能否复现 / 严重程度 P0–P3 / 推荐修法 / 修复影响面。
4. **若无问题必须写 `PASS`；无法确定写 `UNVERIFIED`，不要猜。**
5. **实验代码 ≠ 生产代码**：实验脚本放 `scripts/`，commit 前缀 `exp(...)`，**不改 `app/`**（除非评审通过的修复）。
6. 完成一个 Step 后回报：**修改文件 / commit hash / 测试结果 / 关键实测数据 / 是否需重跑评测 / 影响面**。
7. 涉及数据/索引的改动，必须说明**是否需要重建索引、是否需重跑评测、是否需要 Docker 复测**。

## 6. 环境事实（会影响你的判断）

- 语料：`documents 12,366` / `chunks 109,554`（32 只股票：公告 cninfo + 新闻 eastmoney）
- 向量：Qdrant collection `financial_chunks`（109,554 点，1024 维）；BGE-M3 与 bge-reranker-v2-m3 **已缓存本地**（离线优先加载）
- BM25：`data/bm25/manifest.json` → `chunk_id_hash=333554725291f85e`、`bge-m3-v1`、109,554 docs（改动语料后必须重建，否则加载即抛 `IndexOutOfSync`）
- 量化：因子 **1,577,076 行 / 1,044 只**；行情**已推进至 2026-09-18**（增量更新用 `scripts/update_daily.py`：baostock 优先、akshare 兜底）
- **Docker Desktop 当前已停止** → 任何需要 Qdrant 的复核，先启动：
  `docker compose -f docker/docker-compose.yml up -d qdrant`（API：`ai-finance-api`，宿主 8000）
- 评测口径：`candidate-k = fusion-k = final-k = 20`、`BM25_WINDOW=0`；**不要引用旧表数字**（不等预算）
- `.env` 已配置 DeepSeek（`LLM_BASE_URL=https://api.deepseek.com`、`LLM_MODEL=deepseek-flash`）
  与 `QDRANT_URL=http://localhost:6333`（**必须显式设置**，否则脚本会静默走 `data/qdrant` 本地旧副本）
- 测试总量：`218 passed`（新增/改动测试后请同步 README「测试」一行）

## 7. 协作约定（我明确要求）

1. 全程只用你这一个实例，不换模型、不引入第二个 Agent。
2. **每次任务开始前、以及完成后，都要声明"该任务/下一个任务适合 `deepseek-v4-flash` 还是 `deepseek-v4-pro`"并说明理由。**
3. 不输出阶段报告；进度靠 git log / README / `reports/`。
4. 向量库、语料范围等技术细节由你决定，不必逐项问我。

## 8. 你现在要做的

**只读**上述文件，然后向我复述：

1. 项目现状（含当前 Step 与未完成项）
2. 铁律与文档纪律
3. Step 5 方案的内容，以及它待确认的 4 个问题
4. 你认为**下一个任务**适合 `deepseek-v4-flash` 还是 `deepseek-v4-pro`，理由

**复述完等我确认，再动工。** 若我直接说"继续 Step 5"，请先确认那 4 个问题（或按方案默认执行并明确说明你采用了哪些默认）。
