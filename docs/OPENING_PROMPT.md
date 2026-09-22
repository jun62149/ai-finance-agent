# 新会话开场白（直接整段粘贴给新对话框）

---

你是本项目唯一的开发实例（DeepSeek）。**本会话第一步只读**：读完下列文件后向我复述理解，**等我确认再动工**。

## 0. 项目与当前状态

- 项目：**AI 金融智能投研 Agent**（求职向二次开发；基底是已答辩本科毕设《基于 XGBoost 的 A 股选股分析系统》）
- 私有仓库：`C:\docm\projects\ai-finance-agent` ｜ HEAD：以 `git log -1` 为准（本文件最后更新于 `6ca4703`）｜ 工作区**干净** ｜ tag：`review-batch1-before-fix`
- 对外快照：`C:\docm\projects\ai-finance-agent-public`（119 文件、单次初始提交 `2be8365`、**尚未推送**；不含 `reports/` 与内部流程文档）
- 环境：仓库内 `.venv`（Python 3.12）｜ 测试 **218 passed** ｜ GPU：RTX 4070 12GB（torch CUDA 版）
- **本机 shell 是 Windows PowerShell 5.1**：`Set-Content -Encoding utf8` 会写 BOM、`Get-Content -Raw` 按 GBK 解码 → **改文本一律用编辑器/文件工具或 Python，不要用 PS 文本管道**（已因此写坏过 `.env` 与公开 README 的编码）

## 1. 按顺序读这些（都是最新事实来源）

1. `README.md` —— 项目定位、架构图、启动方式、**统一预算口径的评测数字**（唯一对外门面）
2. `docs/TODO.md` —— 未完成项、记录规则（勾选必须等对应 commit 落地）
3. `docs/evaluation_before_fix.md` → `docs/evaluation_after_fix.md` —— 评测协议变更前后（**旧数字不可与新数字直接比较**）
4. 三份面试材料：`docs/bm25_latency_comparison.md`、`docs/bm25_index_consistency.md`、`docs/rrf_parameter_experiment.md`
5. **`docs/timezone_semantics.md`** —— 时间语义契约（Step 5 的长期参考：市场日=UTC+8、A/B 类时间分工、BEFORE/AFTER 数值、UTC 容器验证）
6. 审计与实施记录：`reports/2026-09-13_修复方案设计.md`、`reports/2026-09-13_Batch1审计结果.md`、`reports/2026-09-13_Step5方案_时区语义.md`、**`reports/2026-09-13_Step5审计结果.md`（含 5-8 trace matrix 与 findings）**、`reports/step5_publish_ts_scan_baseline.md`、`reports/step5_5-2_断点记录.md`
7. `docs/R1.md`、`docs/R2.md`、`docs/R3.md` —— **外部评审模型的评审协议与验收要求**（我们按它工作；R1 的 Batch 2/3/4 尚未执行）
8. `reports/resume_project_cards.md` —— 简历/面试素材卡（**含"现在不能写的话"清单**，不要外发）
9. 历史背景（**已冻结，只读不改**）：`docs/CONTEXT_HANDOFF.md`、`docs/phase0_report.md`、`docs/archive/*`

> 注：Hindsight 记忆服务在本机未配置 key（401），**项目记忆以 `reports/` + `docs/` 为准**。

## 2. 进度（评审驱动的修复批次）

| Step | 内容 | 状态 |
| --- | --- | --- |
| Step 0 | git checkpoint（tag `review-batch1-before-fix`） | ✅ `77a6acb` |
| Step 1 | **Issue 2**：评测预算统一（candidate/fusion/final-k 拆分 + 显式校验） | ✅ `0e0ee92` |
| Step 2 | **Issue 1**：BM25 取消候选截断（全库打分后过滤，`BM25_WINDOW` 默认 0） | ✅ `3738d85` |
| Step 3 | **Issue 3**：RRF 参数网格实验（12 格 + 2-fold + reranker 校验）→ **保持默认 k=60 等权** | ✅ `0f4c777`、README `1cb550f` |
| Step 4 | **Issue 5**：BM25 manifest + `IndexOutOfSync` 一致性校验 + 空正文丢弃 | ✅ `91fcc82`、`e28dbc7`、`3dad5ef` |
| **Step 5** | **Issue 4**：UTC+8 时间语义统一（5-0 ~ 5-8 全部 PASS，已封板） | ✅ `1fdeb7c`→`8ffec47`（文档 `e02c941`/`d37fa40`） |
| **下一步** | **Batch 2 → Batch 3 → Batch 4 → 最终 P0–P3 清单**（按 `docs/R1.md` 原顺序；Batch 1 已完成） | ⏳ 待开始 |

Step 5 收尾状态：不执行 backfill（全量扫描 109,554 点 `mismatch=0`）；不修改生产代码/测试/Docker TZ；**N1 / N2 / O1 三个 finding 已记录但按评审决定不实施**（见 `reports/2026-09-13_Step5审计结果.md` §12）。

Batch 划分（`docs/R1.md`）：Batch 2 = `stock_comparison` / Agent 终止条件 / Evidence Validator；Batch 3 = `clean_text` / SQLite 并发 / 评测方法学；Batch 4 = 真实问答 trace / Docker 部署 / README 求职展示风险 / 最终 P0–P3 清单。

## 3. 铁律（不可违背）

1. **LLM 不负责量化计算**；只做理解、路由、工具调用、证据组织与解释。Agent 只有 5 个工具，**无 SQL/Shell/文件工具**。
2. **无数据不造数据**：结构化错误（`NoData`/`StockNotFound`/`ModelUnavailable`/`LLMUnavailable`/`IndexOutOfSync`…），空结果显式 `has_evidence=false`，**禁止 sample_data 回退**。
3. **证据可追溯**：检索结果必须带 `chunk_id/document_id/stock_code/doc_type/title/source/publish_time/url/content`，缺一即抛 `EvidenceMetadataMissing`——**LLM 读不到的引用不算证据**。
4. **`strong_prob` 是弱信号辅助字段**（分类 recall≈0.0005，见 `models/model_metrics.json`），任何位置都不得表述为可靠上涨概率。
5. 密钥只在 `.env`；DB / pkl / 向量库 / 模型权重 / 日志不进 Git。
6. 不动原毕设目录（`C:\docm\work\stock_xgboost_system`、`C:\docm\stock_analysis\stock_xgboost_system`）。
7. 小步提交、可回滚；每个 commit 后跑测试。

## 4. 文档纪律（评审明确要求）

- **区分"指标提升"与"正确性修复"**：例如 BM25 那次是"修复 metadata filter 下候选截断导致的漏召回"，**不是**"性能提升 N 倍"。
- **不隐藏旧结果**：旧口径数字全部留档在 `docs/evaluation_before_fix.md`。
- 参数结论表述为"**当前评测集下的经验参数**"，不写"理论最优"。
- 报告一律**新建文件**放 `reports/`（不改旧文档）；`docs/` 下的评测/一致性材料由评审要求新增，可继续维护。
- **历史与冻结文档不得改写**：`docs/CONTEXT_HANDOFF.md`、`docs/phase0_report.md`、`docs/archive/*`、`reports/2026-09-10_*`、`reports/2026-09-13_Step5方案_时区语义.md`。发现其中有过期/笔误，只能在**新**文档里以"历史记录 → 最终核实"的方式更正。
- **公开快照只含 curated 集合**：`app/ tests/ scripts/ frontend/ docker/ data/eval/ README.md .env.example .gitignore requirements.txt` + 6 份对外技术文档（`docs/` 内：timezone_semantics、bm25_index_consistency、bm25_latency_comparison、rrf_parameter_experiment、evaluation_before/after_fix）。**`reports/`、`docs/OPENING_PROMPT.md`、`docs/R1–R3.md`、`docs/CONTEXT_HANDOFF.md`、`docs/TODO.md`、`docs/phase0_report.md`、`docs/archive/*`、简历素材卡一律不外发。**

## 5. 工作方式（评审驱动，必须遵守）

1. **先方案、后实现**：每个 Step 先出方案 → 我转给外部评审模型 → **评审通过后才动代码**。
2. **审计批次只审不改**：Batch 2/3/4 各自先**只读审计**并输出"计划/基线"→ 等我复核 → 才实施；**不跨 Batch**、不顺手修无关项、不为清 finding 而改动已验收内容。
3. **决策规则预先写死**（如"差距 <0.01 就保持简单默认"），避免事后挑参数；事后观察只能作为"探索性记录，不采纳"。
4. 审计发现用 **8 字段格式**：文件路径 / 函数类 / 当前实现 / 为什么有风险 / 能否复现 / 严重程度 P0–P3 / 推荐修法 / 修复影响面。
5. **若无问题必须写 `PASS`；无法确定写 `UNVERIFIED`，不要猜。**
6. **实验代码 ≠ 生产代码**：实验脚本放 `scripts/`，commit 前缀 `exp(...)`，**不改 `app/`**（除非评审通过的修复）。
7. 完成一个 Step 后回报：**修改文件 / commit hash / 测试结果 / 关键实测数据 / 是否需重跑评测 / 影响面**。
8. 涉及数据/索引的改动，必须说明**是否需要重建索引、是否需重跑评测、是否需要 Docker 复测**。

## 6. 环境事实（会影响你的判断）

- 语料：`documents 12,366` / `chunks 109,554`（32 只股票：公告 cninfo 4,623 篇 + 新闻 eastmoney 7,743 篇）
- 向量：Qdrant collection `financial_chunks`（109,554 点，1024 维）；BGE-M3 与 bge-reranker-v2-m3 **已缓存本地**（离线优先加载）
- BM25：`data/bm25/manifest.json` → `chunk_id_hash=333554725291f85e`、`bge-m3-v1`、109,554 docs（改动语料后必须重建，否则加载即抛 `IndexOutOfSync`）
- 量化：因子 **1,577,076 行 / 1,044 只**；行情**已推进至 2026-09-18**（`stock_daily` 161 万行；增量更新用 `scripts/update_daily.py`：baostock 优先、akshare 兜底）
- Docker：引擎在运行（29.8.0），但**本项目容器未启动**（现在只有别的项目的 `milvus-standalone`）。需要 Qdrant 时先：
  `docker compose -f docker/docker-compose.yml up -d qdrant`
- 评测口径：`candidate-k = fusion-k = final-k = 20`、`BM25_WINDOW=0`；**不要引用旧表数字**（不等预算）
- `.env`：已配 `LLM_API_KEY`（35 位）与 `LLM_MODEL=deepseek-flash`；**`QDRANT_URL` 目前刻意停用**——见下条
- **待修（未获批前不要动）**：`app/rag/vector_store.py::_build_client` 里 `target_url = url if url is not None else Config.QDRANT_URL` 会让该变量覆盖显式传入的 `path=`，导致向量相关单测、`scripts/migrate_qdrant.py`（本地源）、`scripts/bench_retrieval.py`（本地列）指向错误目标；修复前**不要把 `QDRANT_URL` 写进 `.env`**，只在需要时 `$env:QDRANT_URL="http://localhost:6333"`
- 测试总量：`218 passed`（改动测试后请同步 README「测试」一行）
- 证据文件 `logs/*.txt` 是 PS 5.1 写出的非 UTF-8（Tee-Object），阅读时注意编码

## 7. 协作约定（我明确要求）

1. 全程只用你这一个实例，不换模型、不引入第二个 Agent。
2. **每次任务开始前、以及完成后，都要声明"该任务/下一个任务适合 `deepseek-v4-flash` 还是 `deepseek-v4-pro`"并说明理由。**
3. 不输出阶段报告；进度靠 git log / README / `reports/`。
4. 向量库、语料范围等技术细节由你决定，不必逐项问我。

## 8. 你现在要做的

**只读**上述文件，然后向我复述：

1. 项目现状（当前 HEAD、测试总量、Step 5 收尾状态、以及**下一个 Batch 是什么**）
2. 铁律与文档纪律（尤其"审计批次只审不改"与"历史文档不得改写"）
3. Batch 2 的三块审计范围（`stock_comparison` / Agent 终止条件 / Evidence Validator）各自要回答什么问题
4. 你认为**下一个任务**适合 `deepseek-v4-flash` 还是 `deepseek-v4-pro`，理由

**复述完等我确认，再动工。** 若我直接说"开始 Batch 2"，请先只做**只读审计**并输出计划/基线，**不要直接改代码**。

## 9. 待办工程项（与 Batch 审计无关，需我批准后才动）

1. `_build_client` 优先级修复（1 行：显式 `path` 优先于 `Config.QDRANT_URL`）→ 修完可把 `QDRANT_URL` 写回 `.env`。
2. 公开快照推送（建 GitHub/Gitee 仓库 → `git remote add` → push）；可选补 `LICENSE` 与 1~2 张截图。
3. README/TODO 收尾（README benchmark 表与结论段、RRF k=10 复核、adversarial eval case）。
4. Step 5 的 N1 / N2 / O1 三项 finding（评审已决定暂不处理）。
