# 简历项目卡：AI 金融智能投研 Agent（可维护）

> 用途：**简历/面试专用素材库**。这里只放"每一句都能被仓库证据支持"的表述，并明确标出哪些话**现在还不能写**、以及解锁条件。
> 与评测/审计文档的关系：本文件是**下游消费者**，数字的唯一来源是 `README.md` 与 `docs/evaluation_*.md`；审计结论的来源是 `reports/`。
> 维护规则见 §7。**本文件不是验收证据，不要在这里发明新数字。**

最后更新：2026-09-19 ｜ 对应 HEAD：`ee14484`（Step 5 封板 `8ffec47`）

---

## 1. 项目定位（写在简历最前面的一句）

> **AI 金融智能投研 Agent（个人项目，求职向二次开发）** —— 在已答辩的本科毕设《基于 XGBoost 的 A 股选股分析系统》基础上，扩展 RAG 检索与 LangGraph Agent 层。

作用：避免"两个 A 股项目"被看成同一件事写两遍（毕设 = 量化选股底座，本项目 = 其上的 AI 检索/问答层）。

**投放顺序建议**

| 目标岗位 | 顺序 |
| --- | --- |
| AI 应用 / LLM / RAG / 后端 | 本项目第一，毕设第二 |
| 量化 / 数据分析 | 毕设第一，本项目第二 |

---

## 2. 已核实可写清单

每条格式：**简历表述** → 证据 → 数字来源。`✅` = 今天即可写。**改写前先看 §4。**

| # | 简历表述（可直接用） | 证据（仓库内可核对） | 数字来源 |
| --- | --- | --- | --- |
| 1 | 知识库：**12,366 篇文档 / 109,554 chunks**，覆盖 32 只 A 股 | Qdrant `points_count=109554` 与 SQLite `chunks=109554 / documents=12366` 一致（5-2 全量扫描） | `reports/step5_publish_ts_scan_baseline.md` §4；`README.md` L12 |
| 2 | 语料构成：cninfo 官方公告（含 PDF 正文）+ 财经新闻 | `README.md` L12 | `README.md` L12 |
| 3 | 混合检索：**BGE-M3 向量 + bm25s(jieba) 关键词 → RRF(k=60) 融合 → bge-reranker-v2-m3 重排**，Qdrant 存储 + 元数据过滤 | `app/rag/{vector_store,bm25,hybrid,reranker,retrieval}.py` | `README.md` L13 |
| 4 | 热态端到端检索 **32~376ms** | — | `README.md` L13 |
| 5 | 评测：自建 **36 题 / 373 条标注**评测集，覆盖 5 类查询（代码/简称/日期约束/公告类型/语义事件） | `data/eval/`、`scripts/build_eval_set.py`、`scripts/evaluate_retrieval.py` | `README.md` L136 |
| 6 | 统一预算协议 `candidate-k = fusion-k = final-k = 20`、`BM25_WINDOW=0`，四配置预算完全相同 | `scripts/evaluate_retrieval.py` | `README.md` L140-141 |
| 7 | 带元数据过滤（贴近 Agent 实际用法）**Recall@5**：Rerank **0.4563** / BM25 0.4351 / Hybrid 0.4248 / Dense 0.3922；**MRR 0.7119**（Rerank）；中位延迟 14 / 29 / 41 / 452 ms | `docs/evaluation_after_fix.md` | `README.md` L143-150 |
| 8 | 不带过滤（更难）Hybrid+Rerank **Recall@5 0.4492** | 同上 | `README.md` L152-159 |
| 9 | RRF 参数做过 **12 格网格**（k∈{10,30,60,100} × 权重 {1:1,1:2,1:3}）+ **2-fold 稳定性** + reranker 短名单校验；结论**保持默认 k=60 + 等权**，未为了单一指标换参数 | `docs/rrf_parameter_experiment.md`、`data/eval/rrf_grid_*.json` | `README.md` L166-169 |
| 10 | 评测协议自查：发现四配置 chunk 预算不等 → 统一后重跑，并**保留旧口径留档**（新旧数字明示不可比较） | `docs/evaluation_before_fix.md` / `docs/evaluation_after_fix.md` | `README.md` L161-164 |
| 11 | Agent：LangGraph 工具调用编排，**仅暴露 5 个受控工具**（1 个检索 + 4 个量化；无 SQL / Shell / 文件工具），结构化错误（`NoData` / `StockNotFound` / `ModelUnavailable` / `IndexOutOfSync` / `LLMUnavailable` / `EvidenceMetadataMissing`），空结果显式 `has_evidence=false`，**禁止造数据回退** | 工具集：`app/tools/registry.py:41-79`（`TOOL_SPECS` 共 5 条）、`:8-9`（"no SQL, shell or filesystem tool here by design"）；错误类：`app/errors.py:20/25/35/45`、`app/agents/llm.py:24`、`app/tools/errors.py:14-36`；`has_evidence`：`app/tools/rag_tools.py:197` | `docs/OPENING_PROMPT.md` §3 铁律（L40-48） |
| 12 | 证据可追溯：每条检索证据携带 `chunk_id / document_id / stock_code / doc_type / title / source / publish_time / url / content`，缺字段即抛错 | `app/tools/rag_tools.py:31`（`REQUIRED_EVIDENCE_FIELDS`）、`:183`（缺字段抛 `EvidenceMetadataMissing`） | 同上 |
| 13 | 工程：FastAPI + **Docker Compose 一键起 Qdrant / API / Web**（本地编排，非线上部署）+ **218 个自动化测试** | `docker/docker-compose.yml`、`tests/` | 本会话实跑 `218 passed` |
| 14 | **缺陷修复①（召回漏检）**：BM25 候选窗口在后置过滤下漏召回，实测最坏 **2 / 356**，16/16 参数组合均有损失 → 改为"全库打分后过滤"（`BM25_WINDOW=0`），并把 WINDOW 保留为显式可选截断 + 触发时告警 | `reports/2026-09-13_Batch1审计结果.md` L15/L100、`reports/2026-09-13_修复方案设计.md` L44 | 审计报告原文 |
| 15 | **缺陷修复②（跨环境时区）**：日期窗口语义不统一导致 UTC 容器下边界整体偏移 8 小时、边界日 00:00–07:59:59 数据被漏 → 统一为显式 UTC+8；边界日窗口命中 **44,601 → 51,723 chunks（+7,122）**，边界日午夜文档 **0 → 1** | `docs/timezone_semantics.md`、`reports/2026-09-13_Step5审计结果.md` | 5-6 UTC 容器实测 |
| 16 | **回归守护**：UTC 容器内旧代码 12 条断言失败、当前代码 35 条全通过；测试总量 **183 → 218**；并为"日期→epoch 转换集中在一处"加了 AST 源码守卫 | `tests/unit/test_timeutil.py`、`logs/step5_utc_pytest_before_code.txt` | 5-5 / 5-6 |
| 17 | 索引一致性：BM25 manifest（chunk 指纹 + embedding 版本）+ 加载时 `IndexOutOfSync` 校验，避免"向量库换了、BM25 没重建"的静默失配 | `docs/bm25_index_consistency.md`、`app/rag/bm25.py` | Step 4 |

> 面试可展开的**完整闭环故事**（稀缺点）：发现真实缺陷 → 量化影响（2/356、+7,122 chunks）→ 跨环境复现（UTC 容器）→ 修复 → 回归测试守护（AST 守卫 + 契约测试）→ 文档归档（`docs/timezone_semantics.md`）。这比"用了什么框架"更能撑住追问。

---

## 3. 简历草稿块（初版可直接粘贴）

```
AI 金融智能投研 Agent（个人项目，求职向二次开发）              2026.08 – 至今
在已答辩的《基于 XGBoost 的 A 股选股分析系统》基础上，构建金融文档检索与 Agent 问答层
· 知识库：12,366 篇文档 / 109,554 chunks（32 只 A 股；cninfo 官方公告含 PDF 正文 + 财经新闻）
· 检索：BGE-M3 向量 + bm25s(jieba) 关键词 → RRF(k=60) 融合 → bge-reranker-v2-m3 重排；
  Qdrant 存储与元数据过滤，热态端到端 32~376ms
· 评测：自建 36 题 / 373 条标注评测集，四组对照（BM25 / Dense / Hybrid / Hybrid+Rerank），
  统一 chunk 预算下带过滤 Recall@5 0.4563、MRR 0.7119；RRF 参数经 12 格网格 + 2-fold 稳定性分析后保持默认
· Agent：LangGraph 工具调用编排，仅暴露 5 个受控工具（无 SQL/Shell/文件），
  结构化错误 + 空结果显式返回，证据带来源字段与正文
· 工程：FastAPI + Docker Compose（Qdrant/API/Web）、218 个自动化测试
· 缺陷修复（自建审计驱动的迭代）：
  1) BM25 后置过滤窗口漏召回，最坏 2/356（16/16 组合均有损失）→ 改为全库打分后过滤；
  2) 日期窗口时区语义不统一，UTC 容器下边界偏移 8 小时、边界日 00:00–07:59:59 数据被漏
     → 统一为显式 UTC+8，边界日命中 44,601 → 51,723 chunks；UTC 容器回归验证通过
```

---

## 4. 现在**不能**写的话（附解锁条件）

| 待解锁表述 | 为什么现在不能写 | 解锁条件 |
| --- | --- | --- |
| "支持**多股票对比分析**" | 审计简报里存在 `stock_comparison → codes[0]`（只查第一只）的嫌疑，尚未验证 | `docs/R1.md` **Batch 2** 审计完成且结论为已修复/无此问题 |
| Agent "最多 N 轮 / 不会死循环 / 终止条件严谨" | `AGENT_MAX_ITERATIONS` 与计划工具执行的语义一致性存疑 | **Batch 2**（Agent termination） |
| "证据链完整、引用**可核验**、无幻觉" 之类**保证级**表述 | 证据校验器的完整性尚未审计；已可写的只是"字段齐全 + 缺字段抛错"（§2 #12） | **Batch 2**（Evidence Validator） |
| SQLite 并发 / WAL / 并发安全 | 未审计 | **Batch 3** |
| "完整解析年报全文 / 无静默截断" | `clean_text` 截断行为未审计 | **Batch 3** |
| 评测方法学结论（如"标签无泄漏""36 题足够支撑结论"） | 未审计 | **Batch 3** |
| 真实问答 trace 截图 / 端到端演示数据 | 未做 Batch 4 的真实 trace | **Batch 4** |
| "生产级 / 上线 / 工业级部署" | 实际是本地 Docker Compose 编排 | 不要写；除非将来真的部署，否则永久禁写 |

**永远不要写**：任何"提升 X%"（评测协议修正过，前后不可比）；`docs/evaluation_before_fix.md` 里的旧数字；"10 万+文档"（109,554 是 chunk 数，文档数是 12,366）。

---

## 5. 单页版 vs 详版

- **一页简历**：用 §3 的块，删掉"缺陷修复 ①"或合并成一行。
- **详版/作品集/面试自述**：用 §2 全表，外加 §2 末尾的闭环故事；面试官追问时按"证据 → 数字 → 我做了什么"三段回答。

---

## 6. 与毕设的关系（避免重复感的表述模板）

> 毕设《基于 XGBoost 的 A 股选股分析系统》：量化选股底座（因子、模型、打分排序、API）。
> 本项目：在其之上补 AI 层（RAG 检索 + Agent 问答），并把工程严谨性拉到可审计水平。

如果简历篇幅紧张，把本项目放在第一、毕设压成一行"底座"即可。

---

## 7. 维护规则（每次更新请遵守）

1. **只增不改数字来源**：任何数字改动，必须同时更新该行的"数字来源"与 §8 变更记录；数字的唯一来源是 `README.md` / `docs/evaluation_after_fix.md`，不要在本文件里重算或挑选更好看的数字。
2. **未验证不下结论**：新表述必须先在仓库里有可核对证据（代码行、测试名、`reports/` 结论、`logs/` 证据文件），否则先放进 §4 等解锁。
3. **Batch 收尾后做一次同步**：Batch 2/3/4 完成时，把对应 §4 条目按审计结论搬进 §2，并在 §8 记一行（日期 + commit + 搬了哪几条）。
4. **README/TODO 收尾后做数字复核**：README 若更新基准表或结论段，逐条核对 §2 的数字，不一致以 README 为准。
5. **不改历史**：本文件是活文档，允许覆盖旧版本；但**审计/评测的原始报告一律不修改**（`reports/2026-09-13_*`、`docs/evaluation_before_fix.md`）。
6. **面试前做一次"逐句可回答"自查**：每一句都要能在 30 秒内指出证据位置；指不出来的句子删掉。

## 8. 变更记录

| 日期 | 变更 | 关联 commit |
| --- | --- | --- |
| 2026-09-19 | 建卡：§2 收录 17 条已核实表述（覆盖 Step 1–5 的修复与评测结论）；§4 列出 8 项待解锁表述，绑定 Batch 2/3/4；§3 生成初版草稿块 | 本文件提交（见 `git log -- reports/resume_project_cards.md`） |
