# AI 金融智能投研 Agent

在已答辩的本科毕设《基于 XGBoost 的 A 股选股分析系统》之上做的**求职向二次开发**：
把遗留量化系统升级为「量化引擎 + 金融知识库 + 受控 Agent + Web UI + 离线评测」的完整作品。

> **定位**：不是"套壳 RAG 聊天框"，而是**在遗留量化系统上做可信 AI 工程化**——
> 量化结果与带来源的文本证据统一编排，且一切可量化评测。

| 维度 | 现状 |
| --- | --- |
| 量化引擎 | XGBoost 双模型（回归 + 分类），1,510,748 行因子 / 1,043 只股票；新旧引擎对拍 **5/5 一致（max_abs_diff=0）** |
| 知识库 | 7,743 篇新闻 + 3,000+ 篇**官方公告（巨潮 cninfo，含 PDF 正文）**，共 **109,554 chunks**（32 只股票 × 近一年） |
| 检索 | BGE-M3 dense + BM25(jieba) → RRF(k=60) → bge-reranker-v2-m3；热态 **32~376ms** |
| Agent | LangGraph 受控工作流，5 个工具，最大 6 轮循环，工具超时 + 结构化错误 |
| 评测 | **36 题评测集 / 373 条标注**，四组对照实验（Recall/Precision/MRR/NDCG/Latency） |
| 测试 | **144 passed** |

---

## 架构

```text
                        ┌──────────────────────────────┐
                        │   Web UI (单页, 无框架/CDN)   │
                        │  报告 · 量化卡片 · 证据来源    │
                        └───────────────┬──────────────┘
                                        │ POST /api/chat, /api/rag/search
                        ┌───────────────▼──────────────┐
                        │        FastAPI Gateway        │
                        │   结构化错误 {code, message}   │
                        └───────────────┬──────────────┘
                                        │
                        ┌───────────────▼──────────────┐
                        │   LangGraph 受控 Agent        │
                        │  query_analyzer               │
                        │  → stock_resolver             │
                        │  → task_planner (确定性计划)   │
                        │  → tool_execution_loop (≤6 轮) │
                        │  → evidence_validator         │
                        │  → report_generator (LLM)     │
                        └──────┬─────────────────┬──────┘
                               │                 │
                 ┌─────────────▼──────┐   ┌──────▼─────────────────────┐
                 │   Quant Tools      │   │   RAG Tool                 │
                 │ get_stock_profile  │   │ search_financial_knowledge │
                 │ get_stock_factors  │   └──────┬─────────────────────┘
                 │ predict_stock      │          │
                 │ rank_stocks        │   ┌──────▼─────────────────────┐
                 └─────────┬──────────┘   │ Hybrid Retrieval           │
                           │              │ dense(BGE-M3) ┐            │
                 ┌─────────▼──────────┐   │ BM25(jieba)   ├→ RRF →Rerank│
                 │ Quant Engine       │   └──────┬────────┘            │
                 │ 因子/预测/评分/排名 │          │                     │
                 └─────────┬──────────┘   ┌──────▼─────────────────────┐
                           │              │ Qdrant (109,554 points)    │
                 ┌─────────▼──────────┐   │ payload: stock/doc_type/ts │
                 │ SQLite             │   └────────────────────────────┘
                 │ 行情/因子/文档/chunks│
                 └────────────────────┘
                           │
                 ┌─────────▼──────────┐
                 │ DeepSeek (LLM)     │  只做理解/路由/组织证据/解释
                 └────────────────────┘
```

**职责边界（铁律）**：LLM **不做量化计算**，只通过工具拿结构化结果；
量化工具与 HTTP API 调用**同一套 service**，错误码完全一致。

---

## 快速开始

### 方式一：Docker（Qdrant + API + UI）

```bash
cp .env.example .env            # 填入 LLM_API_KEY（DeepSeek）
docker compose -f docker/docker-compose.yml up -d --build
# 打开 http://127.0.0.1:8000/
```

GPU 版本（embedding/reranker 走 CUDA，需要 NVIDIA Container Toolkit / WSL2）：

```bash
docker compose -f docker/docker-compose.yml --profile gpu up -d --build api-gpu
```

### 方式二：本地 Python

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
# 可选 GPU：pip install torch --index-url https://download.pytorch.org/whl/cu126

# 数据与模型（不进 Git，需从备份复制）
Copy-Item <备份>\stock_system.db data\
Copy-Item <备份>\models\* models\

# 知识库（首次）
$env:QDRANT_URL = "http://localhost:6333"
.\.venv\Scripts\python.exe scripts/ingest_news.py
.\.venv\Scripts\python.exe scripts/ingest_announcements.py --days 365 --max-bodies 300
.\.venv\Scripts\python.exe scripts/build_embeddings.py
.\.venv\Scripts\python.exe scripts/build_bm25.py

# 启动（UI 与 API 同进程）
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

---

## 效果示例

```text
$ python scripts/ask.py "分析一下京东方A"
intent=stock_analysis (source=rule) stocks=['000725']
plan=[get_stock_profile, predict_stock, get_stock_factors, search_financial_knowledge]
  tool get_stock_profile ok=True   tool predict_stock ok=True
  tool get_stock_factors ok=True   tool search_financial_knowledge ok=True (5 条证据)
evidence=5 evidence_ok=True elapsed=33.8s
```

生成报告的真实片段（体现"可信"而不是"能聊"）：

> **重要前置说明：量化数据的基准日（2026-06-24）早于检索到的多条新闻/公告时间（最晚 2026-08-31）。
> 两者不是同一时点，不能直接拼接为"当前状态"。**
>
> ……`strong_prob` 是**弱信号辅助字段，不代表可靠上涨概率**——不得将其解读为"上涨概率 30.7%"。
>
> ……模型预测 5 日为负收益（-3.07%），但同一基准日的实际动量因子全部为正（ret_5d +16.07%）。
> 二者方向不一致。工具未提供该因子的贡献方向，因此**不做推断**。

---

## 检索评测（P9）

**评测集**：36 题 / 373 条标注文档，覆盖 5 类查询（股票代码 / 公司简称 / 日期约束 /
公告类型与法律术语 / 语义事件问句）。标签由显式规格（股票 + 类型 + 日期窗口 + 标题关键词）
从数据库解析后**冻结**，评测时不再重新推导。
**指标**：Recall@K、Precision@K、MRR、NDCG@K（排名按文档去重后计算）、端到端延迟。

### 带元数据过滤（贴近 Agent 实际用法）

| 配置 | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@5 | 中位延迟 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25-only | 0.4240 | 0.5069 | 0.3944 | 0.6672 | 0.5799 | **4 ms** |
| Dense-only | 0.3784 | 0.4339 | 0.3222 | 0.5736 | 0.4767 | 63 ms |
| Hybrid (RRF) | 0.4109 | 0.5147 | 0.4000 | 0.6755 | 0.5817 | 68 ms |
| **Hybrid + Reranker** | **0.4747** | **0.5414** | **0.4500** | **0.7130** | **0.6464** | 376 ms |

### 不带过滤（更难：必须全库判别）

| 配置 | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@5 | 中位延迟 |
| --- | --- | --- | --- | --- | --- | --- |
| BM25-only | 0.3094 | 0.4378 | 0.3222 | 0.5758 | 0.4667 | **4 ms** |
| Dense-only | 0.2654 | 0.3534 | 0.2389 | 0.4470 | 0.3450 | 26 ms |
| Hybrid (RRF) | 0.3336 | 0.4264 | 0.3444 | 0.5611 | 0.4689 | 30 ms |
| **Hybrid + Reranker** | **0.3833** | **0.4736** | **0.3944** | **0.6435** | **0.5620** | 356 ms |

### 结论（含反直觉处）

1. **Hybrid + Reranker 在两种设定下全面最优**；reranker 是单项收益最大的组件
   （+0.06 Recall@5，代价约 +310ms）。
2. **BM25 单独强于 Dense 单独**：本语料/题集中股票代码、公司简称、法律术语等**精确 token** 权重很高。
3. **融合在无过滤时更有价值**（0.3336 > 0.3094）：候选集大、两路召回互补时 RRF 才有增益；
   有过滤时候选被压小，等权融合会稀释强信号 —— 这是后续可优化点（加权 RRF / 调参）。
4. **日期类查询必须靠元数据过滤**：不过滤时 dense 在 date 类上 Recall@5 掉到 0.0059，
   向量"理解"不了日期。
5. **语义类查询上 dense 反超 BM25**（0.233 vs 0.206），符合预期。

> 复现：`python scripts/evaluate_retrieval.py --top-k 10 [--ignore-filters]`

**评测局限（如实声明）**：标签源自标题关键词，对词面匹配（BM25）天然略有利；
每类仅 5~11 题，跨类比较只作方向性参考；样本量不支持显著性检验。

---

## 工程原则如何落地

| 原则 | 落地方式 |
| --- | --- |
| LLM 不做量化计算 | Agent 只有 5 个工具，无 SQL/Shell/文件工具（有测试守住）；量化工具与 API 共用同一 service |
| 无数据不造数据 | 全部错误结构化（`NoData`/`StockNotFound`/`ModelUnavailable`/`LLMUnavailable`…）；空结果显式 `has_evidence=false`，**无 sample_data 回退** |
| 证据可追溯 | 检索结果强制校验 `chunk_id/document_id/stock_code/doc_type/title/source/publish_time/url`，缺一即抛 `EvidenceMetadataMissing` |
| 弱信号不得夸大 | `strong_prob`（分类 recall≈0.0005）在 API schema、工具返回值、系统 prompt 三处标注"弱信号辅助字段" |
| 工具边界 | 白名单 + 单工具超时 + 最大 6 轮 / 最多 12 次调用；越权调用记为 `ToolNotAllowed` |
| 不静默回退 | `as_of` 无数据返回 `NoData`，绝不回退 latest；向量带 `embedding_version`，不同版本不混库 |
| 密钥不入库 | 全部走 `.env`（已接线到 `os.environ`）；DB / pkl / 向量库 / 模型权重均 gitignore |

---

## API

| Method | Endpoint | 说明 |
| --- | --- | --- |
| GET | `/api/health` | 健康检查（含各表行数） |
| GET | `/api/stocks/{code}/factors?as_of=` | 因子（支持历史时点） |
| GET | `/api/stocks/{code}/prediction?as_of=` | XGBoost 预测 + 风险 + 评分 |
| GET | `/api/stocks/ranking?top_n=&board=` | 选股排名（仅入池股票） |
| POST | `/api/rag/search` | 混合检索，返回带来源证据与四路分数 |
| POST | `/api/chat` | Agent 问答，返回报告 + 执行轨迹 + 量化结果 + 证据 |
| GET | `/` | Web UI |

错误统一为 `{"detail": {"code": "...", "message": "..."}}`。

---

## 项目结构

```text
app/
├── main.py            FastAPI 入口（同时提供 UI）
├── api/               stocks / rag / chat 路由
├── agents/            LangGraph 状态机、策略边界、prompt、LLM 客户端
├── tools/             5 个 Agent 工具 + 注册表 + 结构化错误边界
├── rag/               sources(东财/cninfo) · cleaner · chunker · ingestion
│                      embedding(BGE-M3) · vector_store(Qdrant) · bm25 · hybrid
│                      reranker · retrieval · indexing · eval_metrics · migrate
├── quant/             因子 / 预测 / 评分 / 排名（纯逻辑，源自毕设）
├── services/          Stock / Prediction 服务契约（API 与工具共用）
└── schemas/           Pydantic 输出契约
scripts/               ingest_* · build_embeddings · build_bm25 · evaluate_retrieval
                       ask · migrate_qdrant · bench_retrieval · purge_documents
frontend/              单页 UI（无框架、无 CDN）
docker/                Dockerfile(CPU) · Dockerfile.gpu · docker-compose.yml
data/eval/             冻结的评测集与实验结果
reports/               各阶段任务报告（按日期归档）
```

---

## 数据来源与规模

- **行情/因子**：本地 SQLite（2020-01-02 ~ 2026-06-24，1,043 只股票，154 万日线，151 万因子行）
- **新闻**：东方财富（7,743 篇，近一年）
- **公告**：**巨潮资讯网 cninfo（官方披露平台）**，含 PDF 正文解析 ——
  东财公告正文接口存在 IP 级风控，故改用官方源（并保留了东财作为可选降级源）
- **模型**：XGBoost `xgb-b-20260625-fulla`（Rank IC 0.0737 / Top20 +1.85%）；
  Embedding `BAAI/bge-m3`；Reranker `BAAI/bge-reranker-v2-m3`
- **LLM**：DeepSeek（OpenAI 兼容，`deepseek-v4-flash`）

## 已知限制

1. **行情数据截至 2026-06-24**，而新闻/公告到 2026-09；Agent 会主动提示两者不同时点，
   但演示"最新预测"前应增量抓取行情。
2. **本地行情只覆盖 1,043 只**（深市主板为主）：600519 等有新闻公告但无因子行，
   量化查询按契约返回 `NoData`（**预期行为，非 bug**）。
3. 冷启动首次请求需加载 embedding + reranker（约 30s）；生产部署应在启动钩子预加载。
4. 大年报正文在清洗阶段截断于 20 万字符（963 篇中 11 篇受影响）。
5. 评测集 36 题，类别样本偏小；标签来源偏词面。

## 免责声明

本项目为学习与求职作品。所有输出均为**模型分析结果，非投资建议**。
股市有风险，投资需谨慎。
