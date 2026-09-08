# 项目交接文档（从旧聊天框带入新会话）

> 生成日期：2026-09-07
> 用途：Codex 桌面端换了工作区后，新聊天框直接读本文件即可恢复全部上下文。

## 一、这是什么项目

**AI 金融智能投研 Agent**（求职向二次开发，非论文修改）。

基底是已答辩的本科毕设《基于 XGBoost 的 A 股选股分析系统》。目标是最终形成：

```text
XGBoost 毕设 → Quant Engine（已完成，Phase 0）
            → 金融新闻/公告知识库 + BGE-M3 + Qdrant + BM25
            → Hybrid RAG + Reranker
            → LangGraph Agent + Tool Calling
            → Web UI + RAG/Agent 评测 + Docker
```

用户背景：快秋招，目标 AI/LLM 相关岗位，刚学完 RAG 与 Agent；需要可写简历、可面试追问的完整项目。

## 二、当前目录结构（2026-09-07 已重命名）

```text
C:\docm\projects\
├── ai-finance-agent\          ← 当前项目（本聊天新工作区）
│   ├── app\                   quant/services/schemas/api/main
│   ├── scripts\               init_db / rebuild_factors / predict_latest / 一致性验证
│   ├── tests\                 unit + integration（32 passed）
│   ├── docs\
│   │   ├── phase0_report.md       ← Phase 0 完整验证报告
│   │   └── CONTEXT_HANDOFF.md     ← 本文件
│   ├── data\stock_system.db   （gitignore，本地已有）
│   ├── models\                （gitignore，B 全 A 模型）
│   ├── README.md / .env.example / requirements.txt
│   └── .venv                  Python 3.12
└── misc\                      毕设、旧原型、论文脚本、备份等杂项
```

其他相关位置：

- 毕设代码 B 基线（未动）：`C:\docm\work\stock_xgboost_system`
- 毕设 workspace 快照：`C:\docm\projects\misc\stock_xgboost_system`
- Phase 0 备份：`C:\docm\projects\misc\_phase0_backup_20260907_130131`
- 审计报告/迁移计划：`C:\docm\projects\misc\现有项目审计报告与二次开发迁移计划.md`
- 技术设计 + Codex 执行文档：`C:\docm\AI金融智能投研Agent_技术设计与Codex开发文档.docx`

## 三、Phase 0 已交付（不要再重复做）

- 代码基线 = B 版（新功能全），数据 = 1043 只股票 2020-01-02 ~ 2026-06-24，模型 = B 全 A 训练。
- 全量因子重建完成：1,510,748 行 / 1043 只 / 11 因子，耗时约 24 秒。
- 最新预测跑通：983 只入池。
- 一致性验证：旧 B 引擎 vs 新引擎 5/5 完全一致（max_abs_diff = 0）。
- Quant API Contract（FastAPI）：health / factors / prediction / ranking，支持 as_of，结构化错误，无 demo 假数据回退。
- 测试：32 passed / 0 failed（2026-09-08 Phase 1 起步后全量 40 passed，新增 cleaner/chunker 单测）。
- Git：仓库在 `C:\docm\projects\ai-finance-agent`，当前 HEAD 附近 `ff67090`（docs: update paths after workspace reorg）。

## 四、必须遵守的规则（用户明确要求）

1. LLM 不负责量化计算；只做理解、路由、工具调用、证据组织与解释。
2. `strong_prob` 对应分类模型 recall≈0.0005，只能当辅助字段，不能说成可靠上涨概率；不偷改标签、不重训刷指标。
3. API/Agent 无数据时必须返回结构化错误（NoData/StockNotFound/ModelUnavailable 等），禁止 `sample_data` 假数据回退。
4. RAG 输出必须带 source_id/title/date/source/url 可追溯 metadata。
5. Agent 要有工具边界、超时、结构化错误、最大循环次数。
6. 密钥全放 `.env`，DB/pkl/.env/venv 不进 Git。
7. 金融结果必须标注“模型分析而非投资建议”。
8. 原毕设目录（work、misc 内副本）不要大规模改动；新功能都在 `ai-finance-agent`。
9. 每阶段先跑通最小链路再扩展，小步提交可回滚。

## 五、遗留问题（Phase 1 前要心里有数）

- 行情数据截至 2026-06-24，如需“最新”先增量抓取（需要装 akshare/baostock）。
- 本地行情只有 1043/5208 只（600519 等仅有基础信息无日线，API 正确返回 NoData）。
- 全 A 700 万行数据源当前不在磁盘，只有 B 模型与指标留存。
- xgboost 3.4.1 加载 3.3.0 pickle 有旧格式提示，可运行；后续建议 `Booster.save_model`。
- Docker Desktop daemon 未启动；GPU RTX 4070 12GB 可用。

## 六、Phase 1 进展（2026-09-08 更新，下一次直接续跑）

### 已完成

- `app/rag/`：cleaner（HTML/全半角/空白统一，保留数字）、chunker（段落/句子边界，max_chars=600）、sources（East Money 新闻分页 + 公告列表/正文）、ingestion（幂等入库）。
- DB 新增 `documents` / `chunks` 表（字段按设计文档 §6-7，含 content_hash/embedding_version），`init_db` 幂等。
- **去重口径 = 按股票粒度**：`content_hash = sha256(code|title|content)`。市场通稿会命中多只股票，按股各留一份，保证按 stock_code 过滤检索不丢。
- CLI：`scripts/ingest_news.py`、`scripts/ingest_announcements.py`（`--codes` 试点 / 默认 corpus 模式）。
- **语料范围（已定）**：本地因子最新交易日成交额 Top25 + 7 旗舰（600519/601318/600036/000858/601012/300750/002594）≈ 32 只。
- **新闻已全量入库**：7770 docs / 7770 chunks（2025-09-08 ~ 2026-09-08，每只最多 6 页 × 50 条）。
- **公告仅试点**：600519/000333 各 40 篇（正文接口已通）。全量公告抓取**未跑**。
- 测试：40 passed（原 32 + 新 8）。依赖新增 `akshare`。

### 已记录的决策（用户授权 Codex 自定）

- LLM：DeepSeek，`LLM_MODEL=deepseek-v4-flash`，key 只放 `.env`（不提交）。
- 向量库：Phase 2 用 **Qdrant 本地嵌入模式**（`qdrant-client` path 模式，API 与 Docker 服务版同构，之后切服务只改连接地址）；Phase 1 先落 SQLite。
- 不再输出阶段报告；进度靠 git log / README / 本文档。
- 本地因子/行情只有深市主板（000/002 为主）；600519 等旗舰无因子行但新闻语料正常，量化查询按契约返回 NoData 属预期。

### 下次续跑清单（按顺序）

1. `python scripts/ingest_announcements.py --days 365 --max-bodies 300`（32 只全量公告含正文，预计数分钟）。
2. 入库质量抽查：documents/chunks 计数、按股票取样、再跑一遍确认 0 新增（幂等）。
3. P2：`qdrant-client` 本地模式 + BGE-M3，对 chunks 建向量（记录 `embedding_version`），验证"给定 query 返回相关 chunk"最小链路。
4. P3-P4：BM25 + dense + RRF + reranker；P5-P6：把检索与现有 3 个 service 包装成 Tools；P7-P8：LangGraph 受控 Agent + 最小前端；P9-P10：评测集、对照实验、README/Demo/Docker。

### 遗留问题（沿用 Phase 0）

- 行情数据截至 2026-06-24，演示"最新预测"前需增量抓取（见 phase0_report 剩余问题）。
- 全 A 700 万行数据源不在磁盘；模型为 B 全 A 版本，数据口径见报告。
- 新闻正文多为 East Money 搜索摘要（短文本），深度分析主要靠公告正文。

## 七、启动方式速查

```powershell
cd C:\docm\projects\ai-finance-agent
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m pytest tests -q
```
