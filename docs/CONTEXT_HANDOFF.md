# 项目总览交接文档（新会话唯一入口）

> 生成日期：2026-09-08
> 用途：换对话框时，让新会话先读本文件获得**完整项目总览**；读完后先向用户复述理解并等待确认，**先不要干活**。
> 历史增量文档与报告已归档到 `docs/archive/`，本文件是合并后的总览。

## 0. 协作约定（用户明确要求，必须遵守）

1. **全程只用一个 Codex 实例**（当前模型配置：DeepSeek v4-flash），不换模型、不引入第二个 Agent。
2. **不再输出阶段报告**；进度靠 git log / README / 本文档体现。
3. **每次完成一个任务后，必须声明"下一个任务适合 deepseek-v4-flash 还是 deepseek-v4-pro"**（给建议并说明理由，供用户在新会话选择模型时参考）。
4. 向量库、语料范围等技术细节由 Codex 自定，不用再逐项问用户。
5. 新会话开场流程：读本文件 → 总览项目 → 复述理解 → 等用户确认后再动工（用户会粘贴开场白）。

## 1. 这是什么项目

**AI 金融智能投研 Agent**——求职向二次开发（非论文修改），秋招目标为 AI/LLM 应用岗位。

基底是已答辩本科毕设《基于 XGBoost 的 A 股选股分析系统》。目标形态：

```text
XGBoost 毕设 → Quant Engine（Phase 0 已完成 ✅）
             → 金融新闻/公告知识库 + BGE-M3 + Qdrant + BM25（Phase 1-4 进行中）
             → Hybrid RAG + Reranker
             → LangGraph Agent + Tool Calling
             → Web UI + RAG/Agent 评测 + Docker
```

卖点定位：在遗留量化系统上做可信 AI 工程化——量化结果与带来源的文本证据统一编排 + 可量化评测，而不是"套壳 RAG 聊天框"。

## 2. 项目铁律（源自技术设计文档 §0，长期有效）

- LLM 不负责量化计算；只做理解、路由、工具调用、证据组织与解释。
- `strong_prob`（分类 recall≈0.0005）只能当辅助字段，不得称为可靠上涨概率；不偷改标签、不重训刷指标。
- API/Agent 无数据时必须返回结构化错误（NoData/StockNotFound/ModelUnavailable 等），禁止 sample_data 假数据回退。
- RAG 输出必须带 source_id/title/date/source/url 可追溯 metadata。
- Agent 要有工具边界、超时、结构化错误、最大循环次数。
- 密钥全放 `.env`；DB/pkl/.env/venv 不进 Git。
- 金融结果必须标注"模型分析而非投资建议"。
- 原毕设目录（`C:\docm\work\stock_xgboost_system`、`C:\docm\stock_analysis\stock_xgboost_system`）不做大规模改动。
- 每阶段先跑通最小链路再扩展，小步提交可回滚。

## 3. 文档来源与顺序（归档于 docs/archive/）

1. [技术设计与 Codex 开发文档.docx](archive/AI金融智能投研Agent_技术设计与Codex开发文档.docx)——**用户最早发的总纲**：用户先与网页版 GPT 讨论，由 GPT 产出的技术设计文档后转交给 Codex。定义最终形态、技术栈、DB/API/评测设计、P0~P10、铁律。
2. [01_现有项目审计报告与二次开发迁移计划.md](archive/01_现有项目审计报告与二次开发迁移计划.md)——**Codex 回复①**：只读审计毕设 A/B 两份快照，提出"B 代码 + A 数据 + B 模型"基线与迁移计划。
3. [02_Codex Phase 0 执行指令.txt](archive/02_Codex%20Phase%200%20执行指令-项目基线统一与Quant%20A.txt)——**用户回复**：拍板基线，下达 Phase 0（备份→Git→迁移 Quant Engine→API Contract→as_of→禁假数据→结构化错误→测试→验证报告）。
4. [03_phase0_report.md](archive/03_phase0_report_Phase0验证报告.md)——**Codex 回复②**：Phase 0 验证报告（备份清单、数据规模、因子重建、一致性 5/5、API 示例、32 passed、剩余问题、下一阶段建议）。仓库内 `docs/phase0_report.md` 为同内容修订版。
5. [04_CONTEXT_HANDOFF_增量版_2026-09-08.md](archive/04_CONTEXT_HANDOFF_增量版_2026-09-08.md)——旧增量交接：Phase 0 完成状态 + 2026-09-08 Phase 1 起步进展。
6. **本文件**：合并上述全部的项目总览交接。

## 4. 当前状态

### Phase 0：Quant Engine（已完成，已验证）

- 仓库：`C:\docm\projects\ai-finance-agent`（git），Python 3.12 `.venv`，HEAD 以 `git log` 为准。
- 代码基线 = B 版（新功能全）；数据 = 1043 只股票 2020-01-02 ~ 2026-06-24；模型 = B 全 A 版（`xgb-b-20260625-fulla`）。
- 全量因子：1,510,748 行 / 11 因子；最新预测 983 只入池；新旧引擎对拍 5/5 一致（max_abs_diff=0）。
- FastAPI：health / factors / prediction / ranking，支持 as_of、结构化错误、无假数据回退。
- 备份：`C:\docm\projects\misc\_phase0_backup_20260907_130131`（A 库、B 模型等，含 manifest）。

### Phase 1：新闻/公告 ingestion（进行中，新闻已完成）

- `app/rag/`：cleaner、chunker（语义边界，max_chars=600）、sources（East Money）、ingestion（幂等）。
- DB 新增 `documents`/`chunks` 表（字段按设计文档，含 content_hash/embedding_version），`init_db` 幂等。
- 去重口径 = **按股票粒度** `sha256(code|title|content)`：市场通稿多股各留一份，保证按 stock_code 过滤不丢。
- 语料（已定）：本地因子最新日成交额 Top25 + 7 旗舰（600519/601318/600036/000858/601012/300750/002594）≈ 32 只。
- **新闻已全量入库**：7770 docs / 7770 chunks（2025-09-08 ~ 2026-09-08）。
- **公告仅试点**（600519/000333 各 40 篇，正文接口已通）；**全量公告未跑 = 下一个任务**。
- 测试：40 passed（Phase 0 32 + 新增 cleaner/chunker 8）。

## 5. 已定决策（无需再问用户）

| 项 | 决策 |
| --- | --- |
| LLM 供应商 | DeepSeek（OpenAI 兼容），`LLM_BASE_URL=https://api.deepseek.com`，key 在 `.env` |
| LLM_MODEL | `deepseek-v4-flash` |
| 向量库 | Phase 2 用 **Qdrant 本地嵌入模式**（qdrant-client path 模式，与 Docker 服务版 API 同构，之后切服务只改连接地址） |
| 语料范围 | 32 只股票 × 近 1 年新闻/公告，新闻已入库 |
| 阶段报告 | 不再输出；进度靠 git log / README / 本文档 |
| 本地数据局限 | 因子/行情只覆盖深市主板为主的 1043 只；600519 等旗舰无因子行，量化查询按契约返回 NoData（预期行为，非 bug） |

## 6. 下一次任务（按顺序）

1. **公告全量抓取**：`python scripts/ingest_announcements.py --days 365 --max-bodies 300`（32 只，含正文，预计数分钟；幂等可重跑）。
2. 质量抽查：documents/chunks 计数、按股票取样、重跑确认 0 新增。
3. P2：`qdrant-client` 本地模式 + BGE-M3 对 chunks 建向量（记录 `embedding_version`），验证 query→相关 chunk 最小链路。
4. P3-P4：BM25 + dense + RRF + reranker。
5. P5-P6：把检索与 3 个 quant service 包装成 Tools（语义与 API 一致）。
6. P7-P8：LangGraph 受控 Agent + 最小前端。
7. P9-P10：≥30 条评测集、对照实验、README/架构图/Demo/Docker。

## 7. 启动方式速查

```powershell
cd C:\docm\projects\ai-finance-agent
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m pytest tests -q
.\.venv\Scripts\python.exe scripts/ingest_news.py --codes 600519
.\.venv\Scripts\python.exe scripts/ingest_announcements.py --codes 600519 --days 365 --max-bodies 40
```

## 8. 遗留问题

- 行情数据截至 2026-06-24；演示"最新预测"前需增量抓取（akshare/baostock）。
- 全 A 700 万行数据源不在磁盘；模型为 B 全 A 版，口径见 phase0_report。
- 新闻正文多为 East Money 搜索摘要（短文本），深度分析主要靠公告正文。
- xgboost 3.4.1 加载旧 pickle 有提示，可运行；后续建议 `Booster.save_model` 固化。
