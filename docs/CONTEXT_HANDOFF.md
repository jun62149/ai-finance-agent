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
- 测试：32 passed / 0 failed。
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

## 六、下一步（Phase 1 建议顺序）

1. P1 金融新闻/公告 ingestion：清洗 → 语义切分 → metadata（documents/chunks 表结构按设计文档 §6-7）。
2. P2-P4：BGE-M3 + Qdrant + BM25 → Hybrid + Reranker（先本地小语料验证）。
3. P5-P6：把现有 3 个 service 函数包装成 Quant Tools（语义与 API 一致）。
4. P7-P8：LangGraph 受控 Agent + 最小前端。
5. P9-P10：≥30 条评测集、对照实验、README/架构图/Demo、Docker。

先决确认项（动手 Phase 1 前问用户）：LLM 供应商与 Key（DeepSeek/通义/GLM/OpenAI 兼容均可，入 .env）、新闻/公告语料范围（建议先 20-50 只活跃股票近 1-2 年）、Qdrant 用 Docker 还是本地向量库。

## 七、启动方式速查

```powershell
cd C:\docm\projects\ai-finance-agent
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
.\.venv\Scripts\python.exe -m pytest tests -q
```
