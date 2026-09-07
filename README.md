# ai-finance-agent

在已答辩的本科毕设《基于 XGBoost 的 A 股选股分析系统》之上进行的求职向二次开发。

**Phase 0 目标**：把原 XGBoost 毕设工程化为可靠的 Quant Engine，并建立稳定、无假数据回退的 Quant API Contract。本阶段**不包含** RAG / Agent / LangGraph / 前端。

## 核心原则

- 原毕设目录（`C:\docm\stock_analysis\stock_xgboost_system`、`C:\docm\work\stock_xgboost_system`）保持不动，本仓库为独立工程。
- LLM（后续阶段）不负责量化计算，只通过 Service/Tool Contract 调用 Quant Engine。
- 所有 API 返回真实数据；无数据时返回结构化错误（`NoData` / `StockNotFound` / `ModelUnavailable` 等），**禁止** demo 假数据回退。
- `strong_prob` 对应分类模型 recall 极低，仅作为辅助字段，不得表述为可靠上涨概率。

## 数据与模型来源（Phase 0 基线）

- 行情数据：`C:\docm\stock_analysis\stock_xgboost_system\data\stock_system.db`（2020-01-02 ~ 2026-06-24，5208 只股票基础信息、154 万+ 日线记录、1043 只股票全量历史）。
- 代码语义基线：`C:\docm\work\stock_xgboost_system`（含板块/流动性过滤、过拟合诊断、TopN 等最新逻辑）。
- 模型基线：`C:\docm\work\stock_xgboost_system\models`（全 A 训练，Rank IC 0.0737 / Top20 +1.85%）。

数据库与模型 pickle 不进入 Git，首次使用需从备份/原目录复制：

```powershell
Copy-Item C:\docm\stock_analysis\stock_xgboost_system\data\stock_system.db data\
Copy-Item C:\docm\work\stock_xgboost_system\models\* models\
```

## 环境与运行

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

初始化数据库迁移（补列）并重建全量因子：

```powershell
.\.venv\Scripts\python.exe scripts\init_db.py
.\.venv\Scripts\python.exe scripts\rebuild_factors.py
```

运行量化预测：

```powershell
.\.venv\Scripts\python.exe scripts\predict_latest.py
```

启动 Quant API：

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

## Quant API（Phase 0.5）

```text
GET /api/health
GET /api/stocks/{code}/factors?as_of=YYYY-MM-DD
GET /api/stocks/{code}/prediction?as_of=YYYY-MM-DD
GET /api/stocks/ranking?top_n=10&board=main
```

`as_of` 语义：返回该日期（含）之前最近一个交易日的因子/预测，不自动回退到 latest；无数据返回 `NoData`。

错误均为结构化 JSON：`{"detail": {"code": "...", "message": "..."}}`。

## 目录

```text
app/
├── main.py            # FastAPI 入口
├── api/stocks.py      # Quant API 路由
├── quant/             # 因子 / 预测 / 评分 / 排名（纯逻辑，源自毕设 service）
├── services/          # Stock / Prediction 服务契约
├── schemas/           # Pydantic 输出契约
├── config.py          # 环境变量配置
├── errors.py          # 结构化错误
└── db.py              # SQLite 访问（迁移自毕设）
scripts/               # init_db / rebuild_factors / predict_latest / 一致性验证
tests/                 # unit + integration
docs/                  # 开发文档与 Phase 验证报告
```

## 免责声明

本项目为学习与求职作品，模型输出仅为技术演示，不构成投资建议。
