# Phase 0 验证报告：Quant Engine 基线统一 + Quant API Contract

> 日期：2026-09-07
> 依据：《Codex Phase 0 执行指令》
> 状态：**Phase 0 完成，已停止，未进入 RAG/Agent 开发**

## 1. 项目

- 新仓库路径：`C:\docm\stock_analysis\ai-finance-agent`
- Git 提交：
  - `6ce07a6` chore: initialize ai-finance-agent repository
  - `ee0ec1d` fix: strict api semantics, as_of factor join, test fixtures, consistency compare
  - 本文档提交后新增 docs commit（见 git log）
- Python 环境：仓库内 `.venv`，Python 3.12.13（`py -3.12 -m venv .venv`）
- 主要依赖：fastapi 0.141.1、uvicorn、pydantic 2、pandas 3.0.5、numpy、scikit-learn、xgboost 3.4.1、joblib、pytest、httpx
- 模型版本标识：`xgb-b-20260625-fulla`

## 2. 备份

备份目录：`C:\docm\stock_analysis\_phase0_backup_20260907_130131`

| 内容 | 位置 | 大小 | SHA256 |
| --- | --- | --- | --- |
| A 项目（代码+数据，不含 .venv/__pycache__/logs） | `_phase0_backup_20260907_130131\A_stock_xgboost_system_full` | 244,429,994 B | - |
| B 项目（代码+模型，不含 .idea/__pycache__/logs） | `_phase0_backup_20260907_130131\B_stock_xgboost_system_full` | 3,112,845 B | - |
| A 数据库独立备份 | `_phase0_backup_20260907_130131\A_stock_system_db_backup.db` | 241,672,192 B | `C7A5FA91D81ABD5E6940AB386722FF04CEA57C32910777B0669D0A64BC8C3746` |
| B 模型：xgb_regressor.pkl | `_phase0_backup_20260907_130131\B_models_backup` | 1,395,033 B | `2D17417D89F63E363EFFD6DB1C5B0AA7739146AF2C21DE3712F94DAB775243FF` |
| B 模型：xgb_classifier.pkl | 同上 | 1,401,643 B | `109C9D48842DD27ED8769BB6B8F783F0E07EE9706D4FC1EA97802110D1643A66` |
| B 模型：feature_columns.json / model_metrics.json | 同上 | 204 / 1,806 B | 见备份 manifest |

原始目录均未删除、未改动。manifest 位于备份目录 `backup_manifest.txt`。

## 3. 数据（新仓库 data/stock_system.db）

| 表 | 行数 |
| --- | --- |
| stock_basic | 5,208 |
| stock_daily | 1,543,560 |
| factor_values | **1,510,748**（重建后） |
| prediction_result | 991（983 新预测 + 8 条历史唯一记录） |
| backtest_result | 309 |

行情时间范围：2020-01-02 ~ 2026-06-24；涉及 1,043 只股票（为本地可用数据的全部）。

## 4. 因子重建

- 股票数：1,043
- 因子数：11（ret_5d/10d/20d、ma5_bias/ma20_bias、volatility_10d/20d、volume_ratio_5d、amount_mean_20d、turnover_mean_20d、drawdown_20d）
- 重建耗时：约 24.1 秒（`scripts/rebuild_factors.py`）
- 结果：成功，factor_values 由 20 只股票的 30,710 行重建为 **1,043 只 / 1,510,748 行**
- 训练池规则（保持 B 版，未修改）：上市 ≥120 日、20 日均额 ≥5000 万、非 ST、单股 ≥60 观测

## 5. 模型与预测

- 模型文件：`models/xgb_regressor.pkl`、`xgb_classifier.pkl`（来自 B 全 A 训练基线）
- 最新预测路径：`scripts/predict_latest.py`（与毕设 predict_latest 同语义）
- 预测股票数：983（满足池过滤的最新交易日）
- 预测耗时：约 18.6 秒（含一次全量因子重算）
- 使用的回归指标（B 基线 metrics，非本阶段重训）：MAE 0.0429 / RMSE 0.0666 / Rank IC 0.0737 / Top20 +1.85%

## 6. 迁移前后一致性（旧 B 引擎 vs 新仓库引擎）

方法：将 A 数据库复制为临时库 → 旧 B 代码在该库上跑预测 → 新仓库 service 以同一交易日（as_of=旧引擎实际使用日期）计算 → 对比因子/预测/风险/评分。

结果：**5/5 一致，max_abs_diff = 0.00e+00**（阈值 1e-8）

| code | 名称 | 交易日 | 因子(max diff) | pred_return_5d | strong_prob | risk_score | score |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 000006 | 深振业Ａ | 2026-06-24 | 0 | 0.005522 | 0.371884 | 0.150816 | 4.484244 |
| 000008 | 神州高铁 | 2026-06-24 | 0 | 0.003370 | 0.285308 | 0.066129 | 3.871099 |
| 000009 | 中国宝安 | 2026-06-24 | 0 | 0.000540 | 0.289832 | 0.077368 | 3.614292 |
| 000011 | 深物业A | 2026-05-29 | 0 | -0.000049 | 0.255868 | 0.085677 | 2.977557 |
| 000333 | 美的集团 | 2026-06-24 | 0 | -0.002723 | 0.326504 | 0.061978 | 4.073534 |

期间发现并记录了一个**语义差异（非算法差异）**：`000011` 在 2026-06-24 存在因子行，但当日不满足推荐池流动性条件，旧引擎“最新可推荐预测”取到 2026-05-29。新引擎明确区分「个股因子/预测查询」（任意有数据的日期）与「选股排名」（仅入池股票），两者均已按同一日期对拍一致。

复现脚本：`scripts/verify_consistency.py` + `scripts/legacy_baseline_runner.py`。

## 7. API（FastAPI 最小集合）

启动：`.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000`

### GET /api/health

```json
{
  "status": "ok",
  "service": "ai-finance-agent-quant",
  "model_version": "xgb-b-20260625-fulla",
  "db_exists": true,
  "db_tables": {
    "stock_basic": 5208,
    "stock_daily": 1543560,
    "factor_values": 1510748,
    "prediction_result": 991,
    "backtest_result": 309
  }
}
```

### GET /api/stocks/000333/factors

```json
{
  "code": "000333",
  "name": "美的集团",
  "as_of": "2026-06-24",
  "board": "main",
  "ret_5d": -0.030142,
  "ret_10d": -0.07018,
  "ret_20d": -0.037335,
  "ma5_bias": -0.018583,
  "ma20_bias": -0.053738,
  "volatility_10d": 0.017474,
  "volatility_20d": 0.015458,
  "volume_ratio_5d": 1.006668,
  "amount_mean_20d": 2719439671.8275,
  "turnover_mean_20d": 0.49,
  "drawdown_20d": -0.108498,
  "close": 76.58,
  "pct_change": -1.91
}
```

`as_of` 支持历史时点：`/api/stocks/000333/factors?as_of=2026-03-31` 返回 2026-03-31 当天因子；`as_of=2019-12-30`（无数据）返回结构化错误：

```json
{"detail": {"code": "NoData", "message": "no factor data for stock 000333 as_of=2019-12-30"}}
```

### GET /api/stocks/000333/prediction

```json
{
  "code": "000333",
  "name": "美的集团",
  "as_of": "2026-06-24",
  "pred_return_5d": -0.002723,
  "strong_prob": 0.326504,
  "risk_score": 0.061978,
  "score": 4.073534,
  "model_version": "xgb-b-20260625-fulla",
  "is_historical_recomputation": false,
  "close": 76.58,
  "pct_change": -1.91,
  "board": "main",
  "main_factor": "volume confirmation"
}
```

### GET /api/stocks/ranking?top_n=3

```json
[
  {"code": "001211", "name": "双枪科技", "rank": 1, "score": 6.361093, "pred_return_5d": 0.025656, "risk_score": 0.246079, "strong_prob": 0.459843, "board": "main", "main_factor": "volume confirmation"},
  {"code": "000566", "name": "海南海药", "rank": 2, "score": 6.279943, "pred_return_5d": 0.013821, "risk_score": 0.017981, "strong_prob": 0.361548, "board": "main", "main_factor": "volume confirmation"},
  {"code": "000736", "name": "中交发展", "rank": 3, "score": 5.582866, "pred_return_5d": 0.018568, "risk_score": 0.172025, "strong_prob": 0.394033, "board": "main", "main_factor": "volume confirmation"}
]
```

错误契约（结构化 JSON，禁止静默回退、禁止 demo 假数据）：

| 场景 | HTTP | code |
| --- | --- | --- |
| 代码格式非法 | 400 | InvalidStockCode |
| 股票不存在 | 404 | StockNotFound |
| 指定日期无数据 | 404 | NoData |
| 历史预测不可支持 | 501 | HistoricalPredictionUnavailable |
| 模型文件缺失 | 503 | ModelUnavailable |
| 数据库错误 | 500 | DatabaseError |

## 8. 测试

```text
32 passed, 0 failed, 0 skipped（3 条第三方 deprecation warnings）
```

覆盖：代码归一化、因子计算与手工核对、评分公式、预测/历史 as_of、NoData/StockNotFound/ModelUnavailable、空库不产生假数据、health、JSON 响应、ranking 排序与 board 过滤、API 不允许 sample_data fallback。

## 9. 剩余问题（如实列出）

1. **数据时效**：行情截至 2026-06-24，本阶段未做增量抓取（需要 akshare/baostock 依赖与网络，属后续数据更新任务）。
2. **行情覆盖**：本地 stock_daily 只有 1,043 只股票；贵州茅台(600519)、中国平安(601318) 等仅有基础信息、无日线因子，API 按契约返回 NoData。毕设论文中“约 5200 只/700 万行”的全量数据当前不在磁盘（B 库已于 7/29 清空），模型文件保留但数据源需恢复才能复现全 A 预测。
3. **strong_prob 质量**：分类模型 recall 极低（0.0005），API/schema 已标注“辅助字段、弱信号”，未优化、未重训。
4. **模型序列化**：xgboost 3.4.1 加载 3.3.0 生成的 pickle 时提示旧格式，可运行；建议后续用 `Booster.save_model` 统一固化。
5. **归一化口径**：服务层要求 6 位数字代码；展示层如需接受“茅台/600519.x”等别名，属于后续股票解析模块（Phase 1 Agent 需要），本阶段未实现。
6. **prediction_result 计数 991**：包含 983 条最新预测与 8 条旧日期历史唯一行，属 upsert 预期结果。
7. `HistoricalPredictionUnavailable` 错误类型已定义，但本阶段对“有因子行”的历史日期直接提供 point-in-time 重算（`is_historical_recomputation=true`，可复现、不含未来标签）；若产品上需要“当年实际发布预测存档”，需引入预测发布记录表，属后续任务。

## 10. 下一阶段建议（仅建议，不执行）

1. **P1 知识库 ingestion**：金融新闻/公告采集 → 清洗 → 语义切分 → metadata（document_id/chunk_id/stock_code/doc_type/publish_time/source/url/content_hash）。
2. **BGE-M3 + Qdrant**：先本地小语料验证，再上向量库与 BM25 混合检索（Qdrant 需启动 Docker Desktop 或使用本地向量库降级方案）。
3. **Quant Tools 契约**：把本阶段 3 个 service 函数包装为 Agent 工具（get_stock_factors / predict_stock / rank_stocks），保持与 API 相同的错误语义。
4. **数据恢复/更新**：决定是否恢复全 A 行情（当前 1,043 只子集已足够开发，但全 A 覆盖对“选股”口径更完整）。
5. **股票解析与 as_of 产品化**：为 Agent 增加公司名/简称→代码解析；为预测增加“发布记录”语义后再放开 HistoricalPrediction 能力。
6. 后续每一阶段保持：先最小链路、再评测；README 更新；最终补 RAG/Agent 对照实验、架构图与简历描述。
