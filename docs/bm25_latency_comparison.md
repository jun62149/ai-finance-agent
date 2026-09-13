# BM25 候选窗口：延迟对比（修复前 / 修复后）

> 背景：评审 batch 1 的 Issue 1 —— BM25 原先只对「全局词面 top-200」打分后再做 metadata 过滤，
> 窄过滤条件下会**静默漏召回**（实测 `stock_code+doc_type` 场景：2 命中 vs 全量 356）。
> 修复：`BM25_WINDOW=0`（默认）＝ 全库打分后再过滤；`BM25_WINDOW>0` ＝ 显式截断（会打 WARNING）。

## 测量条件

```text
语料        109,554 chunks（32 只股票：公告 cninfo + 新闻 eastmoney）
测量对象    BM25Index.search_with_diagnostics() —— 真实代码路径，非模拟
每场景      7 次取中位；"最差"为 7 次样本中的最大值
top_k       20（与评测的 final-k 一致）
环境        Windows + 本地 Python 3.12（非容器）
```

## 修复前后对比

| 场景 | 修复前 `window=200` 中位 / 最差 | 修复后 全库 中位 / 最差 | 延迟增量 | 命中数 200 → 全库 |
| --- | --- | --- | --- | --- |
| 普通 query（无过滤） | 0.3 / 533.3 ms | **9.9 / 10.0 ms** | +9.6 ms | 20 → 20 |
| `stock_code` + `doc_type` | 3.0 / 6.8 ms | **20.0 / 37.2 ms** | +17.0 ms | **2 → 20** |
| `date_from` 过滤 | 2.7 / 9.0 ms | **8.3 / 9.2 ms** | +5.6 ms | 20 → 20 |
| `stock` + `doc_type` + `date` | 0.4 / 0.9 ms | **9.9 / 10.1 ms** | +9.5 ms | 17 → 20 |

> **离群值说明**：第一行"533.3 ms"是 7 次样本中的单次离群（首次调用/GC 抖动），中位数为 0.3 ms；
> 修复后同场景最差仅 10.0 ms，反而更稳定。

## 结论

1. **没有出现 100 ms 量级的增长**：修复后最坏中位 **20.0 ms**、最差样本 **37.2 ms**；
2. 修复后 BM25 全库扫描（8~20 ms）**仍低于 dense 单路**（同机实测 28~47 ms），
   因此 hybrid 的总延迟不会发生量级变化（实测 hybrid 32~41 ms，hybrid+rerank ~450 ms，主要耗时在 reranker）；
3. 相对增幅最大的是"窄过滤"场景（3.0 → 20.0 ms，约 6.7×），但绝对值仍在 20 ms 级；
4. **收益是正确性而非速度**：`stock_code + doc_type` 场景从 **2 命中 → 20 命中**（全库下该类过滤实际有 356 条可召回），
   即修掉了"过滤条件下 BM25 候选被截断导致的漏召回"。

## 如何回到"截断模式"

语料继续增长到百万级、需要控制单次检索成本时，可用环境变量显式开启（会打印 WARNING）：

```powershell
$env:BM25_WINDOW = "5000"      # 0 = 全库（默认）；>0 = 只打分前 N 条候选
.\.venv\Scripts\python.exe scripts/search_hybrid.py "回购 进展" --doc-type announcement
# 日志： BM25 search: window=5000 (TRUNCATED) scanned=5000 matched=12 returned=12 elapsed_ms=1.8
#        WARNING BM25 candidate truncation is active (BM25_WINDOW=5000 < corpus=...)
```

## 如何自查（trace 可解释性）

SQLite/接口侧同名字段会随检索结果一起返回，UI 执行轨迹、评测 JSON、Agent 报告都可见：

```text
bm25_window        本次使用的窗口（= 语料规模表示不限）
bm25_truncated     是否真的发生了截断
bm25_scanned       实际扫描的候选数（返回满 top_k 即提前结束）
bm25_after_filter  扫描范围内通过 metadata 过滤的候选数
bm25_corpus        语料总规模
```
