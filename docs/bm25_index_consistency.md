# BM25 索引一致性校验（Step 4）

> 日期：2026-09-13 ｜ 代码：`fix(index): add bm25 manifest validation`
> 目标：消除 **BM25 索引 / Qdrant collection / SQLite chunk 语料** 三者之间**静默不一致**的风险。
> 原则：**小改动**，不引入版本管理系统。

## 1. 问题：三份存储可以静默漂移

```text
SQLite chunks  ──(build_embeddings)──►  Qdrant collection（向量 + payload）
      │
      └────────(build_bm25)──────────►  data/bm25/（bm25s 索引 + chunk_ids + metadata）
```

检索时两路各自独立取数：dense 走 Qdrant（命中后按 `point_id` 回表取正文），BM25 走本地索引
（命中后同样按 `point_id` 回表取正文）。**没有任何一步校验三者是否描述同一份语料。**

两种失败模式（都是静默的）：

| 模式 | 触发 | 后果 |
| --- | --- | --- |
| **漏召回** | 新语料已入库并已嵌入，但忘记重建 BM25 | 词面分支永远看不到新 chunk，指标悄悄变差 |
| **张冠李戴** | chunk 被重灌（`chunks.id` 重新分配），BM25 元数据里的旧 `point_id` 指向**另一条 chunk** | 回表取到**别人的正文**；而证据校验原先只查 `chunk_id/document_id/.../url`，**不查 `content`**，于是错误正文会一路进到 LLM 报告 |

> 注：`EMBEDDING_VERSION` 变化本身**不会**让 BM25 失效（chunk id 与正文不变）；
> 真正危险的是**chunk 集合变化**——此前代码里没有任何区分与提示。

## 2. 设计：一份最小 manifest + 加载时校验一次

### 2.1 manifest（`data/bm25/manifest.json`，由 `build_bm25.py` 写出）

```json
{
  "schema_version": 1,
  "embedding_version": "bge-m3-v1",
  "all_chunks": false,
  "chunk_count": 109554,
  "chunk_id_hash": "333554725291f85e",
  "collection": "financial_chunks",
  "tokenizer": "jieba-cut_for_search",
  "bm25s_version": "0.3.11",
  "built_at": "2026-09-13 22:06:39"
}
```

- `chunk_id_hash` = `sha256("|".join(sorted(chunk_ids)))[:16]` —— **与顺序无关**，能发现"数量相同但集合不同"的情况；
- `embedding_version` + `all_chunks` 记录**建索引时用的取数规则**，校验时用同一规则回查 SQLite（保证可比）。

### 2.2 校验时机与行为

| 场景 | 行为 |
| --- | --- |
| 加载时（`BM25Index.load()`，即检索组件初始化） | 比对 `chunk_count` + `chunk_id_hash`；**每次检索不校验**（成本考虑，见 §4） |
| 一致 | INFO 一行：`BM25 index verified: 109554 chunks (hash …, embedding_version=…)` |
| **不一致** | 抛结构化错误 **`IndexOutOfSync`**（HTTP 503），消息直接给出重建命令 |
| 不一致 + `BM25_ALLOW_STALE=1` | 降级为 **WARNING** 并继续（离线演示/评测可用） |
| **manifest 缺失**（Step 4 之前建的索引） | WARNING 提示"无法校验，重建即可启用"，**仍然可用**（向后兼容） |

### 2.3 顺带堵掉的第二个静默失败

`hybrid_search` 现在会**丢弃"命中但回表取不到正文"的 chunk**，并计数/告警
（返回字段 `dropped_empty_content`）。同时 `rag_tools._to_evidence` 把 `content` 纳入**必填校验**——
**LLM 读不到的引用不算证据**。

## 3. 真实环境验证（三条路径都实测）

```text
正向  BM25Index.load() → INFO BM25 index verified: 109554 chunks (hash 333554725291f85e, embedding_version=bge-m3-v1)
负向  篡改 manifest 的 chunk_id_hash →
      IndexOutOfSync: BM25 index is out of sync: built with 109554 chunks (hash deadbeefdeadbeef)
      but the corpus now has 109554 (hash 33355472…); rebuild it with `python scripts/build_bm25.py`
降级  BM25_ALLOW_STALE=1 → 打印 WARNING 后继续加载（docs = 109554）
```

## 4. 成本

| 项 | 实测 |
| --- | --- |
| 仅校验（SQLite 查 109,554 个 chunk_id + 排序 + 哈希） | **≈ 1073 ms** |
| 完整 `BM25Index.load()`（含 bm25s 索引反序列化 + 元数据 pickle + 校验） | **≈ 1586 ms** |
| 单次检索（BM25，全库窗口） | 8~20 ms |

**结论**：校验成本约 1 秒，**只在进程内首次加载时发生一次**（组件被 `get_components()` 缓存），
不进入每次检索路径。若未来进程频繁冷启动且在意这 1 秒，可把哈希换成
"数量 + 抽样 id 校验"，但在当前规模下没有必要。

## 5. 明确不做的事（避免过度设计）

1. **不做**完整版本管理系统/迁移框架：一次哈希 + 一个整数即可覆盖当前失效模式；
2. **不做**每次 query 校验：成本与收益不匹配；
3. **不做**自动修复（不自动重建索引）：静默重建会掩盖"为什么漂移"的问题；
4. **不做** dense 侧的 payload 指纹：Qdrant 的写入路径与 SQLite 同一事务化流程（`upsert_chunks`），
   且检索时必须回表取正文，真正的风险点已在 §2.3 覆盖 —— 若将来出现跨机写入，再补这一层；
5. **不做**分片索引/稀疏向量（Step 2 已决）。

## 6. 测试

`tests/unit/test_bm25_manifest.py`（9 个）：manifest 字段完整性、正向加载、
**语料增长 → 阻断**、`BM25_ALLOW_STALE` 降级、**缺 manifest → 警告且仍可用**、
哈希被篡改 → 阻断、`chunk_id_hash` 与顺序无关、`verify()` 报告数字、
**空正文命中被丢弃并计数**。

全套：**182 passed**。

## 7. 运维动作（发生 `IndexOutOfSync` 时）

```powershell
.\.venv\Scripts\python.exe scripts/build_bm25.py          # 重建索引并刷新 manifest（约 138s / 109,554 chunks）
.\.venv\Scripts\python.exe -c "from app.rag.bm25 import BM25Index; print(len(BM25Index.load()))"
```
