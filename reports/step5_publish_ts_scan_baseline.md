# Step 5 · 5-2 前置扫描 baseline（publish_ts 一致性）

> 运行时间：2026-09-19 15:24–15:26（CST）｜扫描器 commit：`dbba569`（文件 sha256 前 16 位 `42a8855535d7896b`）
> 证据原文：`logs/step5_publish_ts_scan_raw.txt`（宿主）、`logs/step5_publish_ts_scan_container_pre.txt`（容器）
> 性质：**5-2 的唯一正式结论来源**。按批准的执行计划，本文件即"扫描结果"，回填决策只依据它。

## 1. 结论

**109,554 点全部一致：mismatch = 0、missing = 0 → 不执行回填。**

存量 `publish_ts` 与"由 DB 原始 `publish_time` 字符串在显式 UTC+8 下重建的期望值"逐点相等，且 payload / documents / chunks 三处 `publish_time` 完全自洽，`db` 与 `qdrant` 的 chunk 集合互为对方子集为空（1:1）。按预先写死的决策规则：**不运行 `backfill_publish_ts.py`，只记录**（`logs/backfill_ts.txt` 显示 109,554 个值全部由 2026-09-13 那一次宿主进程写入，本次扫描确认此后没有被 UTC 进程污染）。

## 2. 环境与 provenance

| 项 | 宿主 | 容器 |
| --- | --- | --- |
| 进程时区 | `utc_offset=+08:00` | **`utc_offset=0`（UTC）** |
| Qdrant 目标 | `http://localhost:6333` | `http://qdrant:6333` |
| `git_head` | `87dcac2 +dirty`（扫描时该修复尚未提交） | `unknown (git unavailable)`（未挂载 `.git`） |
| `app/timeutil.py` | `876198f720da5403` | 同 |
| `app/rag/vector_store.py` | `94917f53883ebd54` | 同 |
| `app/rag/retrieval.py` | `239d4a571693a0c6` | 同 |
| `scripts/scan_publish_ts.py` | `42a8855535d7896b` | 同 |
| Qdrant 服务端 | v1.19.1，`points_count=109554`，`segments_count=6`，`optimizer_status=ok` | 同 |

宿主的 `+dirty` 就是扫描器自身的最后两处修改，内容随后原样提交为 `dbba569`，**sha256 未变**，可直接复核。容器以 bind mount 固定代码版本（`app/` + `scripts/`），因此两次运行的代码字节完全一致，唯一变量是进程时区。

**环境事件（记录在案）**：本次会话期间 Docker Desktop 自动更新并重启引擎（`29.6.2 → 29.8.0`，`Docker Desktop Installer` 于 15:08:32 启动）。该重启**确实掐断过一次**扫描，报错为 `ResponseHandlingException: [WinError 10054] 远程主机强迫关闭了一个现有的连接`（`logs/` 中该次运行未落库，见 §7）。此事件不是本次"停顿"的原因，原因见 §7。

## 3. R1 / R2 / R3 执行确认

- **R1（反同义反复）**：`expected_cst` 只由 `to_market_epoch(raw)` 构造，`raw` 来自 DB；扫描器源码中不存在任何 `fromtimestamp(stored)` 形式的期望值构造。`stored` 仅用于与 `expected_cst` 相减。
- **R2（来源分层）**：主锚点 = SQLite `chunks.publish_time`（`mode=ro` 打开）；Qdrant `payload.publish_time` 与 DB 单独比对；`documents.publish_time` 与 `chunks.publish_time` 单独比对。三者互不覆盖。
- **R3（尺子自证）**：四个固定输入/输出与测量结果同文件输出，`calibration_ok = True`。
- **只读判据（按代码/API 路径，不按存储目录 mtime）**：全文件对 `client.` 的调用只有 `scroll` 与 `count`；无 `upsert` / `set_payload` / `delete_*` / `create_*` / `batch_update_points` / `update_vectors` 等任何 mutation 调用；SQLite 以 `file:...?mode=ro` URI 打开，**结构上无法写入**。`docker/qdrant_storage` 的目录 mtime 不作为"是否写入"的依据（容器启动会重写 segment 簿记文件，与数据无关）。

## 4. Baseline totals（两侧逐项对照）

| 指标 | 宿主（UTC+8 进程） | 容器（UTC 进程） |
| --- | --- | --- |
| `db_chunks` / `db_documents` | 109,554 / 12,366 | 109,554 / 12,366 |
| `qdrant_points` | 109,554 | 109,554 |
| `points_scanned` | **109,554** | **109,554** |
| `ok` | **109,554** | **109,554** |
| `utc_written`（delta = −28800） | **0** | **0** |
| `other` | **0** | **0** |
| `missing_publish_ts` | **0** | **0** |
| `unparsable_raw` | **0** | **0** |
| `point_without_db_row` | **0** | **0** |
| `db_chunk_without_point` | **0** | **0** |
| delta 直方图 | 空 | 空 |
| `payload.publish_time` vs DB | **0** 不一致 | **0** 不一致 |
| `documents` vs `chunks` `publish_time` | **0** 不一致 | **0** 不一致 |
| `scroll_retries` | 0 | 0 |
| 全量遍历耗时 | 3.3s | 3.4s |
| DB 读取耗时 | 0.8s | 10.5s（绑定挂载） |

**两侧结果逐项相同**，唯一差异是进程时区与 DB 读取耗时（绑定挂载 IO）。这同时证明**审计本身的环境无关性**：`expected_cst` 不再依赖进程时区。

## 5. 日期型子集 modulo 辅助验证（含一处口径更正）

独立算术校验（不经过本项目任何转换函数）：对 `len(raw) == 10` 的子集断言 `(stored + 28800) % 86400 == 0`。

| 项 | 结果 |
| --- | --- |
| `subset_size` | **101,811** |
| `cst_midnight_ok` | **101,811** |
| `utc_midnight_signature`（`stored % 86400 == 0`） | **0** |
| `modulo_other` | **0** |

**口径更正**：批准时写的"4,623 日期型子集"是**文档级**计数（4,623 个 `documents` 的 `publish_time` 只有日期）；本次扫描按**chunk 级**统计，子集为 **101,811 个 chunk**（占全部 chunk 的 **92.9%**）。原因是这些文档（年报/半年报等）单篇被切成约 22 个 chunk。这不是新问题，但说明**边界缺陷的爆炸半径远大于文档级估计**：在边界日，92.9% 的 chunk 都落在会被旧 UTC 窗口漏掉的那一类（CST 00:00），此前文档级的 39.3% 是低估。

## 6. BEFORE 边界探测（宿主 vs 容器）

主判据 = 生产路径 `ChunkVectorStore._build_filter` 算出的边界 + Qdrant exact count。

| 环境 | gte（生产代码算出） | 渲染 | `window_hits`(exact) |
| --- | --- | --- | --- |
| 宿主（+08:00） | `1777392000` | cst `2026-04-29T00:00:00+08:00` | **51,723** |
| 容器（UTC） | `1777420800` | cst `2026-04-29T08:00:00+08:00` | **44,601** |

差 **7,122** chunk，与事前预测（该日 CST 08:00 前 231 docs / 7,122 chunks）逐位吻合。过滤器条件均为纯 `publish_ts`。

**重复验证**：两侧各复跑一次（`logs/step5_window_*_before_repeat.txt`，用修复后的扫描器 `42a88555`），结果与首次（扫描器 `40972089`）**完全相同** → 该数值不受扫描器终止逻辑修复影响，可作 5-6 复测的对照基线。

## 7. 更正：所谓"scroll 停顿"是扫描器自身的 bug，不是 Docker

断点记录（`8a063bf`）§3 把 600s 未完成归因为"存储/引擎退化，待定位"。**该判断错误，现更正**：

- 直接原因：`_iter_points` 仅以"页为空"作为终止条件，而 Qdrant 的**最后一页仍有数据但返回 `offset=None`**，传 `offset=None` 的 scroll 会**从头重来** → 遍历永不终止。证据：修复前该循环在 109,554 点的集合上跑到 **1,280,000 点**仍未停（约 34,000 点/秒）。此前用于排查的三个一次性探针之所以正常结束，只是因为它们各自都恰好写了 `if off is None: break`。
- 因此"单页 0.01s、走到耗尽却 >600s"这个反直觉现象得到解释：**不是慢，是不终止**。Qdrant 与存储一切正常（`limit=512` 全量 214 页 2.05s，`limit=4096` 1.30s）。
- **Docker 自动更新仍是一条真实的环境风险**：它确实掐断过一次运行（§2），所以本次保留了重试与进度输出；但它不是"停顿"的根因。
- 修复（`dbba569`）：最后一页先 yield 再在 `offset is None` 时终止；并加护栏——遍历点数一旦超过 store 上报的点数，立即判 **RUN INVALID（exit 3）**，避免同类 bug 再次以"有结论"的形式混过去。

## 8. 决策（按预先写死的规则）

```
mismatch == 0 且 missing == 0  → 不运行 backfill_publish_ts.py；只记录
```

**决定：不回填。** 本文件即为该决定的依据；未对 Qdrant、SQLite、索引或向量做任何写入。

## 9. 复现命令

```
$env:QDRANT_URL="http://localhost:6333"     # 必须显式指定，否则会落到 data/qdrant 本地旧副本
.\.venv\Scripts\python.exe -u scripts\scan_publish_ts.py --label "host/pre-fix-fullscan"
.\.venv\Scripts\python.exe -u scripts\scan_publish_ts.py --window 2026-04-29

docker compose -f docker/docker-compose.yml run --rm --no-deps \
  -v "${PWD}\app:/app/app" -v "${PWD}\scripts:/app/scripts" \
  api python scripts/scan_publish_ts.py --label "container/pre-fix-fullscan"
```

**注意**：不设 `QDRANT_URL` 时 `ChunkVectorStore` 会静默切到本地模式，而 `data/qdrant` 那份旧副本**恰好也有 109,554 点**（`ChunkVectorStore()` 会打印 local-mode 告警）。不算集总数就无法分辨——这会给出一个看似正常的假结论。所有扫描命令必须显式带 `QDRANT_URL`，并核对输出里的 `store` 行。
