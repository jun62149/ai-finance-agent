# Step 5 断点记录（5-2 未完成）

> 断点时间：2026-09-16 21:5x（CST）｜HEAD：`3c2f325`｜工作区干净
> 性质：**断点/交接记录，不是 Step 5 交付报告**。正式 baseline 见 5-2 完成后的 `reports/step5_publish_ts_scan_baseline.md`。

## 1. 已完成

| 步骤 | 结果 | 证据 |
| --- | --- | --- |
| 5-0 环境确认 | Docker 29.6.2；Qdrant 1.19.1 healthy；`points_count=109554` = DB `chunks` 109554（store 正确）；容器默认 UTC（`utcoffset=0`） | `logs/step5_window_*_before.txt` 头部 |
| 5-1 timeutil 纯新增 | `1fdeb7c`，`app/timeutil.py`（83 行），未接入任何调用点；`183 passed`（无回归） | calibration 四行全部 OK |
| 5-2 scanner | `3c2f325`，`scripts/scan_publish_ts.py`：MODE 1 全量扫描 / MODE 2 `--window`；只调用 Qdrant `count`+`scroll`；SQLite `mode=ro` | 见下 |
| 5-2 BEFORE 边界证据 | **已完成**，见第 2 节 | `logs/step5_window_host_before.txt`、`logs/step5_window_container_before.txt` |
| 5-2 全量 baseline | **未完成**（被 scroll 停顿阻塞） | `logs/step5_publish_ts_scan_container_pre.txt` 只有头部 |

**5-3 未开始**：所有 A 类生产调用点（`vector_store._date_to_epoch`、`retrieval.py:84`、`sources.py:269`、三个抓取/更新脚本）**仍全部是未修改的原状**。全程无任何数据写入。

## 2. BEFORE 边界探测结果（主判据，已成立）

| 环境 | 进程时区 | gte（生产代码算出的边界） | window_hits (exact) |
| --- | --- | --- | --- |
| 宿主 | `utcoffset=+08:00` | `1777392000` = cst `2026-04-29T00:00:00+08:00` | **51723** |
| 容器 | `utcoffset=0` | `1777420800` = cst `2026-04-29T08:00:00+08:00` | **44601** |

差 **7,122** chunk，与事前预测（2026-04-29 当日 CST 08:00 前 231 docs / 7,122 chunks）**完全一致**；两侧过滤器条件均为纯 `publish_ts`，挂载文件 sha256 一致（timeutil `876198f7`、vector_store `94917f53`、scanner `40972089`），即同一份代码、仅进程时区不同。→ 缺陷复现成立，且这是 5-6 复测的对照基线。

## 3. 阻塞现象（实测，供明天定位）

单次读取都极快，但**遍历到耗尽**会停住：

| 操作 | 耗时 |
| --- | --- |
| `count(exact=True)` | 0.02s |
| `scroll` 首页 limit=512 / 5120 / 20000 | 0.01s / 0.07s / 0.27s |
| `scroll` 前 4 页 limit=512 | 0.05s 合计 |
| `scroll` 前 3 页 limit=10000 | 0.35s 合计 |
| `scroll` **走到耗尽** limit=512 / 10000 | **>600s 未完成（三次一致：宿主、容器、单独跑）** |

已排除的原因：
- 分页不推进 —— 已证 offset 递增、`first_id` 递增（1275→28148→123837→230218），分页正常；
- `count` 慢 —— 0.02s；
- DB JOIN 计划 —— `EXPLAIN QUERY PLAN` 走 `sqlite_autoindex_documents_1`，且头部已打印 `db_rows_loaded=109554`（DB 侧已完成）；
- 并发竞争 —— 停掉并行任务、单独跑仍停在同一步。

**反向事实（重要）**：2026-09-13 的 backfill 曾在同一 Qdrant、同一 `docker/qdrant_storage` 上 **282s 扫完 109,554 点并写入**（`logs/backfill_ts.txt`）→ 存储本身能扫，所以这是**退化**，不是固有性能问题（候选：segment 优化器状态、Windows 绑定挂载 IO、杀毒扫描、服务端 REST 大响应）。

## 4. 明天从这里开始（按序）

1. 逐页计时打到停滞点：记录卡在第几页、当时 `offset`、以及 `GET /collections/financial_chunks` 的 `segments_count` / `optimizer_status` —— 先定位，不猜。
2. 试 gRPC：`QdrantClient(url=..., prefer_grpc=True, grpc_port=6334)` 重跑全量 scroll（REST 批量读很可能是瓶颈；compose 已映射 6334）。
3. 若 gRPC 仍不可用，改用**分桶 count 校验**替代逐点遍历（仍是 R1/R2/R3 口径，只是判据形态从"逐点"变"逐桶穷尽"）：
   由 SQLite 得到每个原始字符串的 `expected_cst`，按 epoch 分桶；用 `count(Filter(publish_ts ∈ [桶下界, 桶上界]))` 与 SQLite 同桶计数比对。若所有桶逐一相等，则不存在系统性 8h 偏移（`-28800` 的整桶错位会立刻暴露）。**这只作为阻塞时的降级方案，首选仍是逐点比对。**
4. 拿到 baseline 后：写 `reports/step5_publish_ts_scan_baseline.md` → 报告 → 等批准 → 才进 5-3。

## 5. 不变的口径（明天不要走样）

- R1 期望值只由 DB `chunks.publish_time` 原始字符串构造；**严禁**从 stored `publish_ts` 反推。
- R2 payload / documents / chunks 三处 `publish_time` 不一致分别统计，不静默取其一。
- R3 calibration 固定值必须与测量结果同文件输出。
- `mismatch>0` 或 `missing>0` → **只报告、暂停、不自动 backfill**。
- 容器复测：保持默认 UTC 进程时区，**不设 TZ**，**不重建镜像**，用 bind mount 固定代码版本并记录 sha256。
- `docs/` 不默认新增；baseline 等产物默认落 `reports/`。
- 评测集、评测协议、指标实现本步不动。

## 6. 环境快照（明天可直接用）

Docker Desktop **保持运行**（省去约 8 分钟启动）：`ai-finance-qdrant`(healthy)、`ai-finance-api`(healthy，旧镜像)、`open-webui`(healthy)。
证据文件：`logs/step5_window_host_before.txt`、`logs/step5_window_container_before.txt`、`logs/step5_publish_ts_scan_container_pre.txt`（仅头部）。
容器运行模板（已验证可用）：
```
docker compose -f docker/docker-compose.yml run --rm --no-deps \
  -v "${PWD}\app:/app/app" -v "${PWD}\scripts:/app/scripts" \
  api python scripts/scan_publish_ts.py --window 2026-04-29 --label "container/..."
```
