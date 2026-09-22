# 时间语义（市场日 / UTC+8）

> 状态：Step 5 已完成并通过验收（5-1 ~ 5-6）｜基线 commit：`4453b3f`
> 适用范围：检索日期窗口、`publish_ts` 生成、cninfo 公告日期解析、抓取窗口、审计时间戳
> 相关证据：`reports/step5_publish_ts_scan_baseline.md`、`reports/2026-09-13_Step5审计结果.md`、`reports/2026-09-13_Step5方案_时区语义.md`（历史方案，未修改）

本文只讲本项目的市场日语义与它带来的具体数值契约，不是 Python timezone 教程。

---

> 注：文中提及的 `reports/` 材料是本机内部记录，不随本仓库分发。

## 1. 目的与适用范围

**市场日 = 北京时间自然日（UTC+8）。** A 股的披露日、公告日期、投资者理解的"最近 N 天"都按北京时间划分，因此系统里凡是"某个日期字符串 ↔ 某个瞬时"的转换，都必须以北京时间解释，而不取决于跑代码的那台机器/容器是什么时区。

三条不可动摇的表述：

1. 语义时区为北京时间（UTC+8）。实现上用**固定偏移** `timezone(timedelta(hours=8))`，与 `Asia/Shanghai` 等价（中国大陆自 1991 年起无夏令时）。不用 `ZoneInfo("Asia/Shanghai")` 是刻意的：它要求系统或 `tzdata` 包提供 IANA 数据库，Windows 上没有该数据会直接抛 `ZoneInfoNotFoundError`；固定偏移零依赖、跨平台行为一致。
2. **epoch 是具体瞬时**，不是"把日期字符串直接当成 UTC 午夜"。`"2026-04-29"` 表示北京时间 4 月 29 日 00:00，其 epoch 是 `1777392000`（= UTC 4 月 28 日 16:00），不是 `1777420800`。
3. 本文覆盖的是**语义时间（A 类）与审计时间（B 类）**的分工与边界，以及历史上因此产生的一次真实缺陷（§3）。

Docker 侧**故意不设置 `TZ`**：容器以 UTC 运行，这样任何"依赖进程本地时区"的写法都会立刻暴露，而不是被环境变量掩盖（见 §6）。

---

## 2. 核心契约

单一实现：`app/timeutil.py`。全仓唯一的 date→epoch 转换入口；测试 `test_epoch_construction_is_centralized` 会断言 `app/` 内除该文件外不存在 `.timestamp()` 调用。

| 名称 | 语义 |
| --- | --- |
| `MARKET_TZ` | `timezone(timedelta(hours=8))`，市场时区固定偏移 |
| `market_today(now=None)` | 给定瞬时的**北京日历日**；`now` 是测试注入点，naive 值按北京时间读 |
| `to_market_epoch(value, *, end_of_day=False)` | 日期/时间字符串 → **北京语义**的 epoch 秒；aware 输入按绝对瞬时尊重原样；不可解析时抛 `ValueError`（响亮失败） |
| `now_stamp()` | B 类审计挂钟字符串，格式 `%Y-%m-%d %H:%M:%S` |

### 2.1 固定常数（R3，测试逐字固化）

```text
to_market_epoch("2026-08-15")                  == 1786723200
to_market_epoch("2026-08-15", end_of_day=True) == 1786809599
to_market_epoch("2026-08-15 00:30:00")         == 1786725000
to_market_epoch("2026-08-15T00:00:00+00:00")   == 1786752000   # aware 不被二次平移
```

### 2.2 市场日边界（本次验收的核心数值）

```text
2026-04-29 00:00:00 +08:00  →  1777392000     # 当日下界（= 2026-04-28T16:00:00Z）
2026-04-29 23:59:59 +08:00  →  1777478399     # 当日上界（end_of_day=True）
```

### 2.3 规则细节

- **aware 不二次平移**：带 `tzinfo` 的输入按其绝对瞬时转换。同一瞬时写成 `+08:00` / `+00:00` / `-04:00` 得到同一 epoch；UTC 午夜与北京午夜相差**恰好 28800 秒**，不会被再叠加一次 +8。
- **`end_of_day` 只对纯日期生效**：`to_market_epoch("2026-04-29", end_of_day=True)` 覆盖到当日 23:59:59（跨度 `86399` 秒）；若输入本身带时间（`"2026-04-29 10:00:00"`），该参数**无效果**——这是已固化的契约，改动前请先看 `tests/unit/test_timeutil.py::test_end_of_day_does_not_extend_a_value_that_already_has_a_time`。
- **naive 值按北京时间读**：`to_market_epoch(datetime(2026, 4, 29))` 与 `"2026-04-29"` 同为 `1777392000`。
- **A/B 两类时间的分工**（B 类绝不参与任何判断）：
  - **A 类（语义时间）**：`days` 窗口、`date_from`/`date_to` 边界、`publish_ts` 派生、cninfo 毫秒 → 日期、抓取/增量窗口 → 一律走 `market_today()` / `to_market_epoch()`。
  - **B 类（审计/展示时间）**：`documents`/`chunks`.`created_at`、`stock_daily.created_at`、`stock_basic.updated_at`、BM25 manifest `built_at`、`logs/system.log` 行首 → 一律走 `now_stamp()`，是**北京时间挂钟、不带偏移**，不参与检索过滤、排序或数据归属判断（全仓无 `ORDER BY created_at`，也不经 API/Agent 返回）。
- **`market_today()` 是自然日，不是交易日**：不建模节假日。增量更新的交易日判定来自数据库里的 `trade_date`。

---

## 3. BEFORE / AFTER 对照（真实数值，不是"修了个时区问题"）

### 3.1 BEFORE —— pre-5-3 代码在 UTC 容器中

`app/rag/vector_store.py` 当时用裸 `naive_datetime.timestamp()`，即**进程本地时区**。容器是 UTC，于是"北京 4 月 29 日"被算成了 UTC 4 月 29 日 00:00：

```text
gte = 1777420800
       = 2026-04-29 00:00:00 UTC
       = 2026-04-29 08:00:00 +08:00      ← 比市场日下界晚 8 小时

lte = 1777507199
       = 2026-04-29 23:59:59 UTC
       = 2026-04-30 07:59:59 +08:00      ← 整个窗口被平移 +8h
```

后果：**北京时间当天 00:00:00 ~ 07:59:59 的数据被错误排除**，同时把次日的凌晨数据错误纳入。在本语料上，`publish_time` 只有日期（`YYYY-MM-DD`）的文档其有效时刻正是北京 00:00，因此这一类文档在边界日会被整体漏掉。

### 3.2 AFTER —— 当前实现（同一 UTC 容器）

```text
gte = 1777392000 = 2026-04-29 00:00:00 +08:00
lte = 1777478399 = 2026-04-29 23:59:59 +08:00
```

### 3.3 命中数（lower-bound-only 窗口，即 `days=` 产生的形状）

```text
BEFORE: 44601
AFTER:  51723
difference = +7122
```

`+7122` 与事前**独立统计**完全吻合：边界日北京时间 08:00 前共有 231 个文档 / 7,122 个 chunk。单日窗口（`date_from=date_to=2026-04-29`）为 `1542 → 7231`，其中 `7231` 同时等于 SQLite 中该日文档的 chunk 总数，与 Qdrant 实测一致。

---

## 4. Qdrant / BM25 证据

```text
collection  = financial_chunks
points_count = 109554        (与 SQLite chunks 109,554 / documents 12,366 一致)
BM25 index   = 109554 chunks, chunk_id_hash=333554725291f85e, embedding_version=bge-m3-v1
```

边界日午夜 chunk（`publish_time = "2026-04-29"`，stored `publish_ts = 1777392000`，与 CST 期望值 delta = 0）在真实查询下的表现：

| | dense（Qdrant 真实过滤） | BM25（真实 search 路径） |
| --- | --- | --- |
| BEFORE | **excluded（0）** | included |
| AFTER | **included（1）** | included |

即：旧代码在**运行时**造成了 dense 与 BM25 的日期窗口分歧（BM25 走 `publish_time[:10]` 字符串比较，天然是日历日语义、与时区无关），修复后两条路径表达同一语义。该一致性由 `tests/unit/test_timeutil.py::test_dense_and_bm25_agree_on_the_same_date_boundary` 固化：同一 `date_from`/`date_to` 下 dense 的 epoch 窗口与 BM25 的字符串窗口必须保留完全相同的文档集合。

旧错误边界不得回归：`gte == 1777420800` 已被断言为 False。

---

## 5. cninfo 毫秒时间戳的日期边界

`app/rag/sources.py::_cninfo_ts_to_date` 用 `datetime.fromtimestamp(ms / 1000, tz=MARKET_TZ)` 取日期。以下时刻均按**北京日期**解释（实测，UTC 容器内运行）：

| 北京时刻 | 毫秒 | 得到日期 |
| --- | --- | --- |
| 2026-08-15 00:00:00 | 1786723200000 | `2026-08-15` |
| **2026-08-15 00:30:00** | 1786725000000 | **`2026-08-15`** |
| 2026-08-15 07:59:59 | 1786751999000 | `2026-08-15` |
| 2026-08-15 08:00:00 | 1786752000000 | `2026-08-15` |
| 2026-08-15 16:00:00 | 1786780800000 | `2026-08-15` |
| 2026-08-15 23:59:59 | 1786809599000 | `2026-08-15` |
| 2026-08-14 23:59:59 | — | `2026-08-14` |

**历史关键案例 `00:30`**：同一个毫秒值，在 UTC 容器 + 旧实现下会被算成**前一天**（`2026-08-14`），修复后正确落在 `2026-08-15`。这是 cninfo 公告常见的发布时间段之外最容易被忽略的一类漂移。

---

## 6. UTC 容器验证（5-6）

验收环境为**真实 UTC 进程**，未使用任何伪造手段：

```text
/etc/localtime            → /usr/share/zoneinfo/Etc/UTC
TZ 环境变量               → 未设置（None）
date == date -u           → Sat Sep 19 07:51:34 UTC 2026
datetime.now().astimezone() → 2026-09-19 07:51:34+00:00  offset=0:00:00
```

明确说明：**5-6 没有通过修改 Docker TZ、faketime、monkeypatch 系统时区或临时改生产代码来"制造" UTC 环境**。容器内被测代码由 bind mount 固定（`app/`、`scripts/`），BEFORE 复现则挂载 `6cab1ad` 的 `app/`，因此两次运行只差被测代码本身。Docker 引擎在会话期间从 29.6.2 自动升级到 29.8.0（一次自动重启曾掐断扫描，已记录）。

---

## 7. 测试矩阵：为什么必须在容器里验一遍

`tests/unit/test_timeutil.py`（35 条）在不同环境 × 代码版本的通过情况：

| 环境 | pre-5-3 代码（`6cab1ad`） | 当前代码（`4453b3f`） |
| --- | --- | --- |
| UTC+8 宿主 | **2 failed / 33 passed**（仅两条源码守卫） | **35 passed** |
| **UTC 容器** | **12 failed / 23 passed** | **35 passed** |

- UTC+8 宿主上，旧的 `naive.timestamp()` 恰好等于 CST，**数值层面无法区分**改前改后——宿主只能暴露结构性问题（如 `retrieval.py` 里的 `date.today()`、`vector_store.py` 里自建的 epoch 转换）。
- UTC 容器才能暴露真实数值错误：cninfo `00:00/00:30/07:59:59`、4 组 dense↔BM25 同界、边界日午夜包含，以及两条守卫，共 12 条失败。
- 因此**UTC 容器验证是本次验收不可替代的一部分**：任何涉及时间语义的改动，除了宿主的全量 `pytest`，都应在容器里跑一遍 `tests/unit/test_timeutil.py`。

全量测试基线：`183 passed`（Step 5 之前）→ **218 passed**（`4453b3f`，新增 35 条）。

---

## 8. 数据安全（5-6 期间）

| 项 | 结论 |
| --- | --- |
| SQLite | **无写入**。`data/stock_system.db` mtime 保持 `2026-09-14 11:01:44`、size 903,839,744 不变；验证脚本以 `?mode=ro` 打开 |
| Qdrant | **无写入**。仅调用 `count` / `scroll` / `get_collection`；未调用 `upsert` / `set_payload` / `delete` / `create_payload_index` |
| collection | **未变化**：`points_count=109554`、`status=green`、`segments_count=6`、`payload_schema = doc_type, publish_ts, stock_code` |
| 回填 | **未执行**（见 §9.1，判定为不需要） |
| Docker TZ | 未修改 |
| 其他 | jieba 首次加载在容器 `/tmp` 写缓存，属容器临时层，不是项目数据 |

---

## 9. 技术债务（真实存在，未处理）

**9.1 存量 `publish_ts` 无需回填，但前提已被记录。**
5-2 全量扫描 109,554 点：`ok=109554`、`mismatch=0`、`missing=0`、`payload↔DB publish_time` 不一致 0、`documents↔chunks` 不一致 0、日期型子集 modulo 交叉校验 `101811/101811` 通过。存量值全部等价于"UTC+8 解释"，因此**决定不回填**（依据 `reports/step5_publish_ts_scan_baseline.md`）。若将来对入库路径有疑问，重跑只读的 `scripts/scan_publish_ts.py` 即可判定，不要先回填。

**9.2 审计时间戳是"北京挂钟、不带偏移"。**
`created_at` / `updated_at` / `built_at` / `system.log` 都是北京时间挂钟字符串，且**不会追溯修正**。跨环境（宿主 vs 容器）做时序比较时不可靠；它们也不参与任何业务判断。

**9.3 历史行可能混有旧的 UTC 容器写入。**
本修复之前，若有写入发生在 UTC 容器中，其审计时间戳会比北京时间早 8 小时。这类混合无法从数据本身区分，已接受为历史事实。

**9.4 海外市场未支持。**
固定 +8 对海外数据是错的。真要做多市场，需要按市场引入时区字段，属明确未做项。

**9.5 两条检索路径必须持续保持同一日历日语义。**
dense 用 epoch 区间、BM25 用 `publish_time[:10]` 字符串比较，两者现已一致并被测试守护。改动任一侧的日期过滤逻辑，必须同时跑 `test_dense_and_bm25_agree_on_the_same_date_boundary`，否则会重新出现"同一个 `days` 两个边界"。

**9.6 `days` 窗口的边界行为是本次的**变更**（修复目标，非副作用）。**
边界日北京时间 00:00~07:59:59 的文档由"被排除"变为"被包含"。任何引用历史 `days` 检索结论的地方都需要按新语义回看。

**9.7 本地模式的静默陷阱。**
未设置 `QDRANT_URL` 时 `ChunkVectorStore` 会回落到本地模式 `data/qdrant`，而那一份旧副本**同样含 109,554 点**（点数量相同，无法凭数量分辨）。所有脚本/复现命令必须显式指定 `QDRANT_URL`，并核对输出中的 `store=...` 行，否则可能对着错误的库得出"看起来正常"的结论。

**9.8 `end_of_day=True` 对带时间的输入无效**（契约已固化为测试，但语义上仍是容易踩的点）。

---

## 10. 如何复核

```powershell
# 1) 宿主机：契约与全量回归
.\.venv\Scripts\python.exe -m pytest -q                       # 期望 218 passed

# 2) 只读一致性扫描（必须显式指定 store，否则会打到本地旧副本）
$env:QDRANT_URL="http://localhost:6333"
.\.venv\Scripts\python.exe scripts\scan_publish_ts.py --label "manual/verify"
#    期望：ok=109554 mismatch=0 missing=0，且 store 行是 url:http://localhost:6333

# 3) 边界窗口探测（生产路径 _build_filter + Qdrant exact count）
.\.venv\Scripts\python.exe scripts\scan_publish_ts.py --window 2026-04-29
#    期望：gte=1777392000；仅下界时 hits=51723；单日窗时 lte=1777478399、hits=7231

# 4) 真实 UTC 容器（Docker Desktop 运行中即可，不需要 api 服务）
docker compose -f docker/docker-compose.yml up -d qdrant
docker compose -f docker/docker-compose.yml run --rm --no-deps `
  -v "${PWD}\app:/app/app" -v "${PWD}\scripts:/app/scripts" -v "${PWD}\tests:/app/tests" `
  api python -m pytest tests/unit/test_timeutil.py -q         # 期望 35 passed
```

### 证据文件（`logs/` 为本地证据，不进 Git）

| 文件 | 内容 |
| --- | --- |
| `logs/step5_window_container_before.txt` | 5-2 历史 BEFORE（容器，`gte=1777420800 / hits=44601`） |
| `logs/step5_utc_before_reproduction.txt` | 5-6 用 `6cab1ad` 的 `app/` 在同一容器里复现 BEFORE（含 `target inside the day window? 0`） |
| `logs/step5_utc_after.txt` | 5-6 AFTER 全量输出（环境证据、market_today、epoch、filter JSON、Qdrant、BM25、写入检查） |
| `logs/step5_utc_pytest.txt` | 容器内当前代码跑 5-5 测试（67 passed） |
| `logs/step5_utc_pytest_before_code.txt` | 容器内 pre-5-3 代码跑同一测试（12 failed / 23 passed） |
| `logs/step5_publish_ts_scan_raw.txt` | 5-2 全量扫描原始输出 |
| `logs/step5_publish_ts_scan_after_a_class.txt` | 5-3 之后复扫（证明存量值未被改动） |

5-6 的运行时验证脚本位于**仓库外**的临时目录（与仓库同级的 `step5-verify/verify_utc.py`，不随仓库分发；只读，可反复执行）。它没有进仓库，是为了避免把一次性验证器变成需要维护的生产代码；如需长期保留，应作为 `scripts/` 下的正式审计脚本重新评审后再加入。
