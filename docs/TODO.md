# 待办清单（评审驱动的收尾项）

> 用途：把评审过程中确认但尚未执行的收尾动作集中记录，避免遗漏。
> 每项完成后请在此标注结果与对应 commit。

## Step 3（评测口径修正后必做）

- [ ] **重新生成 README 的 retrieval benchmark table**
      —— 用统一预算（`candidate-k=20 / fusion-k=20 / final-k=20`）重跑四组实验，
      替换 README「检索评测」一节的两张表；同时在脚注写明：
      旧表为不等预算口径，数字不可与新表直接比较（详见 `docs/evaluation_before_fix.md`）。
- [x] 产出 `docs/evaluation_after_fix.md`（新口径数字 + 与旧口径对照 + BM25 窗口 A/B）—— 已完成 `3738d85`
- [ ] 更新 README 中"结论"段的表述（等预算下 BM25-only 与 RRF k=60 的相对次序）

## 评测集覆盖（Step 2 审计发现，**暂不修改**）

- [ ] 增加 1~3 条 **adversarial case**：`泛化 query + 窄 metadata filter`
      （真实失效模式示例：`query='公告'` + `stock_code=000333` + `doc_type=announcement`
      —— 截断窗口下 2 命中、全库下 356 命中）
      目的：让"BM25 候选截断"这类失效在**指标层面**可观测，而不只靠单元测试守护。
      ⚠️ **现在不要改 `data/eval/retrieval_eval.json`**：会改变 benchmark 可比性，
      使已完成的三次对照（before_fix / after_fix / window A/B）失去一致性。
      若要加，应作为**独立版本**（如 `retrieval_eval_v2.json`）并重跑全部对照。

## Step 5 / Step 6（修复计划的后续步骤）

- [ ] `bm25_manifest.json` + 加载时一致性校验（`IndexOutOfSync`）
- [ ] UTC+8 时区语义统一 + 重跑 `backfill_publish_ts.py`
- [ ] `rag_tools._to_evidence` 必填字段补 `content` 校验；BM25 命中但回表无正文时显式丢弃并计数

## 记录规则

- 本文件的勾选项**只在对应 commit 落地后**才可标记完成；
- 若某项被评审否决或改为其他方案，请保留条目并注明原因，不要直接删除。
