# 待办清单（评审驱动的收尾项）

> 用途：把评审过程中确认但尚未执行的收尾动作集中记录，避免遗漏。
> 每项完成后请在此标注结果与对应 commit。

## Step 3（评测口径修正后必做）

- [ ] **重新生成 README 的 retrieval benchmark table**
      —— 用统一预算（`candidate-k=20 / fusion-k=20 / final-k=20`）重跑四组实验，
      替换 README「检索评测」一节的两张表；同时在脚注写明：
      旧表为不等预算口径，数字不可与新表直接比较（详见 `docs/evaluation_before_fix.md`）。
- [ ] 产出 `docs/evaluation_after_fix.md`（新口径数字 + 与旧口径的对照 + 参数记录）
- [ ] 更新 README 中"结论"段的表述（等预算下 BM25-only 与 RRF k=60 的相对次序）

## Step 5 / Step 6（修复计划的后续步骤）

- [ ] `bm25_manifest.json` + 加载时一致性校验（`IndexOutOfSync`）
- [ ] UTC+8 时区语义统一 + 重跑 `backfill_publish_ts.py`
- [ ] `rag_tools._to_evidence` 必填字段补 `content` 校验；BM25 命中但回表无正文时显式丢弃并计数

## 记录规则

- 本文件的勾选项**只在对应 commit 落地后**才可标记完成；
- 若某项被评审否决或改为其他方案，请保留条目并注明原因，不要直接删除。
