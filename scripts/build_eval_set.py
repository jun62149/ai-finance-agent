"""Build the retrieval evaluation set (Phase 9 groundwork).

Ground truth is resolved from SQLite with explicit, auditable specs — each spec
names the stock / document type / date window / title keywords that identify the
documents which actually answer the query. The resolved labels are frozen into
`data/eval/retrieval_eval.json` so evaluation runs never re-derive them.

Query categories follow the design doc §21 coverage: stock code, company name,
date constraints, announcement types / legal terms, and semantic event questions.

Usage:
    python scripts/build_eval_set.py                 # resolve + write
    python scripts/build_eval_set.py --out data/eval/retrieval_eval.json
    python scripts/build_eval_set.py --show          # print specs without writing
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.db import connect, init_db

# Each spec resolves to a set of documents. `title_any` / `title_all` are
# keyword filters on the document title; `note` explains what answers it.
EVAL_SPECS: list[dict] = [
    # --- A. stock code (exact token; BM25 should win) ---------------------- #
    {"id": "A01", "category": "code", "query": "600519 半年度报告",
     "stock_code": "600519", "doc_type": "announcement", "title_any": ["半年度报告"]},
    {"id": "A02", "category": "code", "query": "000725 回购实施结果",
     "stock_code": "000725", "doc_type": "announcement", "title_all": ["回购", "实施结果"]},
    {"id": "A03", "category": "code", "query": "002466 限制性股票 回购注销 通知债权人",
     "stock_code": "002466", "doc_type": "announcement", "title_all": ["限制性股票", "通知债权人"]},
    {"id": "A04", "category": "code", "query": "000823 可转换公司债券 转股情况",
     "stock_code": "000823", "doc_type": "announcement", "title_any": ["转股"]},
    {"id": "A05", "category": "code", "query": "000776 永续次级债券 发行结果",
     "stock_code": "000776", "doc_type": "announcement", "title_all": ["永续次级债券", "发行结果"]},
    {"id": "A06", "category": "code", "query": "000688 重大资产购买 独立财务顾问报告",
     "stock_code": "000688", "doc_type": "announcement", "title_all": ["重大资产购买", "独立财务顾问"]},

    # --- B. company name / natural phrasing -------------------------------- #
    {"id": "B01", "category": "name", "query": "京东方A 股份回购的进展与实施结果",
     "stock_code": "000725", "doc_type": "announcement", "title_any": ["回购"]},
    {"id": "B02", "category": "name", "query": "贵州茅台上半年的营业收入和净利润表现",
     "stock_code": "600519", "title_any": ["半年度报告"]},
    {"id": "B03", "category": "name", "query": "潍柴动力中期分红派息后调整回购股份价格上限",
     "stock_code": "000338", "doc_type": "announcement", "title_any": ["价格上限"]},
    {"id": "B04", "category": "name", "query": "广发证券债券发行结果公告",
     "stock_code": "000776", "doc_type": "announcement", "title_any": ["发行结果"]},
    {"id": "B05", "category": "name", "query": "安泰科技转让产业基金份额",
     "stock_code": "000969", "doc_type": "announcement", "title_any": ["产业基金"]},
    {"id": "B06", "category": "name", "query": "超声电子对控股子公司增资",
     "stock_code": "000823", "doc_type": "announcement", "title_any": ["增资"]},
    {"id": "B07", "category": "name", "query": "深桑达A 诉讼 由盈转亏",
     "stock_code": "000032", "doc_type": "news", "title_any": ["诉讼"]},
    {"id": "B08", "category": "name", "query": "国城矿业 重大资产购买暨关联交易",
     "stock_code": "000688", "doc_type": "announcement", "title_any": ["重大资产购买"]},

    # --- C. date-constrained ------------------------------------------------- #
    {"id": "C01", "category": "date", "query": "600519 在 2026 年 8 月发布的公告",
     "stock_code": "600519", "doc_type": "announcement",
     "date_from": "2026-08-01", "date_to": "2026-08-31"},
    {"id": "C02", "category": "date", "query": "京东方A 2026 年 1 月的投资者关系活动记录",
     "stock_code": "000725", "doc_type": "announcement", "title_any": ["投资者关系活动"],
     "date_from": "2026-01-01", "date_to": "2026-01-31"},
    {"id": "C03", "category": "date", "query": "广发证券 2026 年 3 月披露的年度报告相关公告",
     "stock_code": "000776", "doc_type": "announcement", "title_any": ["年度报告"],
     "date_from": "2026-03-01", "date_to": "2026-03-31"},
    {"id": "C04", "category": "date", "query": "美的集团 2026 年 3 月 31 日的公告",
     "stock_code": "000333", "doc_type": "announcement",
     "date_from": "2026-03-31", "date_to": "2026-03-31"},
    {"id": "C05", "category": "date", "query": "深桑达A 2026 年半年度报告摘要",
     "stock_code": "000032", "doc_type": "announcement", "title_any": ["半年度报告摘要"]},

    # --- D. announcement types / legal terms -------------------------------- #
    {"id": "D01", "category": "doctype", "query": "广发证券 2025 年度独立董事述职报告",
     "stock_code": "000776", "doc_type": "announcement", "title_any": ["2025年度独立董事述职"]},
    {"id": "D02", "category": "doctype", "query": "2025 年度股东会的法律意见书",
     "doc_type": "announcement", "title_all": ["2025年度股东会", "法律意见书"]},
    {"id": "D03", "category": "doctype", "query": "受托管理事务报告",
     "doc_type": "announcement", "title_any": ["受托管理"]},
    {"id": "D04", "category": "doctype", "query": "重大资产重组 保密措施及保密制度",
     "doc_type": "announcement", "title_any": ["保密措施"]},
    {"id": "D05", "category": "doctype", "query": "限制性股票激励计划部分股票回购注销暨通知债权人",
     "doc_type": "announcement", "title_any": ["通知债权人"]},
    {"id": "D06", "category": "doctype", "query": "大宗原材料套期保值业务可行性分析报告",
     "doc_type": "announcement", "title_any": ["套期保值"]},

    # --- E. semantic / event questions -------------------------------------- #
    {"id": "E01", "category": "semantic", "query": "哪家公司获得了银行贷款承诺用于支持股份回购",
     "doc_type": "announcement", "title_any": ["贷款承诺"]},
    {"id": "E02", "category": "semantic", "query": "哪些公司披露了股份回购方案实施完毕的结果",
     "doc_type": "announcement", "title_all": ["回购", "实施完毕"]},
    {"id": "E03", "category": "semantic", "query": "控股子公司项目延期",
     "doc_type": "announcement", "title_any": ["项目延期"]},
    {"id": "E04", "category": "semantic", "query": "公司聘任会计师事务所的相关议案",
     "doc_type": "announcement", "title_all": ["会计师事务所"], "title_any": ["聘任", "续聘"]},
    {"id": "E05", "category": "semantic", "query": "通信板块资金流入情况",
     "stock_code": "000063", "doc_type": "news", "title_any": ["资金流入"]},
    {"id": "E06", "category": "semantic", "query": "上市险企净利润增速出现分化",
     "stock_code": "601318", "doc_type": "news", "title_any": ["净利增速"]},
    {"id": "E07", "category": "semantic", "query": "公募基金调研转向医药生物领域",
     "doc_type": "news", "title_all": ["调研", "医药生物"]},
    {"id": "E08", "category": "semantic", "query": "私募基金产品业绩大幅回撤",
     "doc_type": "news", "title_any": ["回撤"]},
    {"id": "E09", "category": "semantic", "query": "股权激励相关的回购注销公告",
     "doc_type": "announcement", "title_any": ["股权激励"]},
    {"id": "E10", "category": "semantic", "query": "股东减持股份计划公告",
     "doc_type": "announcement", "title_any": ["减持"]},
    {"id": "E11", "category": "semantic", "query": "日常关联交易预计的公告",
     "doc_type": "announcement", "title_any": ["日常关联交易预计"]},
]

MAX_DOCS_PER_QUERY = 40


def _resolve(spec: dict) -> list[dict]:
    where = []
    params: list = []
    if spec.get("stock_code"):
        where.append("stock_code = ?")
        params.append(spec["stock_code"])
    if spec.get("doc_type"):
        where.append("doc_type = ?")
        params.append(spec["doc_type"])
    if spec.get("date_from"):
        where.append("publish_time >= ?")
        params.append(spec["date_from"])
    if spec.get("date_to"):
        where.append("publish_time <= ?")
        params.append(spec["date_to"] + " 23:59:59")
    for keyword in spec.get("title_all", []):
        where.append("title LIKE ?")
        params.append(f"%{keyword}%")
    title_any = spec.get("title_any") or []
    if title_any:
        where.append("(" + " OR ".join(["title LIKE ?"] * len(title_any)) + ")")
        params.extend(f"%{keyword}%" for keyword in title_any)

    sql = "SELECT document_id, stock_code, stock_name, doc_type, title, publish_time FROM documents"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY publish_time DESC"
    with connect() as conn:
        return [dict(row) for row in conn.execute(sql, tuple(params)).fetchall()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the retrieval eval set")
    parser.add_argument("--out", default="data/eval/retrieval_eval.json")
    parser.add_argument("--show", action="store_true", help="print specs, write nothing")
    args = parser.parse_args()

    init_db()
    items: list[dict] = []
    problems: list[str] = []
    for spec in EVAL_SPECS:
        docs = _resolve(spec)
        if not docs:
            problems.append(f"{spec['id']}: no documents matched — spec needs fixing")
        elif len(docs) > MAX_DOCS_PER_QUERY:
            problems.append(
                f"{spec['id']}: {len(docs)} documents matched (> {MAX_DOCS_PER_QUERY}) — tighten it"
            )
        items.append(
            {
                **spec,
                "relevant_document_ids": [doc["document_id"] for doc in docs],
                "relevant_documents": [
                    {
                        "document_id": doc["document_id"],
                        "stock_code": doc["stock_code"],
                        "title": doc["title"],
                        "doc_type": doc["doc_type"],
                        "publish_time": doc["publish_time"],
                    }
                    for doc in docs[:5]
                ],
                "relevant_count": len(docs),
            }
        )

    by_category: dict[str, int] = {}
    for item in items:
        by_category[item["category"]] = by_category.get(item["category"], 0) + 1
    print(f"queries: {len(items)} | categories: {by_category}")
    for item in items:
        print(f"  {item['id']} [{item['category']:<8}] rel={item['relevant_count']:<3} {item['query']}")
        if item["relevant_documents"]:
            print(f"        e.g. {item['relevant_documents'][0]['title'][:60]}")

    if problems:
        print("")
        for problem in problems:
            print(f"[!] {problem}", file=sys.stderr)

    if args.show:
        return
    if problems:
        print("", file=sys.stderr)
        print("fix the specs above before freezing the eval set", file=sys.stderr)
        raise SystemExit(2)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "description": (
            "Retrieval evaluation set. Labels are resolved from SQLite by explicit "
            "specs (stock / doc_type / date window / title keywords) and frozen here."
        ),
        "categories": by_category,
        "queries": items,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("")
    print(f"wrote {out_path} ({len(items)} queries)")


if __name__ == "__main__":
    main()
