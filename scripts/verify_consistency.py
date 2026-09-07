"""Pre/post migration consistency check between legacy B engine and the new repo.

Pre  = legacy B predict path on a copy of the A database (runner subprocess).
Post = this repo's services on the repo database.

Usage:
    python scripts/verify_consistency.py --legacy-root C:\\path\\to\\legacy \\
        --legacy-db-src C:\\path\\to\\stock_system.db [--codes 600519,000858]
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

PREFERRED_NAMES = ["贵州茅台", "五粮液", "中国平安", "招商银行", "宁德时代", "比亚迪", "美的集团", "紫金矿业"]


def pick_codes(db_path: Path, limit: int = 5) -> list[dict]:
    from app.db import read_sql
    from app.config import Config

    original = Config.DB_PATH
    Config.DB_PATH = db_path
    try:
        df = read_sql(
            """
            SELECT b.stock_code, b.stock_name, COUNT(d.id) AS n
            FROM stock_daily d
            JOIN stock_basic b ON d.stock_code = b.stock_code
            GROUP BY b.stock_code, b.stock_name
            ORDER BY n DESC
            """
        )
    finally:
        Config.DB_PATH = original

    picked: list[dict] = []
    for name in PREFERRED_NAMES:
        if len(picked) >= limit:
            break
        hit = df[df["stock_name"].astype(str).str.contains(name, na=False)]
        if not hit.empty:
            for r in hit.head(limit - len(picked)).itertuples():
                item = {"code": str(r.stock_code).zfill(6), "name": str(r.stock_name)}
                if item not in picked:
                    picked.append(item)
    for r in df.itertuples():
        if len(picked) >= limit:
            break
        item = {"code": str(r.stock_code).zfill(6), "name": str(r.stock_name)}
        if item not in picked:
            picked.append(item)
    return picked[:limit]


def run_legacy(legacy_root: Path, legacy_db: Path, out_json: Path) -> None:
    python = sys.executable
    cmd = [python, str(ROOT / "scripts" / "legacy_baseline_runner.py"), str(legacy_root), str(legacy_db), str(out_json)]
    print("running legacy baseline subprocess ...")
    subprocess.run(cmd, check=True, capture_output=True, text=True, encoding="utf-8", errors="ignore")


def post_snapshot(codes: list[dict], pre_by_code: dict | None = None) -> list[dict]:
    from app.services import prediction_service, stock_service

    records = []
    for item in codes:
        try:
            as_of = None
            if pre_by_code and item["code"] in pre_by_code:
                as_of = pre_by_code[item["code"]].get("trade_date")
            factors = stock_service.factor_row_to_dict(
                stock_service.get_factor_row(item["code"], as_of=as_of).iloc[0]
            )
            pred = prediction_service.predict_stock(item["code"], as_of=as_of)
            records.append(
                {
                    "code": pred.code,
                    "name": pred.name,
                    "trade_date": pred.as_of,
                    "factors": {k: factors[k] for k in _factor_keys()},
                    "pred_return_5d": pred.pred_return_5d,
                    "strong_prob": pred.strong_prob,
                    "risk_score": pred.risk_score,
                    "score": pred.score,
                }
            )
        except Exception as exc:
            records.append({"code": item["code"], "name": item["name"], "error": str(exc)})
    return records


def _factor_keys() -> list[str]:
    return [
        "ret_5d", "ret_10d", "ret_20d", "ma5_bias", "ma20_bias",
        "volatility_10d", "volatility_20d", "volume_ratio_5d",
        "amount_mean_20d", "turnover_mean_20d", "drawdown_20d",
    ]


def compare(pre: list[dict], post: list[dict], tolerance: float = 1e-8) -> dict:
    pre_by_code = {r["code"]: r for r in pre if "error" not in r}
    post_by_code = {r["code"]: r for r in post if "error" not in r}
    rows = []
    max_diff = 0.0
    total = len(post_by_code)
    matched = 0
    for code in sorted(post_by_code):
        p = pre_by_code.get(code)
        q = post_by_code.get(code)
        if p is None:
            rows.append({"code": code, "name": q.get("name", ""), "status": "missing-in-pre"})
            continue
        diffs: dict[str, float] = {}
        for field in _factor_keys() + ["pred_return_5d", "strong_prob", "risk_score", "score"]:
            a = p.get(field)
            b = q.get(field)
            diff = _diff(a, b)
            if diff is not None:
                diffs[field] = diff
                max_diff = max(max_diff, diff)
        row_max = max(diffs.values(), default=0.0)
        status = "match" if row_max <= tolerance else "diff"
        matched += int(status == "match")
        rows.append(
            {
                "code": code,
                "name": p.get("name", ""),
                "trade_date_pre": p.get("trade_date"),
                "trade_date_post": q.get("trade_date"),
                "max_diff": row_max,
                "status": status,
                "pre": p,
                "post": q,
            }
        )
    return {"rows": rows, "max_diff": max_diff, "total": total, "matched": matched}


def _diff(a, b):
    if a is None or b is None:
        return None if a is b else float("inf")
    try:
        return abs(float(a) - float(b))
    except (TypeError, ValueError):
        return float("inf")


def print_report(result: dict) -> None:
    print(f"\nconsistency: matched={result['matched']}/{result['total']}, max_abs_diff={result['max_diff']:.2e}")
    for row in result["rows"]:
        print(
            f"{row['code']} {row.get('name',''):<8s} {row['status']:<18s} "
            f"date_pre={row.get('trade_date_pre')} date_post={row.get('trade_date_post')} "
            f"max_diff={row.get('max_diff', float('nan')):.2e}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--legacy-db-src", required=True)
    parser.add_argument("--codes", default="")
    args = parser.parse_args()

    legacy_root = Path(args.legacy_root)
    legacy_db_src = Path(args.legacy_db_src)
    temp_legacy_db = ROOT / "data" / "_legacy_compare.db"
    out_json = ROOT / "data" / "_legacy_baseline.json"
    shutil.copyfile(legacy_db_src, temp_legacy_db)

    if args.codes:
        codes = [{"code": c.strip().zfill(6), "name": ""} for c in args.codes.split(",") if c.strip()]
    else:
        codes = pick_codes(temp_legacy_db)
    print("codes:", codes)

    run_legacy(legacy_root, temp_legacy_db, out_json)
    pre = json.loads(out_json.read_text(encoding="utf-8"))["records"]
    pre_by_code = {r["code"]: r for r in pre if "error" not in r}
    post = post_snapshot(codes, pre_by_code)
    result = compare(pre, post)
    print_report(result)


if __name__ == "__main__":
    main()
