"""Stock pool helpers, migrated unchanged from legacy B filter_service.

Determines board, special-treatment flag, liquidity and listing-age eligibility.
"""

from __future__ import annotations

import pandas as pd

from app.config import Config


BOARD_LABELS = {
    "main": "主板",
    "growth": "创业板",
    "star": "科创板",
    "bj": "北交所",
    "other": "其他",
}


def normalize_stock_code(stock_code: str) -> str:
    return str(stock_code).strip().zfill(6)


def infer_market(stock_code: str) -> str:
    code = normalize_stock_code(stock_code)
    if code.startswith("6"):
        return "sh"
    if code.startswith(("0", "3")):
        return "sz"
    if code.startswith(("4", "8")):
        return "bj"
    return "unknown"


def stock_board(stock_code: str) -> str:
    code = str(stock_code).zfill(6)
    if code.startswith(("300", "301")):
        return "growth"
    if code.startswith(("688", "689")):
        return "star"
    if code.startswith(("4", "8")):
        return "bj"
    if code.startswith(("000", "001", "002", "003", "600", "601", "603", "605")):
        return "main"
    return "other"


def stock_board_label(stock_code: str) -> str:
    return BOARD_LABELS.get(stock_board(stock_code), "其他")


def is_special_treatment(stock_name: str = "") -> bool:
    name = stock_name or ""
    upper_name = name.upper()
    return any(keyword.upper() in upper_name for keyword in Config.EXCLUDE_STOCK_NAME_KEYWORDS)


def is_regular_recommendation_stock(stock_code: str, stock_name: str = "") -> bool:
    code = normalize_stock_code(stock_code)
    if stock_board(code) in {"growth", "star"}:
        return False
    return not is_special_treatment(stock_name)


def _liquidity_mask(df: pd.DataFrame) -> pd.Series:
    if "amount_mean_20d" in df.columns:
        amount = pd.to_numeric(df["amount_mean_20d"], errors="coerce")
        return amount.fillna(0) >= Config.MIN_AMOUNT
    if "amount" in df.columns:
        amount = pd.to_numeric(df["amount"], errors="coerce")
        return amount.fillna(0) >= Config.MIN_AMOUNT
    return pd.Series(True, index=df.index)


def _listing_age_mask(df: pd.DataFrame) -> pd.Series:
    if "list_date" not in df.columns or "trade_date" not in df.columns:
        return pd.Series(True, index=df.index)
    list_date = pd.to_datetime(df["list_date"], errors="coerce")
    trade_date = pd.to_datetime(df["trade_date"], errors="coerce")
    age_days = (trade_date - list_date).dt.days
    return age_days.fillna(Config.MIN_LISTING_DAYS) >= Config.MIN_LISTING_DAYS


def add_pool_flags(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    result = df.copy()
    if "stock_name" not in result.columns:
        result["stock_name"] = ""

    result["stock_code"] = result["stock_code"].map(normalize_stock_code)
    result["board"] = result["stock_code"].map(stock_board)
    result["board_label"] = result["stock_code"].map(stock_board_label)
    result["is_name_risk"] = result["stock_name"].fillna("").map(is_special_treatment)
    result["is_liquid"] = _liquidity_mask(result)
    result["is_listed_enough"] = _listing_age_mask(result)
    result["is_model_eligible"] = (
        (~result["is_name_risk"]) & result["is_liquid"] & result["is_listed_enough"]
    )
    result["is_regular_board"] = ~result["board"].isin(["growth", "star"])
    result["is_recommendable_regular"] = result["is_model_eligible"] & result["is_regular_board"]
    result["is_recommendable_with_growth"] = result["is_model_eligible"]
    return result


def apply_model_pool_filter(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    flagged = add_pool_flags(df)
    return flagged[flagged["is_model_eligible"]].reset_index(drop=True)


def apply_recommendation_filter(df: pd.DataFrame, include_growth: bool = False) -> pd.DataFrame:
    if df.empty:
        return df

    filtered = add_pool_flags(df)
    mask_col = "is_recommendable_with_growth" if include_growth else "is_recommendable_regular"
    return filtered[filtered[mask_col]].reset_index(drop=True)
