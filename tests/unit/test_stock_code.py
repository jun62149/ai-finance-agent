import pytest

from app.quant.pool import infer_market, normalize_stock_code, stock_board
from app.errors import InvalidStockCode
from app.services.stock_service import validate_code


def test_normalize_stock_code_pads_short_digits():
    assert normalize_stock_code("1") == "000001"
    assert normalize_stock_code(" 600519 ") == "600519"


def test_infer_market():
    assert infer_market("600519") == "sh"
    assert infer_market("000001") == "sz"
    assert infer_market("300750") == "sz"
    assert infer_market("430047") == "bj"


def test_stock_board():
    assert stock_board("600519") == "main"
    assert stock_board("000001") == "main"
    assert stock_board("300750") == "growth"
    assert stock_board("688111") == "star"
    assert stock_board("430047") == "bj"


@pytest.mark.parametrize("bad", ["abc", "12345", "6005191", "60 0519", ""])
def test_validate_code_rejects_invalid(bad):
    with pytest.raises(InvalidStockCode):
        validate_code(bad)


def test_validate_code_accepts_six_digits():
    assert validate_code("600519") == "600519"
