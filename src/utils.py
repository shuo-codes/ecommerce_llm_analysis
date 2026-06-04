"""工具函数：时间解析、格式化、安全除法等"""

import pandas as pd
import numpy as np
from datetime import datetime


def safe_divide(numerator, denominator):
    """安全除法，分母为 0 时返回 0"""
    if denominator == 0 or pd.isna(denominator):
        return 0.0
    return numerator / denominator


def parse_datetime(series: pd.Series) -> pd.Series:
    """将 Series 转为 datetime，解析失败置为 NaT"""
    return pd.to_datetime(series, errors="coerce")


def format_currency(value: float) -> str:
    """将金额格式化为带千分位分隔的字符串"""
    if pd.isna(value):
        return "¥0.00"
    return f"¥{value:,.2f}"


def format_percent(value: float) -> str:
    """将小数格式化为百分比字符串"""
    if pd.isna(value):
        return "0.00%"
    return f"{value * 100:.2f}%"


def extract_year_month(dt_series: pd.Series) -> pd.Series:
    """从 datetime Series 提取 YYYY-MM 格式"""
    return dt_series.dt.strftime("%Y-%m")


def extract_year_week(dt_series: pd.Series) -> pd.Series:
    """从 datetime Series 提取 YYYY-WW 格式"""
    return dt_series.dt.strftime("%Y-%W")


def filter_by_date_range(df: pd.DataFrame, date_col: str, start_date, end_date) -> pd.DataFrame:
    """按日期范围过滤 DataFrame"""
    if start_date is not None:
        df = df[df[date_col] >= pd.Timestamp(start_date)]
    if end_date is not None:
        df = df[df[date_col] <= pd.Timestamp(end_date)]
    return df
