"""数据清洗模块：去重、缺失值处理、类型转换、异常值过滤、宽表构建"""

import pandas as pd
from src.utils import parse_datetime


def clean_orders(orders: pd.DataFrame) -> pd.DataFrame:
    """清洗订单数据"""
    df = orders.copy()

    # 去重
    df = df.drop_duplicates(subset=["order_id"])

    # 时间字段转换
    df["order_time"] = parse_datetime(df["order_time"])
    df["pay_time"] = parse_datetime(df["pay_time"])

    # 删除时间解析失败的记录
    df = df.dropna(subset=["order_time"])

    # 金额和数量转数值
    df["order_amount"] = pd.to_numeric(df["order_amount"], errors="coerce")
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")

    # 过滤异常金额（<=0）
    df = df[df["order_amount"] > 0]

    # 过滤异常数量
    df = df[df["quantity"] > 0]

    # 只保留合法订单状态
    valid_statuses = ["paid", "refunded", "canceled"]
    df = df[df["order_status"].isin(valid_statuses)]

    return df


def clean_users(users: pd.DataFrame) -> pd.DataFrame:
    """清洗用户数据"""
    df = users.copy()
    df = df.drop_duplicates(subset=["user_id"])
    df["register_time"] = parse_datetime(df["register_time"])
    df = df.dropna(subset=["register_time"])
    return df


def clean_products(products: pd.DataFrame) -> pd.DataFrame:
    """清洗商品数据"""
    df = products.copy()
    df = df.drop_duplicates(subset=["product_id"])
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df[df["price"] > 0]
    return df


def build_analysis_table(
    orders: pd.DataFrame,
    users: pd.DataFrame,
    products: pd.DataFrame,
) -> pd.DataFrame:
    """将订单表关联用户表和商品表，构建分析宽表"""
    df = orders.merge(users, on="user_id", how="left")
    df = df.merge(products, on="product_id", how="left")
    return df
