"""数据读取模块：从 CSV 文件加载订单、用户、商品数据"""

import pandas as pd
import os


def load_orders(path: str) -> pd.DataFrame:
    """加载订单 CSV 文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"订单数据文件不存在: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def load_users(path: str) -> pd.DataFrame:
    """加载用户 CSV 文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"用户数据文件不存在: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def load_products(path: str) -> pd.DataFrame:
    """加载商品 CSV 文件"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"商品数据文件不存在: {path}")
    return pd.read_csv(path, encoding="utf-8-sig")


def load_all_data(data_dir: str) -> dict:
    """加载全部数据文件，返回 dict[str, pd.DataFrame]"""
    return {
        "orders": load_orders(os.path.join(data_dir, "orders.csv")),
        "users": load_users(os.path.join(data_dir, "users.csv")),
        "products": load_products(os.path.join(data_dir, "products.csv")),
    }
