"""指标计算模块：GMV、订单量、客单价、复购率、退款率等核心经营指标"""

import pandas as pd
import numpy as np
from src.utils import safe_divide


def calculate_overall_metrics(df: pd.DataFrame) -> dict:
    """计算整体经营指标"""
    paid = df[df["order_status"] == "paid"]
    refunded = df[df["order_status"] == "refunded"]

    gmv = paid["order_amount"].sum()
    order_count = paid["order_id"].nunique()
    paying_users = paid["user_id"].nunique()

    avg_order_value = safe_divide(gmv, order_count)
    avg_user_value = safe_divide(gmv, paying_users)

    total_non_canceled = len(paid) + len(refunded)
    refund_count = len(refunded)
    refund_rate = safe_divide(refund_count, total_non_canceled)

    # 复购率：支付订单数 >= 2 的用户
    user_order_counts = paid.groupby("user_id")["order_id"].nunique()
    repurchase_users = (user_order_counts >= 2).sum()
    repurchase_rate = safe_divide(repurchase_users, paying_users)

    return {
        "gmv": round(gmv, 2),
        "order_count": order_count,
        "paying_users": paying_users,
        "avg_order_value": round(avg_order_value, 2),
        "avg_user_value": round(avg_user_value, 2),
        "refund_rate": round(refund_rate, 4),
        "repurchase_rate": round(repurchase_rate, 4),
    }


def calculate_time_trend(df: pd.DataFrame, freq: str = "M") -> pd.DataFrame:
    """按天/周/月统计趋势指标并计算环比变化率

    Args:
        df: 分析宽表
        freq: D=天, W=周, M=月
    """
    paid = df[df["order_status"] == "paid"].copy()
    paid["period"] = paid["order_time"].dt.to_period(freq)

    trend = paid.groupby("period").agg(
        gmv=("order_amount", "sum"),
        order_count=("order_id", "nunique"),
        paying_users=("user_id", "nunique"),
    ).reset_index()

    trend["period"] = trend["period"].astype(str)
    trend["avg_order_value"] = trend.apply(
        lambda r: round(safe_divide(r["gmv"], r["order_count"]), 2), axis=1
    )
    trend["gmv_mom"] = trend["gmv"].pct_change()
    trend["order_count_mom"] = trend["order_count"].pct_change()
    trend["avg_order_value_mom"] = trend["avg_order_value"].pct_change()

    return trend


def calculate_category_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """计算品类维度指标"""
    paid = df[df["order_status"] == "paid"]
    refunded = df[df["order_status"] == "refunded"]

    gmv_by_cat = paid.groupby("category")["order_amount"].sum()
    orders_by_cat = paid.groupby("category")["order_id"].nunique()
    users_by_cat = paid.groupby("category")["user_id"].nunique()

    total_gmv = gmv_by_cat.sum()
    total_non_canceled = paid.groupby("category")["order_id"].nunique() + refunded.groupby("category")["order_id"].nunique()
    refund_by_cat = refunded.groupby("category")["order_id"].nunique()

    result = pd.DataFrame({
        "category": gmv_by_cat.index,
        "gmv": gmv_by_cat.values,
        "order_count": orders_by_cat.values,
        "paying_users": users_by_cat.values,
    })

    result["avg_order_value"] = result.apply(
        lambda r: round(safe_divide(r["gmv"], r["order_count"]), 2), axis=1
    )
    result["gmv_ratio"] = result["gmv"] / total_gmv

    # 退款率
    result = result.merge(
        pd.DataFrame({
            "category": refund_by_cat.index,
            "refund_count": refund_by_cat.values,
        }),
        on="category",
        how="left",
    )
    result["refund_count"] = result["refund_count"].fillna(0)
    result["total_non_canceled"] = result["order_count"] + result["refund_count"]
    result["refund_rate"] = result.apply(
        lambda r: round(safe_divide(r["refund_count"], r["total_non_canceled"]), 4), axis=1
    )

    result = result.sort_values("gmv", ascending=False).reset_index(drop=True)
    return result[["category", "gmv", "order_count", "paying_users", "avg_order_value", "refund_rate", "gmv_ratio"]]


def calculate_channel_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """计算渠道维度指标"""
    paid = df[df["order_status"] == "paid"]
    refunded = df[df["order_status"] == "refunded"]

    gmv_by_ch = paid.groupby("channel")["order_amount"].sum()
    orders_by_ch = paid.groupby("channel")["order_id"].nunique()
    users_by_ch = paid.groupby("channel")["user_id"].nunique()

    total_gmv = gmv_by_ch.sum()
    refund_by_ch = refunded.groupby("channel")["order_id"].nunique()

    result = pd.DataFrame({
        "channel": gmv_by_ch.index,
        "gmv": gmv_by_ch.values,
        "order_count": orders_by_ch.values,
        "paying_users": users_by_ch.values,
    })

    result["avg_order_value"] = result.apply(
        lambda r: round(safe_divide(r["gmv"], r["order_count"]), 2), axis=1
    )
    result["gmv_ratio"] = result["gmv"] / total_gmv

    result = result.merge(
        pd.DataFrame({"channel": refund_by_ch.index, "refund_count": refund_by_ch.values}),
        on="channel", how="left"
    )
    result["refund_count"] = result["refund_count"].fillna(0)
    result["total_non_canceled"] = result["order_count"] + result["refund_count"]
    result["refund_rate"] = result.apply(
        lambda r: round(safe_divide(r["refund_count"], r["total_non_canceled"]), 4), axis=1
    )

    result = result.sort_values("gmv", ascending=False).reset_index(drop=True)
    return result[["channel", "gmv", "order_count", "paying_users", "avg_order_value", "refund_rate", "gmv_ratio"]]


def calculate_repurchase_metrics(df: pd.DataFrame) -> dict:
    """计算用户复购指标"""
    paid = df[df["order_status"] == "paid"]

    user_stats = paid.groupby("user_id").agg(
        order_count=("order_id", "nunique"),
        total_amount=("order_amount", "sum"),
        first_order_time=("order_time", "min"),
        last_order_time=("order_time", "max"),
    ).reset_index()

    user_stats["avg_order_value"] = user_stats.apply(
        lambda r: round(safe_divide(r["total_amount"], r["order_count"]), 2), axis=1
    )
    user_stats["is_repurchase"] = user_stats["order_count"] >= 2

    total_users = len(user_stats)
    repurchase_users = user_stats["is_repurchase"].sum()

    # 按渠道计算复购率
    user_channel = paid.groupby("user_id")["channel"].agg(lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else x.iloc[0]).reset_index()
    user_stats = user_stats.merge(user_channel, on="user_id", how="left")

    channel_repurchase = user_stats.groupby("channel").agg(
        total_users=("user_id", "nunique"),
        repurchase_users=("is_repurchase", "sum"),
    ).reset_index()
    channel_repurchase["repurchase_rate"] = channel_repurchase.apply(
        lambda r: round(safe_divide(r["repurchase_users"], r["total_users"]), 4), axis=1
    )

    overall = {
        "total_users": total_users,
        "repurchase_users": int(repurchase_users),
        "overall_repurchase_rate": round(safe_divide(repurchase_users, total_users), 4),
        "channel_repurchase": channel_repurchase.to_dict("records"),
        "user_detail": user_stats.to_dict("records"),
    }

    return overall
