"""异常检测模块：基于环比阈值识别指标异常波动，并对 GMV 下滑进行归因拆解"""

import pandas as pd
from src.metrics import calculate_time_trend, calculate_category_metrics, calculate_channel_metrics
from src.utils import safe_divide

# 异常阈值
GMV_DECLINE_THRESHOLD = -0.20      # GMV 环比下降超 20%
GMV_SURGE_THRESHOLD = 0.30         # GMV 环比增长超 30%
ORDER_DECLINE_THRESHOLD = -0.20    # 订单量环比下降超 20%
AOV_DECLINE_THRESHOLD = -0.15      # 客单价环比下降超 15%


def detect_trend_anomalies(trend_df: pd.DataFrame) -> list:
    """检测时间趋势中的异常波动"""
    anomalies = []

    for _, row in trend_df.iterrows():
        period = row["period"]

        # GMV 异常
        gmv_mom = row.get("gmv_mom")
        if pd.notna(gmv_mom):
            if gmv_mom < GMV_DECLINE_THRESHOLD:
                anomalies.append({
                    "period": period,
                    "metric": "gmv",
                    "change_rate": round(gmv_mom, 4),
                    "anomaly_type": "decline",
                    "description": f"{period} GMV 环比下降 {abs(gmv_mom)*100:.1f}%，超过设定阈值 ({abs(GMV_DECLINE_THRESHOLD)*100:.0f}%)",
                })
            elif gmv_mom > GMV_SURGE_THRESHOLD:
                anomalies.append({
                    "period": period,
                    "metric": "gmv",
                    "change_rate": round(gmv_mom, 4),
                    "anomaly_type": "surge",
                    "description": f"{period} GMV 环比异常增长 {gmv_mom*100:.1f}%，超过设定阈值 ({GMV_SURGE_THRESHOLD*100:.0f}%)",
                })

        # 订单量异常
        order_mom = row.get("order_count_mom")
        if pd.notna(order_mom) and order_mom < ORDER_DECLINE_THRESHOLD:
            anomalies.append({
                "period": period,
                "metric": "order_count",
                "change_rate": round(order_mom, 4),
                "anomaly_type": "decline",
                "description": f"{period} 订单量环比下降 {abs(order_mom)*100:.1f}%，超过设定阈值",
            })

        # 客单价异常
        aov_mom = row.get("avg_order_value_mom")
        if pd.notna(aov_mom) and aov_mom < AOV_DECLINE_THRESHOLD:
            anomalies.append({
                "period": period,
                "metric": "avg_order_value",
                "change_rate": round(aov_mom, 4),
                "anomaly_type": "decline",
                "description": f"{period} 客单价环比下降 {abs(aov_mom)*100:.1f}%，超过设定阈值",
            })

    return anomalies


def analyze_gmv_drop_reason(df: pd.DataFrame, current_period: str, previous_period: str) -> dict:
    """对 GMV 下滑进行归因拆解：从订单量×客单价、品类、渠道三个维度定位原因

    Args:
        df: 分析宽表
        current_period: 当前月份，如 '2024-07'
        previous_period: 对比月份，如 '2024-06'
    """
    paid = df[df["order_status"] == "paid"].copy()
    paid["month"] = paid["order_time"].dt.strftime("%Y-%m")

    curr = paid[paid["month"] == current_period]
    prev = paid[paid["month"] == previous_period]

    if len(curr) == 0 or len(prev) == 0:
        return {"error": f"期间数据为空: {current_period} 或 {previous_period}"}

    curr_gmv = curr["order_amount"].sum()
    prev_gmv = prev["order_amount"].sum()
    gmv_change = safe_divide(curr_gmv - prev_gmv, prev_gmv)

    curr_orders = curr["order_id"].nunique()
    prev_orders = prev["order_id"].nunique()
    order_change = safe_divide(curr_orders - prev_orders, prev_orders)

    curr_aov = safe_divide(curr_gmv, curr_orders)
    prev_aov = safe_divide(prev_gmv, prev_orders)
    aov_change = safe_divide(curr_aov - prev_aov, prev_aov)

    # 品类维度拆解
    curr_cat = curr.groupby("category")["order_amount"].sum().reset_index()
    curr_cat.columns = ["category", "curr_gmv"]
    prev_cat = prev.groupby("category")["order_amount"].sum().reset_index()
    prev_cat.columns = ["category", "prev_gmv"]
    cat_comparison = curr_cat.merge(prev_cat, on="category", how="outer").fillna(0)
    cat_comparison["gmv_change"] = cat_comparison["curr_gmv"] - cat_comparison["prev_gmv"]
    cat_comparison["gmv_change_rate"] = cat_comparison.apply(
        lambda r: safe_divide(r["gmv_change"], r["prev_gmv"]), axis=1
    )
    cat_comparison = cat_comparison.sort_values("gmv_change")
    top_decline_categories = cat_comparison.head(3).to_dict("records")

    # 渠道维度拆解
    curr_ch = curr.groupby("channel")["order_amount"].sum().reset_index()
    curr_ch.columns = ["channel", "curr_gmv"]
    prev_ch = prev.groupby("channel")["order_amount"].sum().reset_index()
    prev_ch.columns = ["channel", "prev_gmv"]
    ch_comparison = curr_ch.merge(prev_ch, on="channel", how="outer").fillna(0)
    ch_comparison["gmv_change"] = ch_comparison["curr_gmv"] - ch_comparison["prev_gmv"]
    ch_comparison["gmv_change_rate"] = ch_comparison.apply(
        lambda r: safe_divide(r["gmv_change"], r["prev_gmv"]), axis=1
    )
    ch_comparison = ch_comparison.sort_values("gmv_change")
    top_decline_channels = ch_comparison.head(3).to_dict("records")

    # 生成可能原因
    possible_reasons = []
    if order_change < -0.05:
        possible_reasons.append(f"订单量下降 {abs(order_change)*100:.1f}% 是 GMV 下滑的主要原因")
    if aov_change < -0.05:
        possible_reasons.append(f"客单价下降 {abs(aov_change)*100:.1f}% 进一步拉低了 GMV")
    if len(top_decline_categories) > 0:
        worst_cat = top_decline_categories[0]
        if worst_cat["gmv_change"] < 0:
            possible_reasons.append(
                f"{worst_cat['category']}品类 GMV 下降贡献最大"
            )
    if len(top_decline_channels) > 0:
        worst_ch = top_decline_channels[0]
        if worst_ch["gmv_change"] < 0:
            possible_reasons.append(
                f"{worst_ch['channel']}渠道 GMV 下降明显"
            )

    return {
        "current_period": current_period,
        "previous_period": previous_period,
        "gmv_change": round(gmv_change, 4),
        "order_count_change": round(order_change, 4),
        "avg_order_value_change": round(aov_change, 4),
        "top_decline_categories": top_decline_categories,
        "top_decline_channels": top_decline_channels,
        "possible_reasons": possible_reasons,
    }
