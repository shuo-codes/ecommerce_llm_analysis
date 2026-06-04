"""业务分析模块：组装多维分析结果，生成结构化分析摘要"""

import pandas as pd
from src.metrics import (
    calculate_overall_metrics,
    calculate_time_trend,
    calculate_category_metrics,
    calculate_channel_metrics,
    calculate_repurchase_metrics,
)


def build_analysis_summary(df: pd.DataFrame) -> dict:
    """生成完整结构化分析摘要，供可视化展示和 DeepSeek 报告生成使用"""
    overall = calculate_overall_metrics(df)
    trend_monthly = calculate_time_trend(df, freq="M")
    trend_weekly = calculate_time_trend(df, freq="W")
    category = calculate_category_metrics(df)
    channel = calculate_channel_metrics(df)
    repurchase = calculate_repurchase_metrics(df)

    # Top 品类和渠道
    top_categories = category.head(5).to_dict("records")
    top_channels = channel.head(5).to_dict("records")

    # 趋势摘要
    trend_summary = trend_monthly[["period", "gmv", "order_count", "avg_order_value", "gmv_mom"]].to_dict("records")

    return {
        "overall_metrics": overall,
        "trend_summary": trend_summary,
        "trend_monthly": trend_monthly,
        "trend_weekly": trend_weekly,
        "category_metrics": category,
        "channel_metrics": channel,
        "repurchase_metrics": repurchase,
        "top_categories": top_categories,
        "top_channels": top_channels,
    }
