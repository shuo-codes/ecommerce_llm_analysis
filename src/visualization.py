"""可视化模块：使用 matplotlib 绘制经营分析图表"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

# 中文字体设置
# 策略：sans-serif 列表第一字体负责中文，回退字体负责 ￥ 等特殊符号
import matplotlib.font_manager as fm
import os

# 清除 matplotlib 字体缓存
_font_cache_dir = matplotlib.get_cachedir()
for _fname in os.listdir(_font_cache_dir):
    if _fname.endswith(".json"):
        os.remove(os.path.join(_font_cache_dir, _fname))
fm._load_fontmanager(try_read_cache=False)

# 注册 SimHei 字体文件
_simhei_path = "C:/Windows/Fonts/simhei.ttf"
if os.path.exists(_simhei_path):
    fm.fontManager.addfont(_simhei_path)

# sans-serif 回退链：SimHei 负责中文，DejaVu Sans 负责 ￥ 等特殊符号
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

COLORS = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B",
          "#2A9D8F", "#E9C46A", "#F4A261", "#E76F51", "#264653"]


def _style_ax(ax, title: str, xlabel: str = "", ylabel: str = ""):
    """通用图表样式设置"""
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(labelsize=10)


def plot_gmv_trend(trend_df: pd.DataFrame, anomalies: list = None) -> plt.Figure:
    """GMV 趋势折线图，标记异常月份"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    periods = trend_df["period"].tolist()
    gmv = trend_df["gmv"].tolist()

    ax.plot(periods, gmv, marker="o", linewidth=2, color=COLORS[0], markersize=6)
    ax.fill_between(range(len(periods)), gmv, alpha=0.1, color=COLORS[0])

    # 标记异常月份
    if anomalies:
        anomaly_periods = [a["period"] for a in anomalies if a["metric"] == "gmv" and a["anomaly_type"] == "decline"]
        for i, p in enumerate(periods):
            if p in anomaly_periods:
                ax.scatter(i, gmv[i], color="red", s=120, zorder=5, edgecolors="white", linewidth=1.5)
                ax.annotate("异常", (i, gmv[i]), textcoords="offset points",
                            xytext=(0, -20), ha="center", fontsize=9, color="red", fontweight="bold")

    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"￥{x/10000:.0f}万"))
    _style_ax(ax, "GMV 月度趋势", ylabel="GMV")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_order_count_trend(trend_df: pd.DataFrame) -> plt.Figure:
    """订单量趋势折线图"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    periods = trend_df["period"].tolist()
    orders = trend_df["order_count"].tolist()

    ax.plot(periods, orders, marker="s", linewidth=2, color=COLORS[1], markersize=6)
    ax.fill_between(range(len(periods)), orders, alpha=0.1, color=COLORS[1])
    _style_ax(ax, "订单量月度趋势", ylabel="订单量")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_avg_order_value_trend(trend_df: pd.DataFrame) -> plt.Figure:
    """客单价趋势折线图"""
    fig, ax = plt.subplots(figsize=(10, 4.5))
    periods = trend_df["period"].tolist()
    aov = trend_df["avg_order_value"].tolist()

    ax.plot(periods, aov, marker="D", linewidth=2, color=COLORS[2], markersize=6)
    ax.fill_between(range(len(periods)), aov, alpha=0.1, color=COLORS[2])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"￥{x:.0f}"))
    _style_ax(ax, "客单价月度趋势", ylabel="客单价")
    plt.xticks(rotation=45)
    plt.tight_layout()
    return fig


def plot_category_gmv(category_df: pd.DataFrame, top_n: int = 10) -> plt.Figure:
    """品类 GMV 排名横向柱状图"""
    df = category_df.head(top_n).sort_values("gmv", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    bars = ax.barh(df["category"], df["gmv"], color=COLORS[:len(df)], edgecolor="white")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"￥{x/10000:.0f}万"))
    _style_ax(ax, f"品类 GMV 排名 (Top {top_n})", xlabel="GMV")

    for bar, val in zip(bars, df["gmv"]):
        ax.text(bar.get_width() + max(df["gmv"]) * 0.01, bar.get_y() + bar.get_height()/2,
                f"￥{val/10000:.1f}万", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_channel_gmv(channel_df: pd.DataFrame) -> plt.Figure:
    """渠道 GMV 对比柱状图"""
    df = channel_df.sort_values("gmv", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 4.5))

    bars = ax.barh(df["channel"], df["gmv"], color=COLORS[:len(df)], edgecolor="white")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"￥{x/10000:.0f}万"))
    _style_ax(ax, "渠道 GMV 对比", xlabel="GMV")

    for bar, val in zip(bars, df["gmv"]):
        ax.text(bar.get_width() + max(df["gmv"]) * 0.01, bar.get_y() + bar.get_height()/2,
                f"￥{val/10000:.1f}万", va="center", fontsize=9)

    plt.tight_layout()
    return fig


def plot_category_gmv_pie(category_df: pd.DataFrame) -> plt.Figure:
    """品类销售占比饼图"""
    df = category_df.head(8)
    other_gmv = category_df.iloc[8:]["gmv"].sum() if len(category_df) > 8 else 0

    labels = df["category"].tolist()
    values = df["gmv"].tolist()
    if other_gmv > 0:
        labels.append("其他")
        values.append(other_gmv)

    fig, ax = plt.subplots(figsize=(7, 7))
    wedges, texts, autotexts = ax.pie(
        values, labels=labels, autopct="%1.1f%%",
        colors=COLORS[:len(labels)], startangle=90,
        pctdistance=0.75,
    )
    for t in autotexts:
        t.set_fontsize(9)
    for t in texts:
        t.set_fontsize(10)
    ax.set_title("品类 GMV 销售占比", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    return fig


def plot_repurchase_by_channel(channel_repurchase: list) -> plt.Figure:
    """各渠道复购率对比图"""
    if not channel_repurchase:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "暂无复购数据", ha="center", va="center", fontsize=14)
        return fig

    channels = [r["channel"] for r in channel_repurchase]
    rates = [r["repurchase_rate"] * 100 for r in channel_repurchase]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(channels, rates, color=COLORS[:len(channels)], edgecolor="white")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.1f}%"))
    _style_ax(ax, "各渠道复购率对比", ylabel="复购率")

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{rate:.1f}%", ha="center", fontsize=10, fontweight="bold")

    plt.tight_layout()
    return fig
