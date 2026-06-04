"""Streamlit 主程序：电商经营分析与智能洞察生成"""

import os
import json
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

import streamlit as st
import pandas as pd

from src.data_loader import load_all_data
from src.data_cleaning import clean_orders, clean_users, clean_products, build_analysis_table
from src.metrics import (
    calculate_overall_metrics,
    calculate_time_trend,
    calculate_category_metrics,
    calculate_channel_metrics,
    calculate_repurchase_metrics,
)
from src.analysis import build_analysis_summary
from src.anomaly import detect_trend_anomalies, analyze_gmv_drop_reason
from src.visualization import (
    plot_gmv_trend,
    plot_order_count_trend,
    plot_avg_order_value_trend,
    plot_category_gmv,
    plot_channel_gmv,
    plot_category_gmv_pie,
    plot_repurchase_by_channel,
)
from src.llm_report import call_deepseek_api, generate_fallback_report
from src.utils import format_currency, format_percent, safe_divide

st.set_page_config(
    page_title="电商经营分析系统",
    page_icon="📊",
    layout="wide",
)

st.title("基于 DeepSeek-V4 的电商经营分析与智能洞察生成")
st.caption("使用 pandas 完成指标计算，使用 DeepSeek-V4 辅助生成经营分析报告")

# ============================================================
# 侧边栏
# ============================================================
with st.sidebar:
    st.header("⚙️ 数据配置")
    data_dir = st.text_input("数据目录", value="data")

    st.subheader("📅 分析时间范围")
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("开始日期", value=datetime(2024, 1, 1))
    with col2:
        end_date = st.date_input("结束日期", value=datetime(2024, 12, 31))

    freq_map = {"按天": "D", "按周": "W", "按月": "M"}
    freq_label = st.selectbox("趋势分析频率", list(freq_map.keys()), index=2)
    freq = freq_map[freq_label]

    st.divider()
    st.subheader("🤖 DeepSeek 配置")
    use_llm = st.checkbox("调用 DeepSeek 生成分析报告", value=False)

    api_key = None
    model = "deepseek-v4-flash"
    if use_llm:
        api_key = st.text_input(
            "DeepSeek API Key",
            type="password",
            value=os.getenv("DEEPSEEK_API_KEY", ""),
            help="也可在 .env 文件中设置 DEEPSEEK_API_KEY",
        )
        model = st.selectbox("模型", ["deepseek-v4-flash", "deepseek-v4-pro"], index=0)

    st.divider()
    st.caption("📊 电商经营分析系统 v1.0")

# ============================================================
# 加载数据
# ============================================================
@st.cache_data
def load_and_clean_data(data_dir: str):
    raw = load_all_data(data_dir)
    orders = clean_orders(raw["orders"])
    users = clean_users(raw["users"])
    products = clean_products(raw["products"])
    df = build_analysis_table(orders, users, products)
    return df, orders, users, products

try:
    df, orders, users, products = load_and_clean_data(data_dir)
except FileNotFoundError as e:
    st.error(f"数据文件未找到: {e}")
    st.info("请确保 data/ 目录下有 orders.csv、users.csv、products.csv 三个文件。可运行 `python generate_data.py` 生成模拟数据。")
    st.stop()

df["order_date"] = df["order_time"].dt.date
df = df[(df["order_date"] >= start_date) & (df["order_date"] <= end_date)]

if len(df) == 0:
    st.warning("当前时间范围内没有数据，请调整日期筛选。")
    st.stop()

# ============================================================
# 预计算所有指标（各 Tab 共用）
# ============================================================
overall = calculate_overall_metrics(df)
trend_monthly = calculate_time_trend(df, freq="M")
trend_display = calculate_time_trend(df, freq=freq)
category_df = calculate_category_metrics(df)
channel_df = calculate_channel_metrics(df)
repurchase = calculate_repurchase_metrics(df)
anomalies = detect_trend_anomalies(trend_monthly)
analysis_summary = build_analysis_summary(df)
analysis_summary["anomalies"] = anomalies

# ============================================================
# 分析洞察函数（规则驱动，不依赖 LLM）
# ============================================================

def generate_health_insight(overall, trend_monthly):
    """生成经营健康度摘要"""
    gmv = overall["gmv"]
    aov = overall["avg_order_value"]
    refund_rate = overall["refund_rate"]
    repurchase_rate = overall["repurchase_rate"]
    order_count = overall["order_count"]

    lines = [f"本期 GMV 为 **{gmv/10000:,.0f} 万元**，订单量 **{order_count:,} 单**，客单价 **{aov:,.0f} 元**。"]

    # 最新月趋势判断
    if len(trend_monthly) >= 2:
        latest = trend_monthly.iloc[-1]
        prev = trend_monthly.iloc[-2]
        gmv_mom = latest.get("gmv_mom")
        order_mom = latest.get("order_count_mom")
        aov_mom = latest.get("avg_order_value_mom")

        if pd.notna(gmv_mom) and pd.notna(order_mom):
            if gmv_mom > 0.05:
                if order_mom > aov_mom:
                    lines.append(f"GMV 环比增长 **{gmv_mom*100:.1f}%**，增长主要来自**订单量提升**，说明购买规模在扩大。")
                else:
                    lines.append(f"GMV 环比增长 **{gmv_mom*100:.1f}%**，增长主要来自**客单价提升**，建议关注高价品类占比变化。")
            elif gmv_mom < -0.05:
                if order_mom < aov_mom:
                    lines.append(f"GMV 环比下降 **{abs(gmv_mom)*100:.1f}%**，下滑主因是**订单量减少**，问题更可能出在流量或转化环节。")
                else:
                    lines.append(f"GMV 环比下降 **{abs(gmv_mom)*100:.1f}%**，下滑主因是**客单价降低**，需关注品类结构和折扣力度。")
            else:
                lines.append("GMV 环比基本持平，经营状态稳定。")

    # 增长质量判断
    if repurchase_rate > 0.30:
        lines.append(f"复购率 **{repurchase_rate*100:.1f}%**，用户粘性较好，增长具有持续性基础。")
    else:
        lines.append(f"复购率 **{repurchase_rate*100:.1f}%**，处于偏低水平，建议加强用户留存和复购激励。")

    if refund_rate > 0.10:
        lines.append(f"退款率 **{refund_rate*100:.1f}%**，偏高，需要重点关注高退款品类和渠道。")
    else:
        lines.append(f"退款率 **{refund_rate*100:.1f}%**，处于可控范围。")

    return "\n\n".join(lines)


def classify_category(gmv_rank, gmv_ratio, refund_rate, aov, avg_aov):
    """分类品类健康度标签"""
    if gmv_ratio > 0.15 and refund_rate < 0.08:
        return "🟢 核心优势品类"
    elif gmv_ratio > 0.15 and refund_rate >= 0.08:
        return "🟡 高贡献但需关注"
    elif gmv_ratio <= 0.10 and aov > avg_aov:
        return "🔵 潜力品类"
    elif refund_rate >= 0.12:
        return "🔴 低效风险品类"
    else:
        return "⚪ 一般品类"


def classify_channel(gmv_ratio, repurchase_rate, refund_rate):
    """分类渠道质量标签"""
    if gmv_ratio > 0.15 and repurchase_rate > 0.25 and refund_rate < 0.08:
        return "🟢 优质渠道"
    elif gmv_ratio > 0.15 and refund_rate >= 0.10:
        return "🟡 高流量但质量需关注"
    elif gmv_ratio <= 0.10 and repurchase_rate > 0.25:
        return "🔵 小而美渠道"
    elif repurchase_rate <= 0.20 and refund_rate >= 0.10:
        return "🔴 低效渠道"
    else:
        return "⚪ 一般渠道"


def generate_anomaly_diagnosis(anomalies, trend_monthly, df):
    """增强异常诊断：输出更完整的判断"""
    if not anomalies:
        return "本期未检测到显著异常波动，各项指标处于正常范围。"

    diag_lines = []
    for a in anomalies:
        period = a["period"]
        metric = a["metric"]
        change = a["change_rate"]
        atype = a["anomaly_type"]
        direction = "下降" if atype == "decline" else "异常增长"
        diag_lines.append(f"- **{period} {metric} 环比{direction} {abs(change)*100:.1f}%**")

    # 组合判断
    gmv_anomaly = any(a["metric"] == "gmv" and a["anomaly_type"] == "decline" for a in anomalies)
    refund_anomaly = any(a["metric"] == "refund_rate" and a["anomaly_type"] == "surge" for a in anomalies)
    order_anomaly = any(a["metric"] == "order_count" and a["anomaly_type"] == "decline" for a in anomalies)

    if gmv_anomaly and order_anomaly:
        diag_lines.append("\n💡 **综合判断**：GMV 下滑同时伴随订单量下降，问题更可能出在**流量获取或用户转化**环节，建议排查各渠道引流效率和转化漏斗。")
    elif gmv_anomaly and refund_anomaly:
        diag_lines.append("\n💡 **综合判断**：GMV 下滑同时伴随退款率上升，建议**优先排查高退款品类和渠道**，可能存在商品质量或服务问题。")
    elif gmv_anomaly:
        diag_lines.append("\n💡 **综合判断**：GMV 出现下滑，建议按 GMV = 订单量 × 客单价框架，结合品类和渠道维度进一步拆解定位。")

    return "\n".join(diag_lines)


# ============================================================
# 品类/渠道标签生成
# ============================================================
avg_aov = category_df["avg_order_value"].mean()
category_df["category_tag"] = category_df.apply(
    lambda r: classify_category(
        r["gmv_ratio"], r["gmv_ratio"], r["refund_rate"],
        r["avg_order_value"], avg_aov
    ), axis=1
)

# 渠道复购率
if repurchase.get("channel_repurchase"):
    ch_rep_map = {r["channel"]: r["repurchase_rate"] for r in repurchase["channel_repurchase"]}
else:
    ch_rep_map = {}
channel_df["repurchase_rate"] = channel_df["channel"].map(ch_rep_map).fillna(0)
channel_df["channel_tag"] = channel_df.apply(
    lambda r: classify_channel(r["gmv_ratio"], r["repurchase_rate"], r["refund_rate"]),
    axis=1
)

# ============================================================
# Tab 布局
# ============================================================
tabs = st.tabs([
    "🏠 经营总览",
    "📈 趋势分析",
    "🛍️ 品类分析",
    "📡 渠道分析",
    "🔄 用户复购",
    "⚠️ 异常诊断",
    "🤖 AI 报告",
])

# ============================================================
# Tab 1: 经营总览
# ============================================================
with tabs[0]:
    # 数据概览
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("订单总数", f"{len(df):,}")
    with col2:
        st.metric("用户数", f"{df['user_id'].nunique():,}")
    with col3:
        st.metric("商品数", f"{df['product_id'].nunique():,}")
    with col4:
        st.metric("品类数", f"{df['category'].nunique():,}")
    with col5:
        st.metric("渠道数", f"{df['channel'].nunique():,}")
    with col6:
        t_min = df["order_time"].min().strftime("%Y-%m-%d")
        t_max = df["order_time"].max().strftime("%Y-%m-%d")
        st.metric("时间范围", t_min, delta=t_max)

    st.divider()

    # 核心指标卡片
    st.subheader("核心经营指标")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.metric("GMV", f"{overall['gmv']/10000:,.2f} 万元")
    with col2:
        st.metric("订单量", f"{overall['order_count']:,}")
    with col3:
        st.metric("支付用户数", f"{overall['paying_users']:,}")
    with col4:
        st.metric("客单价", f"{overall['avg_order_value']:,.0f} 元")
    with col5:
        st.metric("复购率", format_percent(overall["repurchase_rate"]))
    with col6:
        st.metric("退款率", format_percent(overall["refund_rate"]))

    if len(trend_monthly) >= 2:
        latest = trend_monthly.iloc[-1]
        prev = trend_monthly.iloc[-2]
        st.caption(
            f"环比变化 — GMV: {format_percent(latest['gmv_mom'])} | "
            f"订单量: {format_percent(latest['order_count_mom'])} | "
            f"客单价: {format_percent(latest['avg_order_value_mom'])}"
        )

    st.divider()

    # 经营健康度摘要
    st.subheader("📋 经营健康度摘要")
    insight_text = generate_health_insight(overall, trend_monthly)
    st.info(insight_text)

# ============================================================
# Tab 2: 趋势分析
# ============================================================
with tabs[1]:
    st.subheader("指标趋势")

    subtab1, subtab2, subtab3 = st.tabs(["GMV 趋势", "订单量趋势", "客单价趋势"])

    with subtab1:
        fig = plot_gmv_trend(trend_display, anomalies)
        st.pyplot(fig)
        if len(trend_display) >= 2:
            last_mom = trend_display.iloc[-1]["gmv_mom"]
            if pd.notna(last_mom):
                if last_mom > 0.05:
                    st.success(f"本期 GMV 环比增长 {last_mom*100:.1f}%，整体呈上升趋势。")
                elif last_mom < -0.05:
                    st.warning(f"本期 GMV 环比下降 {abs(last_mom)*100:.1f}%，建议结合品类和渠道维度进一步分析。")
                else:
                    st.info(f"本期 GMV 环比变化 {last_mom*100:.1f}%，保持稳定。")

    with subtab2:
        fig = plot_order_count_trend(trend_display)
        st.pyplot(fig)

    with subtab3:
        fig = plot_avg_order_value_trend(trend_display)
        st.pyplot(fig)

# ============================================================
# Tab 3: 品类分析
# ============================================================
with tabs[2]:
    st.subheader("品类销售分析")

    col1, col2 = st.columns([3, 2])
    with col1:
        fig = plot_category_gmv(category_df)
        st.pyplot(fig)
    with col2:
        fig = plot_category_gmv_pie(category_df)
        st.pyplot(fig)

    st.subheader("品类健康度明细")
    st.caption("标签根据 GMV 占比、退款率、客单价综合判定")

    display_cat = category_df[["category", "gmv", "order_count", "avg_order_value",
                                "refund_rate", "gmv_ratio", "category_tag"]].copy()
    st.dataframe(
        display_cat.style.format({
            "gmv": lambda x: f"{x/10000:,.2f}万",
            "avg_order_value": lambda x: f"{x:,.0f}元",
            "refund_rate": "{:.2%}",
            "gmv_ratio": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # 品类健康度解读
    core_cats = category_df[category_df["category_tag"].str.contains("核心")]
    risk_cats = category_df[category_df["category_tag"].str.contains("风险")]
    potential_cats = category_df[category_df["category_tag"].str.contains("潜力")]

    if len(core_cats) > 0:
        names = "、".join(core_cats["category"].tolist())
        st.success(f"**核心优势品类**：{names} — GMV 占比高且退款率低，是当前经营基本盘。")
    if len(risk_cats) > 0:
        names = "、".join(risk_cats["category"].tolist())
        st.warning(f"**低效风险品类**：{names} — 退款率偏高，建议排查退款的二级品类分布和退款原因。")
    if len(potential_cats) > 0:
        names = "、".join(potential_cats["category"].tolist())
        st.info(f"**潜力品类**：{names} — GMV 占比较低但客单价较高，可尝试加大曝光和推广力度。")

# ============================================================
# Tab 4: 渠道分析
# ============================================================
with tabs[3]:
    st.subheader("渠道销售分析")

    fig = plot_channel_gmv(channel_df)
    st.pyplot(fig)

    st.subheader("渠道质量明细")
    st.caption("标签结合 GMV 占比、复购率、退款率综合判定")

    display_ch = channel_df[["channel", "gmv", "order_count", "avg_order_value",
                              "refund_rate", "repurchase_rate", "gmv_ratio", "channel_tag"]].copy()
    st.dataframe(
        display_ch.style.format({
            "gmv": lambda x: f"{x/10000:,.2f}万",
            "avg_order_value": lambda x: f"{x:,.0f}元",
            "refund_rate": "{:.2%}",
            "repurchase_rate": "{:.2%}",
            "gmv_ratio": "{:.2%}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # 渠道质量解读
    quality_chs = channel_df[channel_df["channel_tag"].str.contains("优质")]
    high_risk_chs = channel_df[channel_df["channel_tag"].str.contains("质量需关注")]
    niche_chs = channel_df[channel_df["channel_tag"].str.contains("小而美")]
    low_chs = channel_df[channel_df["channel_tag"].str.contains("低效")]

    if len(quality_chs) > 0:
        names = "、".join(quality_chs["channel"].tolist())
        st.success(f"**优质渠道**：{names} — GMV 贡献高、复购好、退款低，建议保持投放力度。")
    if len(high_risk_chs) > 0:
        names = "、".join(high_risk_chs["channel"].tolist())
        st.warning(f"**高流量但质量需关注**：{names} — 流量规模大但退款率偏高，建议优化选品和售后。")
    if len(niche_chs) > 0:
        names = "、".join(niche_chs["channel"].tolist())
        st.info(f"**小而美渠道**：{names} — 规模不大但复购率高、用户质量好，可尝试加大投放测试。")
    if len(low_chs) > 0:
        names = "、".join(low_chs["channel"].tolist())
        st.warning(f"**低效渠道**：{names} — 复购和退款指标均不理想，需评估是否继续投入。")

# ============================================================
# Tab 5: 用户复购
# ============================================================
with tabs[4]:
    st.subheader("用户复购分析")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("总购买用户数", f"{repurchase['total_users']:,}")
    with col2:
        st.metric("复购用户数", f"{repurchase['repurchase_users']:,}")
    with col3:
        st.metric("整体复购率", format_percent(repurchase["overall_repurchase_rate"]))

    channel_rep = repurchase.get("channel_repurchase", [])
    if channel_rep:
        fig = plot_repurchase_by_channel(channel_rep)
        st.pyplot(fig)

    st.subheader("复购率解读")
    overall_rep = repurchase["overall_repurchase_rate"]
    if overall_rep > 0.35:
        st.success(f"整体复购率 {overall_rep*100:.1f}%，用户粘性较好。说明商品或服务能够有效留住用户，增长基础扎实。")
    elif overall_rep > 0.25:
        st.info(f"整体复购率 {overall_rep*100:.1f}%，处于中等水平。建议通过会员体系、复购优惠券等方式提升复购。")
    else:
        st.warning(f"整体复购率 {overall_rep*100:.1f}%，偏低。新客占比过高，需关注用户留存和首次购买后的触达。")

    if channel_rep:
        best_ch = max(channel_rep, key=lambda x: x["repurchase_rate"])
        worst_ch = min(channel_rep, key=lambda x: x["repurchase_rate"])
        st.caption(
            f"复购率最高渠道：**{best_ch['channel']}**（{best_ch['repurchase_rate']*100:.1f}%），"
            f"最低渠道：**{worst_ch['channel']}**（{worst_ch['repurchase_rate']*100:.1f}%）"
        )

# ============================================================
# Tab 6: 异常诊断
# ============================================================
with tabs[5]:
    st.subheader("异常波动诊断")

    # 增强诊断
    diagnosis = generate_anomaly_diagnosis(anomalies, trend_monthly, df)
    if anomalies:
        st.warning(diagnosis)
    else:
        st.success(diagnosis)

    if anomalies:
        st.divider()
        st.subheader("异常明细")

        anomaly_df = pd.DataFrame(anomalies)
        st.dataframe(
            anomaly_df[["period", "metric", "change_rate", "anomaly_type", "description"]].style.format({
                "change_rate": "{:.2%}",
            }),
            use_container_width=True,
            hide_index=True,
        )

        # GMV 下滑归因
        gmv_anomalies = [a for a in anomalies if a["metric"] == "gmv" and a["anomaly_type"] == "decline"]
        if gmv_anomalies:
            st.subheader("🔍 GMV 下滑归因拆解")
            trend_list = trend_monthly["period"].tolist()
            for anomaly in gmv_anomalies:
                period = anomaly["period"]
                idx = trend_list.index(period) if period in trend_list else -1
                if idx > 0:
                    prev_period = trend_list[idx - 1]
                    attr = analyze_gmv_drop_reason(df, period, prev_period)
                    with st.expander(f"{period} GMV 下滑归因详情", expanded=True):
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("GMV 变化", format_percent(attr.get("gmv_change", 0)))
                        with col2:
                            st.metric("订单量变化", format_percent(attr.get("order_count_change", 0)))
                        with col3:
                            st.metric("客单价变化", format_percent(attr.get("avg_order_value_change", 0)))

                        st.markdown("**归因分析：**")
                        for reason in attr.get("possible_reasons", []):
                            st.markdown(f"- {reason}")

                        decline_cats = attr.get("top_decline_categories", [])
                        if decline_cats:
                            st.markdown("**GMV 下降最多的品类：**")
                            cat_data = [{
                                "品类": c["category"],
                                "当月GMV": f"{c['curr_gmv']/10000:,.1f}万",
                                "上月GMV": f"{c['prev_gmv']/10000:,.1f}万",
                                "变化": format_percent(c["gmv_change_rate"]),
                            } for c in decline_cats]
                            st.dataframe(pd.DataFrame(cat_data), use_container_width=True, hide_index=True)

    # 指标异常矩阵
    st.divider()
    st.subheader("📊 指标异常矩阵")
    if len(trend_monthly) >= 2:
        latest = trend_monthly.iloc[-1]
        prev = trend_monthly.iloc[-2]

        matrix_data = []
        for metric, threshold, label in [
            ("gmv_mom", -0.20, "GMV"),
            ("order_count_mom", -0.20, "订单量"),
            ("avg_order_value_mom", -0.15, "客单价"),
        ]:
            val = latest.get(metric)
            if pd.notna(val):
                if val < threshold:
                    status = "🔴 异常下降"
                elif val < 0:
                    status = "🟡 轻微下降"
                elif val > 0.30:
                    status = "🟢 异常增长"
                elif val > 0:
                    status = "🟢 正常增长"
                else:
                    status = "⚪ 持平"
                matrix_data.append({"指标": label, "环比变化": format_percent(val), "状态": status})

        refund_latest = overall.get("refund_rate", 0)
        refund_status = "🔴 偏高" if refund_latest > 0.12 else ("🟡 关注" if refund_latest > 0.08 else "🟢 正常")
        matrix_data.append({"指标": "退款率", "环比变化": format_percent(refund_latest), "状态": refund_status})

        st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)

# ============================================================
# Tab 7: AI 报告
# ============================================================
with tabs[6]:
    st.subheader("🤖 DeepSeek-V4 智能分析报告")

    if use_llm and api_key:
        if st.button("🚀 生成智能分析报告", type="primary"):
            with st.spinner("正在调用 DeepSeek-V4 生成分析报告..."):
                try:
                    report = call_deepseek_api(analysis_summary, api_key, model)
                    st.success("报告生成成功！")
                    st.markdown(report)
                    st.download_button(
                        "📥 下载报告 (Markdown)",
                        data=report,
                        file_name=f"ecommerce_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                        mime="text/markdown",
                    )
                except Exception as e:
                    st.error(f"API 调用失败: {e}")
                    st.info("使用本地模板报告作为备选...")
                    report = generate_fallback_report(analysis_summary)
                    st.markdown(report)
    elif use_llm and not api_key:
        st.warning("请输入 DeepSeek API Key")
    else:
        st.info('在侧边栏勾选"调用 DeepSeek 生成分析报告"并输入 API Key，即可生成 AI 分析报告。')

    # 始终可以查看本地模板报告
    with st.expander("📄 查看本地模板报告（无需 API Key）"):
        report = generate_fallback_report(analysis_summary)
        st.markdown(report)
        st.download_button(
            "📥 下载报告 (Markdown)",
            data=report,
            file_name=f"ecommerce_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
        )

st.divider()
st.caption("电商经营分析系统 — 指标由 pandas 计算，报告由 DeepSeek-V4 辅助生成")
