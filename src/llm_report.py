"""DeepSeek 报告生成模块：基于结构化分析结果生成自然语言经营分析报告"""

import json
import os
from datetime import datetime


SYSTEM_PROMPT = """你是一名电商业务数据分析师。下面给你的是已经由 pandas / SQL 计算完成的结构化指标结果。
请注意：
1. 不要编造数据。
2. 不要使用输入中不存在的指标。
3. 所有结论必须基于给定数据。
4. 如果数据不足以判断原因，请明确说明"需要进一步结合活动、库存、投放等数据验证"。
5. 输出内容要面向业务方，语言简洁、结构清晰。

请根据以下数据生成一份经营分析报告，报告包含：

一、整体经营表现总结
二、核心指标变化解读
三、主要异常波动及可能原因
四、品类表现分析
五、渠道表现分析
六、用户复购情况分析
七、后续优化建议"""


def build_report_prompt(analysis_summary: dict) -> str:
    """构建传给 DeepSeek 的 user message（仅结构化数据，不含 system prompt）"""
    return json.dumps(analysis_summary, ensure_ascii=False, indent=2, default=str)


def call_deepseek_api(analysis_summary: dict, api_key: str, model: str = "deepseek-v4-flash") -> str:
    """调用 DeepSeek API 生成经营分析报告

    system message 传行为约束，user message 只传已计算完成的结构化指标数据。
    模型不接触原始订单明细，不负责数值计算。
    """
    try:
        from openai import OpenAI
    except ImportError:
        return generate_fallback_report(analysis_summary)

    client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    user_message = build_report_prompt(analysis_summary)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.7,
        max_tokens=4096,
    )

    return response.choices[0].message.content


def generate_fallback_report(analysis_summary: dict) -> str:
    """当 DeepSeek API 不可用时的本地模板报告"""
    overall = analysis_summary.get("overall_metrics", {})
    trend = analysis_summary.get("trend_summary", [])
    anomalies = analysis_summary.get("anomalies", [])
    top_categories = analysis_summary.get("top_categories", [])
    top_channels = analysis_summary.get("top_channels", [])
    repurchase = analysis_summary.get("repurchase_metrics", {})

    gmv = overall.get("gmv", 0)
    order_count = overall.get("order_count", 0)
    aov = overall.get("avg_order_value", 0)
    refund_rate = overall.get("refund_rate", 0)
    repurchase_rate = repurchase.get("overall_repurchase_rate", 0)

    report_lines = [
        "# 电商经营分析报告",
        f"\n生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"\n> 注：本报告为本地模板生成（DeepSeek API 未连接）。展示核心指标摘要，完整 AI 分析报告请配置 API Key 后重新生成。",
        "\n---",
        "\n## 一、整体经营表现总结",
        f"\n- **GMV**：¥{gmv:,.2f}",
        f"- **订单量**：{order_count}",
        f"- **客单价**：¥{aov:,.2f}",
        f"- **退款率**：{refund_rate*100:.2f}%",
        f"- **复购率**：{repurchase_rate*100:.2f}%",
        "\n---",
        "\n## 二、核心指标变化解读",
    ]

    if trend:
        report_lines.append("\n| 月份 | GMV | 订单量 | 客单价 | GMV环比 |")
        report_lines.append("|------|-----|--------|--------|---------|")
        for t in trend[-6:]:  # 最近6个月
            mom = f"{t.get('gmv_mom', 0)*100:.1f}%" if t.get("gmv_mom") is not None else "-"
            report_lines.append(
                f"| {t['period']} | ¥{t['gmv']:,.0f} | {t['order_count']} | "
                f"¥{t['avg_order_value']:,.2f} | {mom} |"
            )

    report_lines.append("\n---")
    report_lines.append("\n## 三、主要异常波动及可能原因")

    if anomalies:
        for a in anomalies:
            report_lines.append(f"\n- **{a['period']}**：{a['description']}")
    else:
        report_lines.append("\n未检测到显著异常波动。")

    report_lines.append("\n---")
    report_lines.append("\n## 四、品类表现分析")

    if top_categories:
        report_lines.append("\n| 品类 | GMV | 占比 |")
        report_lines.append("|------|-----|------|")
        for c in top_categories:
            report_lines.append(f"| {c['category']} | ¥{c['gmv']:,.0f} | {c['gmv_ratio']*100:.1f}% |")

    report_lines.append("\n---")
    report_lines.append("\n## 五、渠道表现分析")

    if top_channels:
        report_lines.append("\n| 渠道 | GMV | 占比 |")
        report_lines.append("|------|-----|------|")
        for ch in top_channels:
            report_lines.append(f"| {ch['channel']} | ¥{ch['gmv']:,.0f} | {ch['gmv_ratio']*100:.1f}% |")

    report_lines.append("\n---")
    report_lines.append("\n## 六、用户复购情况分析")
    report_lines.append(f"\n- 整体复购率：{repurchase_rate*100:.2f}%")

    channel_rep = repurchase.get("channel_repurchase", [])
    if channel_rep:
        for cr in channel_rep:
            report_lines.append(f"  - {cr['channel']}：{cr['repurchase_rate']*100:.2f}%")

    report_lines.append("\n---")
    report_lines.append("\n## 七、后续优化建议")
    report_lines.append("\n1. 针对异常下滑月份，建议结合当月的营销活动、竞品动态和库存情况进行深入复盘。")
    report_lines.append("2. 重点关注高 GMV 占比品类的退款率，降低退款率可有效提升整体 GMV。")
    report_lines.append("3. 对低复购率渠道，可尝试优化用户触达策略和售后服务体验。")
    report_lines.append("4. 建议引入更细粒度的用户行为数据，建立转化漏斗分析体系。")

    return "\n".join(report_lines)
