# 基于 DeepSeek-V4 的电商经营分析与智能洞察生成

## 项目背景

随着电商平台经营数据规模不断扩大，数据分析师需要快速完成指标计算、异常定位和分析报告撰写。本项目以电商订单、用户、商品和渠道数据为基础，构建经营分析看板，并引入 DeepSeek-V4 对已计算完成的结构化指标进行总结和解读，实现从数据指标到业务报告的半自动化生成。

> 

## 技术栈

| 技术            | 用途                |
| ------------- | ----------------- |
| Python 3.10+  | 项目主语言             |
| pandas        | 数据清洗、指标计算、多维分析    |
| matplotlib    | 数据可视化（折线图、柱状图、饼图） |
| Streamlit     | 交互式数据分析页面         |
| DeepSeek API  | 基于结构化指标生成自然语言分析报告 |
| python-dotenv | 环境变量管理            |

## 核心指标体系

### 整体经营指标

- **GMV** = 已支付订单金额总和
- **订单量** = 已支付订单数
- **支付用户数** = 产生支付行为的去重用户数
- **客单价** = GMV / 订单量
- **人均消费金额** = GMV / 支付用户数
- **退款率** = 退款订单数 / (已支付 + 已退款) 订单数
- **复购率** = 支付订单数 ≥ 2 的用户数 / 支付用户数

### 趋势指标

- 按天/周/月统计 GMV、订单量、支付用户数、客单价
- 环比变化率

### 品类分析

- 品类 GMV、订单量、客单价、退款率、GMV 占比

### 渠道分析

- 渠道 GMV、订单量、客单价、退款率、GMV 占比

### 异常检测

- GMV 环比下降超过 20% → 标记为下滑异常
- 订单量环比下降超过 20% → 标记为下滑异常
- 客单价环比下降超过 15% → 标记为下滑异常
- GMV 环比增长超过 30% → 标记为增长异常
- 异常归因：GMV = 订单量 × 客单价 拆解 + 品类/渠道维度下钻

## 项目结构

```
ecommerce_llm_analysis/
├── app.py                    # Streamlit 主程序
├── generate_data.py          # 模拟数据生成脚本
├── requirements.txt          # 项目依赖
├── .env.example              # 环境变量示例
├── README.md                 # 项目说明
├── data/
│   ├── orders.csv            # 订单数据
│   ├── users.csv             # 用户数据
│   └── products.csv          # 商品数据
├── src/
│   ├── data_loader.py        # 数据读取
│   ├── data_cleaning.py      # 数据清洗
│   ├── metrics.py            # 指标计算
│   ├── analysis.py           # 分析摘要组装
│   ├── anomaly.py            # 异常检测与归因
│   ├── visualization.py      # matplotlib 图表
│   ├── llm_report.py         # DeepSeek 报告生成
│   └── utils.py              # 工具函数
└── outputs/
    └── report.md             # LLM 生成的分析报告
```

## 运行方式

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 生成模拟数据

```bash
python generate_data.py
```

### 3. 配置 DeepSeek API Key（可选）

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 DEEPSEEK_API_KEY
```

### 4. 启动 Streamlit

```bash
streamlit run app.py
```

启动后访问 http://localhost:8501 即可看到分析页面。

## 数据说明

模拟数据集包含（时间范围 2024-01 ~ 2024-12）：

| 数据表          | 字段                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------ |
| orders.csv   | order_id, user_id, product_id, order_time, pay_time, order_amount, quantity, order_status, channel, province, city |
| users.csv    | user_id, register_time, gender, age, city_level, user_source                                                       |
| products.csv | product_id, product_name, category, sub_category, price                                                            |

数据特点：

- 8 个品类，118 个商品 SKU
- 5 个渠道：自然流量、抖音、小红书、百度、朋友推荐
- 6 月（618）和 11 月（双11）有明显 GMV 峰值
- 7-8 月人为设置 GMV 下滑，用于验证异常检测模块


