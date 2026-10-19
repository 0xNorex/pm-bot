# S1 策略：NoFade — 完整说明

## 一句话
买被市场高估的 YES 桶对应的 NO。市场系统性把 YES 定价高 11-13pp，我们做空这个偏差。

## 原理

### 核心发现（纯市场校准，无模型、无前视）
| YES 价格 | 平均定价 | 实际胜率 | 高估 |
|---|---|---|---|
| 0.05-0.15 | 0.084 | 0.056 | +2.8pp |
| 0.15-0.25 | 0.189 | 0.067 | **+12.2pp** |
| 0.25-0.40 | 0.311 | 0.182 | **+12.9pp** |
| 0.40-0.60 | 0.464 | 0.350 | **+11.4pp** |

市场在 0.15-0.60 区间系统性高估 YES 11-13pp。这是行为偏差（favorite-longshot bias），不是模型优势。

### 为什么用模型？
模型（GFS ensemble + 经验误差）用来**定位**哪些桶被高估：
- 信号条件：`P_market(YES) - P_model(YES) > 0.12`
- 模型不需要完美，只需要比市场准。市场高估是独立存在的。

### 为什么买 NO 而不是卖 YES？
Polymarket 没有做空机制。买 NO = 做空 YES。NO 价格 = 1 - YES 价格。

## 执行细节

### 信号生成（每日 08:00 CST）
1. **找市场**：5 城（NY/Chicago/London/Dallas/SF）当天的最高温事件
2. **拿预报**：Open-Meteo GFS ensemble，取当日最高温的 30 个成员
3. **算模型概率**：对每个桶，用 `N(fcst_mean - 0.54, 0.90)` 算落入区间的概率
   - bias -0.54°C, sigma 0.90°C（150 样本实测）
4. **比市场价**：`edge = YES_market - P_model`
5. **过滤**：`edge > 0.12` 才触发
6. **拿可成交价**：CLOB `/price` 接口查 NO token 的真实买入价（不是显示价）
7. **记录价差**：`gap = executable - displayed`，监控执行成本

### 下单
- **方向**：买 NO（对应 YES 被高估的桶）
- **金额**：$20/笔（固定，不加仓不减仓）
- **时机**：信号出来就下（T+0，决议前 ~24h）
- **一个事件可多笔**：平均 1.8 笔/事件，最多 4 笔。多笔天然对冲（19 个多信号事件中 0 全败）

### 持有到决议
- 不止损、不止盈，持有到市场决议
- NO 赢：每个 share 得 $1
- NO 输：亏掉本金

### 结算
- 按 Polymarket 官方决议（outcomePrices）
- 用页面显示值（0.1°F 四舍五入），不是真实最高温
- **结算源（weather.gov NWS）**：
  | 城市 | 站点 | NWS 链接 |
  |---|---|---|
  | New York | KLGA (LaGuardia) | weather.gov/wrh/timeseries?site=klga |
  | Chicago | KORD (O'Hare) | weather.gov/wrh/timeseries?site=kord |
  | London | EGLC (London City) | weather.gov/wrh/timeseries?site=eglc |
  | Dallas | KDAL (Love Field) | weather.gov/wrh/timeseries?site=kdal |
  | San Francisco | KSFO (SFO Intl) | weather.gov/wrh/timeseries?site=ksfo |

## 参数
| 参数 | 值 | 说明 |
|---|---|---|
| edge 阈值 | 0.12 | walk-forward 训练集最优 |
| 交易窗口 | 24h | 6h 窗口样本外衰减 |
| 每笔金额 | $20 | 低于中位数成交（$25） |
| 城市 | 5 | NY/Chicago/London/Dallas/SF |
| 滑点假设 | 2.5% | $20 对应的保守估计 |

## 回测成绩（30 天，66 笔）
- 总 P&L：+$193（$20/笔）
- 胜率：81.8%
- 夏普（年化）：9.5
- 最大回撤：-$42
- 日均：+$8.0

## 风险
1. **滑点**：真实滑点未知，CLOB 报价接口不建模 size impact。用 $20 起步实测。
2. **预报 lead time**：previous-runs 的 lead time 存疑，P_model 可能偏准。实盘信号会比回测少，但方向正确。
3. **市场修正**：如果市场参与者发现这个偏差，edge 会缩小。Andreev 的 2-day edge 就消失了。
4. **黑天鹅**：极端天气事件可能导致模型完全失效。单笔 $20 限制损失。

## Paper 交易状态
- 本金：$1000（模拟）
- 每笔：$20
- 2026-10-19：8 笔已建仓，$160 投入，$840 现金
- 每日 08:00 自动跑信号
