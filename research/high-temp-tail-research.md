# Polymarket 高温尾部套利 — 研究简报

## 结论
高温尾部（"≥X°C"开口桶）套利逻辑成立且有真实盈利案例，但不是"无脑捡钱"：学术证据显示偏差是**时变**的，近端（1-12h）是热门桶被高估、买 NO 赚钱；Hans323 的买 YES 尾部大赚依赖的是**模型 vs 市场的信息差**，不是统计套利。pm-bot 的 EMOS 校准路线是对的——这是公开 repo 里最少人做的差异化。

## 尾部低估的证据
- **Hans323 实战**：2026年2月伦敦高温市场，8¢ 建仓 $92,632，模型概率显著高于 8%，温度落入区间，$1.11M 出场（12x）。模式固定：买 2-8¢、市场隐含概率远低于模型概率的尾部。来源：[predictionmarketspicks playbook](https://predictionmarketspicks.com/downloads/polymarket-whale-playbooks.pdf)、[vegaforge](https://vegaforge.dev/blog/weather-bots-polymarket)
- **学术证据（arXiv，十城每日最高温市场）**：近端 1-12h 窗口热门桶系统性高估，买 NO 策略 $1,840 本金赚 +$1,011（+55% pre-fee，20.7% 胜率）；30-48h 窗口买 NO 微亏（-0.13%）。即偏差存在但随期限衰减，远期尾部不要碰。来源：[arxiv.org/pdf/2610.13593](https://arxiv.org/pdf/2610.13593)
- **机理**：散户凭体感/消费级预报定价，415+ 活跃天气市场、$4.6M 量，NOAA 预报 1-2 天准确率 85-90% 而市场常只给 15-40% 隐含概率——这个 gap 就是 edge。来源：[tradetheoutcome](https://www.tradetheoutcome.com/polymarket-weather-strategy/)（建议 edge 阈值 ≥8-10%，覆盖 2-3% 手续费）
- **极端案例**：0.1¢ 桶曾打出 49,000% 回报（Lucknow/Atlanta）；香港 29°C 市场 NO 端 1.3¢ → 77x。来源：[cryptonews](https://cryptonews.net/news/finance/32768888/)、[ainvest](https://www.ainvest.com/news/polymarket-bet-pays-77-1-hong-kong-doesn-hit-29degc-today-weather-models-won-2606/)

## 别人用的 edge 来源（GitHub）
- **预报数据**：Open-Meteo（聚合 GFS/ECMWF/HRRR，免费）是主流；进阶用 ECMWF IFS 51 成员集合（[yannbellec](https://github.com/yannbellec/polymarket-weather-bot)，走 Copernicus CDS）；NWS/NOAA + METAR 实测（[MihirM9](https://github.com/MihirM9/polymarket-weather-bot)，专做美国城市**每日最高温桶市场**，Gaussian CDF 算桶概率 + Kelly + 13 层风控，自带 backtesting/replay 引擎）
- **校准**：pm-bot 的 EMOS 是差异化点，公开 repo 里只有 [kevinbadi/polymarket-weather-quant](https://github.com/kevinbadi/polymarket-weather-quant) 标榜 "calibrated"，其他多用原始集合离散度——集合预报天生 underdispersed，尾部概率被系统性压低，不校准就吃不到真 tail edge
- **执行**：2 分钟轮询是默认 cadence；预报每 6 小时更新一次，市场滞后几秒到几分钟——"forecast latency" 是量化玩家的游乐场（[predictmarketcap](https://predictmarketcap.com/analysis/weather-betting-boom)）
- **策略形态**：Hans323（重注单点 2-8¢）/ neobrother（ladder 全覆盖 5 桶×3¢，中一个就 560%）/ gopfan2（YES<15¢、NO>45¢ 系统化刷量）/ meropi（数百个 $1-3 微注）

## 已知的坑（按致命程度）
1. **结算数据源**：不按"官方最高温"结算，按**结算页面上显示的最大值**。WU 时代用十分位规则（99.7% 吻合），2026年8月后切 weather.gov 小时表。整度 vs 十分位能差出 1°F，桶才 2°F 宽——定价模型必须用对规则。来源：[Medium 实测](https://medium.com/@mkhlndrv/i-tried-to-beat-polymarkets-weather-markets-with-a-forecast-model-the-market-won-f32c2d6f0f1c)
2. **传感器可被操纵**：2026年4月巴黎戴高乐机场站被人为加热 spikes（疑似吹风机），$119 → $21,000，Météo-France 已刑事报案，Polymarket 被迫换结算站到 Le Bourget。**只做大城市主流站，避开小站**。来源：[cryptonews](https://cryptonews.net/news/finance/32768888/)
3. **机场 vs 市区**：KLGA vs 曼哈顿能差出整度，pm-bot 已有 station bias 校正——这是对的，保留
4. **流动性**：尾部桶盘口薄，Hans323 的 $92k 是例外；小资金用 ladder/微注，大资金会滑点吃掉 edge
5. **远期市场常不存在**：想做的 "2 天后" 市场三个月里大部分时间根本没开——策略只能做近端，这也正好是 edge 最强的窗口
6. **季节**：秋天冷暖空气拉锯，有团队回测秋季准确率仅 63.7%，接近抛硬币

## 对 pm-bot tail 策略的直接建议
- 先修 parent 已发现的 bug：`_tail_strategy` 收不到真 tail 桶（主循环已过滤）
- 配置对标公开最优实践：`tail_min_edge` 8% 与 tradetheoutcome 的阈值一致，ok；`tail_max_price` 10¢ 略保守（Hans323 做到 2-8¢，gopfan2 到 15¢），可放宽到 15¢ 做 A/B
- 加一条：只做 `is_high_tail`，`is_low_tail` 另测——高温尾和低温尾的偏差机制不同（高温多为热浪事件驱动，模型更新更快）
- 回测用 MihirM9 式的 replay 框架思路：历史预报 + 历史盘口，而不要只用收盘价