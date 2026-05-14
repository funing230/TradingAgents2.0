# TradingAgents2.0 一夜持股法 MVP 蓝图（Task C）

更新时间：`2026-05-06`
工作目录：`/home/sun/.openclaw/workspace/TradingAgents2.0`

## 一、MVP 目标

构建一版 **可重复、可审计、可持续日更** 的一夜持股法 baseline：

- 股票池：`CSI300`
- 买入时点：`T 日收盘`
- 卖出时点：`T+1 日开盘`
- 标签：`overnight_return_open`
- 输出：每日候选列表 + 历史回测摘要 + 特征表 + 清洗审计

MVP 不追求一开始就接入复杂 agent，而是先把 **数据链 → 特征链 → 标签链 → 排序链 → 回测链** 跑通。

---

## 二、MVP 应使用的数据接口

## 1）Universe
- `index_weight(index_code='000300.SH')`
- `stock_basic(...)`

用途：
- 构建动态 `CSI300` 股票池
- 过滤新股 / 行业 / 市场板块

## 2）标签与价格
- `trade_cal(...)`
- `daily(...)`
- `adj_factor(...)`

用途：
- 构建 `T close -> T+1 open`
- 统一交易日
- 做复权价格处理

## 3）特征
- `daily_basic(...)`
- `moneyflow(...)`
- `stk_factor(...)`

用途：
- 生成流动性、估值轻过滤、资金流、技术指标等特征

## 4）事件过滤
- `news(...)`

用途：
- 做高风险消息过滤或事件增强实验

---

## 三、建议的数据表设计

### 1）Universe 表
建议路径：
- `data/overnight_mvp/universe/csi300_universe_daily.csv`

建议字段：
- `trade_date`
- `ts_code`
- `weight`
- `industry`
- `market`
- `list_date`
- `is_new_listing`

---

### 2）Raw price / label 表
建议路径：
- `data/overnight_mvp/labels/overnight_labels_daily.csv`

建议字段：
- `ts_code`
- `trade_date`
- `close`
- `next_trade_date`
- `next_open`
- `overnight_return_open`
- `gap_days`
- `adj_factor`
- `source`
- `is_trainable`
- `outlier_reason`

说明：
- 初版可直接复用现有 `data/overnight_labels/csi300_overnight_labels_clean_20240101_20260430.csv`

---

### 3）Feature 表
建议路径：
- `data/overnight_mvp/features/overnight_features_daily.csv`

建议字段：

#### 价格行为
- `ret_1d`
- `ret_3d`
- `ret_5d`
- `ret_10d`
- `amplitude_1d`
- `close_to_high`
- `close_to_low`
- `close_vs_open`

#### 流动性 / 规模
- `vol`
- `amount`
- `turnover_rate`
- `volume_ratio`
- `circ_mv`
- `total_mv`

#### 资金流
- `net_mf_amount`
- `buy_elg_amount`
- `buy_lg_amount`
- `buy_md_amount`
- `buy_sm_amount`
- `mf_large_minus_small`

#### 技术指标
- `macd`
- `macd_dif`
- `macd_dea`
- `kdj_k`
- `kdj_d`
- `kdj_j`
- `rsi_6`
- `rsi_12`
- `rsi_24`
- `boll_pos`
- `cci`

#### 风险 / 过滤标签
- `is_long_gap`
- `is_limit_move_like`
- `is_soft_outlier`
- `is_extreme`
- `has_major_news`

---

## 四、MVP 候选打分逻辑

初版不必上复杂模型，先做一个 **可解释打分器**。

### 方案 A：规则加权得分（建议先做）

示例：
- 正向：
  - `net_mf_amount` 高
  - `buy_elg_amount` / `buy_lg_amount` 偏强
  - `turnover_rate` 适中偏高
  - `volume_ratio` 放大但不过热
  - `close_to_high` 较高
  - `rsi_6 / rsi_12` 不过热
  - `macd` / `macd_dif` 改善
- 负向：
  - `is_long_gap=True`
  - `is_limit_move_like=True`
  - `is_extreme=True`
  - `volume_ratio` 过高且 `rsi_6` 过热
  - `重大负面 news`

输出：
- `overnight_score`
- `rank_in_day`

### 方案 B：轻量模型排序（第二步）
可选：
- Logistic regression
- LightGBM ranker / regressor
- XGBoost regressor

目标：
- 预测 `overnight_return_open > 0`
- 或直接预测 `overnight_return_open`

---

## 五、MVP 回测协议

### 回测定义
- 调仓频率：每日
- 买入：`T close`
- 卖出：`T+1 open`
- 股票池：当日 `CSI300`
- 持仓数：`Top N`，建议 `N in {5, 10, 20}`
- 权重：等权

### 成本假设
- 手续费：双边显式加入
- 滑点：按 `bps` 固定假设加入
- 不可交易行：剔除或记为 failed execution（先显式记录）

### 需要输出的核心指标
- 年化收益
- 累计收益
- 平均单笔隔夜收益
- 胜率
- 最大回撤
- Sharpe / Calmar
- 日均持仓数
- 月度收益分布
- 分行业表现
- 高 gap / 长假样本敏感性

### 需要做的最小对照组
1. `CSI300` 等权随机选股
2. `按昨日涨跌幅排序`
3. `按资金流排序`
4. `规则加权得分`

---

## 六、推荐的执行顺序

### Phase 1：固定 live 数据链
1. 修复 repo 内有效 `TUSHARE_TOKEN`
2. 保留 `TUSHARE_API_URL=http://121.40.135.59:8010/`
3. 做最小连通性自检：
   - `trade_cal`
   - `daily`
   - `adj_factor`
   - `index_weight`
   - `moneyflow`
   - `stk_factor`

### Phase 2：构建统一特征表
1. 以 `ts_code + trade_date` 为主键
2. 拼接：
   - universe
   - daily
   - adj_factor
   - daily_basic
   - moneyflow
   - stk_factor
   - clean labels
3. 生成单一 `overnight_features_daily.csv`

### Phase 3：先跑规则版候选打分
1. 输出每日 `Top N`
2. 生成最简单回测结果
3. 校验是否优于随机 / 简单动量 baseline

### Phase 4：再上轻量模型
1. 时间切分训练 / 验证 / 测试
2. 做滚动窗口回测
3. 看真实增益是否稳定

---

## 七、当前 blocker 与建议

### blocker
仓库内当前 `TradingAgents2.0/.env` 的 `TUSHARE_TOKEN` 虽然已配置，且 `TUSHARE_API_URL` 已补到：
- `http://121.40.135.59:8010/`

但 repo 原生 provider 实测仍返回：
- `token invalid`

### 建议
在不把明文 token 写进文档的前提下，下一步应完成其一：

1. 更新 `.env` 中当前失效 token 为新的有效 token
2. 改为通过外部安全注入环境变量启动脚本
3. 给 provider 增加第二优先级环境变量，例如：
   - `TRADINGAGENTS_TUSHARE_TOKEN`

---

## 八、Task C 输出结论

当前最合理的 MVP 方案不是继续扩大战线，而是：

1. **先修 repo 内 token**
2. **复用现有 clean labels 作为训练 / 回测主标签**
3. **把 `daily + daily_basic + moneyflow + stk_factor` 拼成统一特征表**
4. **先做规则加权版 Top-N 一夜持股 baseline**

只要第 1 步完成，后面 2-4 步就可以连续推进。