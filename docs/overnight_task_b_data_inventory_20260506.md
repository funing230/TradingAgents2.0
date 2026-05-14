# TradingAgents2.0 一夜持股法数据资产清单（Task B）

更新时间：`2026-05-06`
工作目录：`/home/sun/.openclaw/workspace/TradingAgents2.0`

## 一、当前结论

目前 `TradingAgents2.0` 已经具备一夜持股法的 **历史标签资产**、**复权缓存资产**、**股票池资产** 和 **可工作的 Tushare 特殊通道方案**。  
但要进入稳定的 repo 内日更，还存在一个明确 blocker：

- `TradingAgents2.0/.env` 中当前 `TUSHARE_TOKEN` 走 `TUSHARE_API_URL=http://121.40.135.59:8010/` 时，仓库原生 provider 返回：`token invalid`
- 用户现场给出的示例代码（显式传入另一枚 token）可成功拉取真实数据

因此当前状态应区分为：

- **历史研究 / baseline 数据：可用**
- **特殊通道本身：可用**
- **仓库当前默认 live 配置：未完全打通（token 需更新）**

---

## 二、已经存在的本地数据资产

### 1）隔夜标签主数据
路径：
- `data/overnight_labels/csi300_overnight_labels_20240101_20260430.csv`
- `data/overnight_labels/csi300_overnight_labels_clean_20240101_20260430.csv`
- `data/overnight_labels/csi300_overnight_labels_20250725_20250731.csv`

用途：
- 直接作为 `T close -> T+1 open` 监督标签
- 支撑 baseline 排序、统计分析、规则回测、模型训练样本构造

关键字段（来自 `csi300_overnight_labels_clean_20240101_20260430.csv` 表头）：
- `ts_code`
- `trade_date`
- `close`
- `next_trade_date`
- `next_open`
- `overnight_return_open`
- `gap_days`
- `source`
- `is_valid`
- `market_board`
- `limit_threshold`
- `is_extreme`
- `is_soft_outlier`
- `is_limit_move_like`
- `is_missing_or_invalid`
- `is_long_gap`
- `outlier_reason`
- `is_trainable`

样本规模：
- 总行数：`183256`
- 股票数：`328`
- 起始交易日：`2024-01-02`
- 截止交易日：`2026-04-29`
- `is_trainable=True`：`181483`

可用于一夜持股法的程度：**极高**

---

### 2）清洗审计与异常样本审计
路径：
- `data/overnight_labels/csi300_overnight_clean_audit_20240101_20260430.md`
- `data/overnight_labels/csi300_overnight_clean_audit_20240101_20260430.json`
- `data/overnight_labels/csi300_overnight_outlier_review_20240101_20260430.csv`
- `data/overnight_labels/csi300_overnight_labels_audit_20240101_20260430.md`
- `data/overnight_labels/csi300_overnight_labels_audit_20240101_20260430.json`

用途：
- 明确哪些样本是长假 gap、涨跌停类、极端跳空、软异常值
- 给训练集 / 回测集过滤规则提供直接依据
- 提供 reviewer 视角下的可解释审计链

已知审计摘要：
- `Total rows: 183256`
- `Symbols: 328`
- `Trainable rows: 181483`
- `Excluded rows: 1773`
- `Hard extreme rows: 25`
- `Limit-move-like rows: 324`
- `Soft outlier rows: 641`
- `Long gap rows: 1658`

可用于一夜持股法的程度：**极高**

---

### 3）股票池 / Universe 数据
路径：
- `data/overnight_labels/csi300_universe_20240101_20260430.csv`
- `data/overnight_labels/csi300_universe_20250725_20250731.csv`

来源逻辑：
- 优先：`index_weight(index_code='000300.SH', ...)`
- 回退：本地 `LOCAL_CS300_HISTORY`
- 再回退：`stock_basic` top-N proxy

用途：
- 定义可交易股票池
- 避免把全市场噪声直接混进策略
- 支撑动态 CSI300 成分股研究

可用于一夜持股法的程度：**高**

---

### 4）复权因子缓存
路径：
- `data/tushare_adj_factor_cache/*.csv`

用途：
- 长区间价格复权
- 避免分红送转导致历史隔夜收益不可比
- 为 forward / backward adjustment 提供支持

可用于一夜持股法的程度：**高**

---

### 5）真实 smoke / demo 结果
路径：
- `data/overnight_labels/smoke_real.csv`
- `data/overnight_labels/overnight_labels_demo_live_20260506.csv`

说明：
- `smoke_real.csv` 说明过去曾成功通过 `tushare.daily+trade_cal` 获取真实样例
- `overnight_labels_demo_live_20260506.csv` 记录过一次失败尝试，错误与 token 路由有关

可用于一夜持股法的程度：**中**（更偏验证链路，不是主训练资产）

---

## 三、当前已验证可 live 获取的 Tushare 数据

已用特殊通道：
- `http://121.40.135.59:8010/`

已验证成功的接口：
- `index_basic`
- `trade_cal`
- `daily`
- `adj_factor`
- `stock_basic`
- `daily_basic`
- `moneyflow`
- `top10_holders`
- `index_weight`
- `stk_factor`
- `news`

其中对一夜持股法最重要的接口分层如下。

### A. 核心必需
- `trade_cal`
- `daily`
- `adj_factor`
- `stock_basic`
- `index_weight`

### B. 强增强
- `daily_basic`
- `moneyflow`
- `stk_factor`

### C. 条件增强
- `news`
- `top10_holders`

---

## 四、哪些数据可以被一夜持股法使用

### 1）标签层
可直接使用：
- `overnight_return_open`
- `next_open`
- `close`
- `gap_days`
- `is_trainable`
- `outlier_reason`

用途：
- 训练标签
- 回测目标收益
- 样本过滤

### 2）股票池层
可直接使用：
- `index_weight` 构造 `CSI300` 动态成分股
- `stock_basic` 过滤上市日期过短、行业、市场板块

### 3）价格与流动性层
可直接使用：
- `open/high/low/close`
- `pct_chg`
- `vol`
- `amount`
- `turnover_rate`
- `volume_ratio`
- `circ_mv`
- `total_mv`

### 4）资金流层
可直接使用：
- `buy_elg_amount`
- `buy_lg_amount`
- `buy_md_amount`
- `buy_sm_amount`
- `net_mf_amount`

### 5）技术指标层
可直接使用：
- `macd_dif`
- `macd_dea`
- `macd`
- `kdj_k`
- `kdj_d`
- `kdj_j`
- `rsi_6`
- `rsi_12`
- `rsi_24`
- `boll_upper`
- `boll_mid`
- `boll_lower`
- `cci`

### 6）事件/消息层
可条件使用：
- `news.title`
- `news.content`
- `news.datetime`

---

## 五、Task B 输出结论

### 已有数据已经足够支撑：
- 一夜持股法 baseline 研究
- 候选打分 / 排序式策略
- 训练样本构造
- 简单回测与结果审计

### 仍然缺少但不阻止 MVP 启动的：
- 分钟级尾盘数据
- 集合竞价数据
- 更精细的公告/事件时间戳
- 开盘后 09:35 / 09:45 / 10:00 的多退出点数据

### 当前唯一明确 blocker：
- 仓库 `.env` 当前 `TUSHARE_TOKEN` 无法在 repo 内通过 `http://121.40.135.59:8010/` 认证
- 需要把仓库 token 更新为当前可用的 token，或在安全前提下改为从外部安全注入
