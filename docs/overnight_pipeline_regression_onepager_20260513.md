# Overnight Pipeline 回归一页摘要

更新时间：`2026-05-13`
工作目录：`/home/sun/.openclaw/workspace/TradingAgents2.0`
时间窗：`2026-01-01 ~ 2026-04-30`
Top-N：`5`

---

## 1. 结论

本轮 `overnight` 主 pipeline **回归通过**：

- `scripts/run_overnight_full_pipeline.py` 成功执行
- 关键产物全部生成
- 历史 `KeyError: 'total_return'` 未复现
- 工程链路稳定可复跑

但从策略效果看，**当前 baseline / 因子变体仍整体为负收益**，因此结论应表述为：

> **工程集成稳定，但策略 alpha 目前未被验证成立。**

---

## 2. 本轮执行命令

```bash
cd /home/sun/.openclaw/workspace/TradingAgents2.0
python3 scripts/run_overnight_full_pipeline.py --start-date 2026-01-01 --end-date 2026-04-30 --top-n 5
```

执行结果：

- `exit code 0`

---

## 3. 产物检查

本轮成功生成/刷新：

- `data/overnight_mvp/features/overnight_features_20260101_20260430.csv`
- `data/overnight_mvp/backtest_inputs/topn_baseline_input_20260101_20260430.csv`
- `data/overnight_mvp/audit/overnight_feature_build_20260101_20260430.md`
- `data/overnight_mvp/experiments/top5_factor_filter_batch_20260101_20260430_top5_open.csv`
- `data/overnight_mvp/experiments/top5_factor_filter_batch_20260101_20260430_top5_open.md`
- `data/overnight_mvp/backtest_results/batch/top5_batch_summary_20260101_20260430_top5.csv`
- `data/overnight_mvp/backtest_results/batch/top5_batch_summary_20260101_20260430_top5.md`
- `data/overnight_mvp/pipeline_manifests/full_pipeline_20260101_20260430_top5.md`

---

## 4. 数据覆盖与样本规模

来自 feature build audit：

- Label rows used: `24375`
- Feature rows written: `24375`
- Top-5 rows written: `375`
- Trade dates covered: `2026-01-05 -> 2026-04-29`
- Symbols covered: `326`
- Selected days in Top-5: `75`

说明：

- 本轮不是空跑，样本和交易日覆盖是完整的
- 特征表与 Top-5 输入均已稳定生成

---

## 5. 基线成本回测摘要

最佳成本组合（按 total return / sharpe 都是同一组）：

- fee: `5 bps`
- slippage: `0 bps`
- ending_capital: `980,879`
- total_return: `-1.9121%`
- annualized_return: `-6.2810%`
- max_drawdown: `-9.8291%`
- sharpe: `-0.4303`
- calmar: `-0.6390`
- trade_days: `75`
- trade_count: `375`
- win_rate: `32.5333%`
- avg_trade_return_net: `-0.0224%`

解释：

- 成本越高，结果进一步恶化
- 当前 baseline 还没有表现出可接受的正向收益或风险调整后收益

---

## 6. 因子/过滤规则实验摘要

共跑了 `16` 个变体。

### 最佳 total return 变体
- variant: `baseline__strict_risk`
- total_return: `-3.9046%`
- sharpe: `-1.3948`
- max_drawdown: `-7.4959%`
- trade_days: `57`
- trade_count: `285`

### 最佳 sharpe 变体
- variant: `gap_aware__base`
- total_return: `-4.7409%`
- sharpe: `-0.9536`
- max_drawdown: `-10.1377%`
- trade_days: `75`
- trade_count: `375`

解释：

- 某些过滤规则能略微改善部分风险指标
- 但所有主要变体仍然是负收益
- 现阶段更像是在“找相对没那么差的版本”，还不是“找到可用策略”

---

## 7. 上游依赖与当前限制

audit 中仍然记录了 1 个上游限制：

- `permission-safe mode: skipped daily_basic / moneyflow / stk_factor fetches under current token limits; using label-derived features + cached stock_basic only`

这意味着：

- pipeline 目前能跑通
- 但特征构建处于**降级模式**
- 当前结果尚不能代表“完整 live 特征集”下的最终策略表现

---

## 8. 本轮最重要判断

### 工程层
- **通过**
- 一键入口可复跑
- 历史报错点未复现
- 产物链完整

### 研究层
- **未通过策略验收**
- baseline 与变体总体仍为负收益
- 暂时不能得出“策略有效”的结论

### 数据依赖层
- **部分受限**
- Tushare 权限/限频仍影响 live 增强特征
- 当前更接近“可运行 MVP + 降级特征模式”

---

## 9. 下一步建议

优先级建议如下：

1. **补齐/恢复 live 特征能力**
   - 优先恢复 `daily_basic` / `moneyflow` / `stk_factor`
   - 否则当前因子实验上限偏低

2. **做更明确的对照组**
   - 随机 Top-N
   - 昨日涨跌幅排序
   - 仅资金流排序
   - 当前规则打分

3. **把结果导向“问题定位”而不是继续盲扫参数**
   - 是候选打分有问题
   - 是标签噪声太大
   - 还是交易成本完全吃掉 edge

4. **固定这组时间窗做长期回归基线**
   - `2026-01-01 ~ 2026-04-30`
   - `top_n=5`
   - 后续每次改动都复跑这一组

---

## 10. 一句话结论

> 这轮 overnight pipeline 回归从工程角度是成功的；但从策略角度，当前结果仍偏负，说明“结合已打通”不等于“策略已有效”。
