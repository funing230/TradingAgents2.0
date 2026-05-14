# TradingAgents2.0 测试基线与回归说明

更新时间：`2026-05-13`
工作目录：`/home/sun/.openclaw/workspace/TradingAgents2.0`

## 一、目的

本文档记录当前仓库的一组**可复用测试基线**，用于回答两个问题：

1. 仓库现在是否处于“可继续联调/可继续开发”的健康状态
2. 未来改动后，如何快速判断是否把 overnight + TradingAgents2.0 集成链路打坏

---

## 二、当前基线结论

截至 `2026-05-13`，仓库已经达到以下状态：

- `overnight` 主 pipeline 可完整执行
- 全量 `pytest` 可通过
- 外部实时 provider 受限项已改为**合理 skip**，不会再把外部限频/权限问题误报成仓库代码失败

### 当前推荐结论

- **工程集成状态**：通过
- **overnight 主链路状态**：通过
- **provider 实时外部依赖状态**：部分受环境限制，但已隔离为 skip
- **正式策略验收状态**：仍需基于收益表现单独判断，不应等同于“测试通过”

---

## 三、推荐测试命令

### 1）全量测试

```bash
cd /home/sun/.openclaw/workspace/TradingAgents2.0
pytest -q
```

### 2）provider 定向测试

```bash
cd /home/sun/.openclaw/workspace/TradingAgents2.0
pytest -q tests/test_akshare_provider.py tests/test_tushare_provider.py
```

### 3）市场路由/配置定向测试

```bash
cd /home/sun/.openclaw/workspace/TradingAgents2.0
pytest -q tests/test_market_routing.py
```

### 4）overnight 主 pipeline 回归

```bash
cd /home/sun/.openclaw/workspace/TradingAgents2.0
python3 scripts/run_overnight_full_pipeline.py --start-date 2026-01-01 --end-date 2026-04-30 --top-n 5
```

---

## 四、当前实测结果基线

### A. 全量 pytest 基线

实测命令：

```bash
pytest -q
```

实测结果：

- `130 passed`
- `5 skipped`
- `27 subtests passed`
- `exit code 0`

### B. provider 定向测试基线

实测命令：

```bash
pytest -q tests/test_akshare_provider.py tests/test_tushare_provider.py
```

实测结果：

- `71 passed`
- `5 skipped`
- `exit code 0`

### C. market routing 定向测试基线

实测命令：

```bash
pytest -q tests/test_market_routing.py
```

实测结果：

- `25 passed`
- `exit code 0`

### D. overnight pipeline 基线

实测命令：

```bash
python3 scripts/run_overnight_full_pipeline.py --start-date 2026-01-01 --end-date 2026-04-30 --top-n 5
```

实测结果：

- 特征表生成成功
- 因子/过滤批量实验成功
- 成本批量回测成功
- manifest 写出成功
- 退出码：`0`

关键产物：

- `data/overnight_mvp/features/overnight_features_20260101_20260430.csv`
- `data/overnight_mvp/backtest_inputs/topn_baseline_input_20260101_20260430.csv`
- `data/overnight_mvp/experiments/top5_factor_filter_batch_20260101_20260430_top5_open.md`
- `data/overnight_mvp/backtest_results/batch/top5_batch_summary_20260101_20260430_top5.md`
- `data/overnight_mvp/pipeline_manifests/full_pipeline_20260101_20260430_top5.md`

---

## 五、为什么现在会有 skip

当前 skip 不是仓库本地逻辑失败，而是为了隔离**外部实时依赖的不稳定性**。

### 1）AkShare integration skip 条件
可能原因包括：

- 远端连接被关闭
- 网络瞬时波动
- 数据源反爬 / 限流
- 临时不可用

这类问题不应被解释为本地业务逻辑 regression。

### 2）Tushare integration skip 条件
可能原因包括：

- 接口限频
- 账号权限不足
- 网络瞬时故障

当前测试已区分：

- **rate limit**
- **permission denied**

并在真实集成测试中视作环境限制，而不是仓库代码失败。

---

## 六、这轮修复包含的关键调整

### 1）AkShare 路由测试修正
将原先“所有 method 都必须包含 `akshare`”的断言修正为：

- 市场数据接口必须包含 `akshare`
- overnight candidate 系列接口必须是 `local-only`

这与当前 interface 设计一致。

### 2）Tushare 异常语义修正
新增：

- `TusharePermissionError`

并保留其与 `TushareRateLimitError` 的兼容关系，以避免破坏原有 fallback 逻辑。

### 3）真实 provider 集成测试去假失败
对 AkShare / Tushare 的真实外部调用，在外部限制条件下改为：

- `skip`

从而避免把：

- 权限不足
- 限频
- 远端断连
- 网络瞬时错误

误报成代码失败。

### 4）pytest integration marker 注册
已在 `pyproject.toml` 中注册：

- `integration`

避免 `PytestUnknownMarkWarning`。

---

## 七、回归判据

未来每次改动后，可以用以下标准判断是否通过回归：

### 必须满足

1. `pytest -q` 退出码为 `0`
2. `scripts/run_overnight_full_pipeline.py` 在基线时间窗上退出码为 `0`
3. 关键产物路径仍能正常生成
4. 不出现新的未解释 fail

### 可以接受

1. 少量 provider integration 被 `skip`
2. 第三方库产生少量 `DeprecationWarning`
3. 外部实时接口偶发不可用，但被测试框架正确隔离

### 不可接受

1. overnight pipeline 再次出现主链路报错
2. `pytest -q` 出现新的稳定 fail
3. routing 设计与测试再次失配
4. provider 错误重新混淆成无法分类的异常

---

## 八、建议的日常使用顺序

如果只想快速验证改动是否安全，建议按下面顺序执行：

### 快速检查

```bash
pytest -q tests/test_market_routing.py
pytest -q tests/test_akshare_provider.py tests/test_tushare_provider.py
```

### 完整检查

```bash
pytest -q
python3 scripts/run_overnight_full_pipeline.py --start-date 2026-01-01 --end-date 2026-04-30 --top-n 5
```

这样可以同时覆盖：

- 路由正确性
- provider 行为
- overnight 主集成链路

---

## 九、当前结论

截至当前基线，最合适的说法是：

> `TradingAgents2.0` 与“一夜持股法”的工程集成已经打通，自动化测试已达到可继续联调与回归维护的状态；但策略收益表现与外部数据权限能力仍需单独评估，不应与“测试通过”混为一谈。
