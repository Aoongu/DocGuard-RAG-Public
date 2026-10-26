# S7 Validation B2 有效复现实验

本目录保存 2026-10-19 的 `s7-validation-b2-reproduction-01-20261019`。它使用新目录与新 `run_id`，没有覆盖或恢复重跑先前无 API 的本地资产失败目录。该批次是唯一一次实际调用 DeepSeek API 的 Validation B2，不补跑任何失败题，也不根据结果修改 Prompt、配置或检索参数。

## 冻结结果

- 固定样本总数：60；成功 44，失败 16，完整 Validation 失败率为 16/60 = 26.67%。
- 可回答题：26 成功、4 失败；4 个失败题的 Dense Top-10 与黄金 chunk 交集均为 0。
- 边界题：18 成功、12 失败；成功边界题的 answer/citation rate 均为 1.0，不能解释为正确拒答。
- 成功可回答题 Citation Precision/Recall：0.618590/0.472685，分母 `n=26`。该指标只在成功可回答题上计算，存在生成成功选择偏差。
- 完整批次 token：input 206,334、cached input 8,320、output 11,336。
- 完整批次费用估算：0.036528660～0.073057320 美元；DeepSeek Responses 不返回实际账单价格，因此只报告冻结单价上下界。
- 端到端延迟 p50/p95：1422.019/2454.003 ms。
- 16 个失败全部是 API 返回后的 `GenerationValidationError: verified answer requires at least one citation`；失败响应的 usage/cost、Evidence、请求身份、时延与原文 hash 已保留，失败原文未落盘。

## 失败归因

- `s4-val-004/005/006/009`：主因 `retrieval_miss`，多标签同时包含 `generation_contract_failure`。
- 其余 12 个失败均为边界题，没有黄金 chunk，主因及唯一原因码为 `generation_contract_failure`。
- 多标签题数统计允许重复计入原因类别；详细逐题记录见 `failure_reviews.jsonl`。

Validation 结果不用于回调 Prompt 或配置，也不能解释为未见 Test 泛化。Test split 未读取、未运行，继续保持 `locked_until_s12`。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `run.json` | 1,283 | `78a54de09de976ad1b47c237dfae255089091c24d3ffacffa0169baf234da0b0` |
| `results.jsonl` | 1,426,831 | `0e150ff4b96a9e2999216b2a83f7f83d3a7ac57ee31b656008a03faf702bd62f` |
| `failures.jsonl` | 2,160 | `d94d374840bae880eb193b597313efe19cfa23f274fae373d77f58c2465ef0cd` |
| `summary.json` | 1,170 | `77e1e19d4c486836d2e204dcf98e4e14b7710e89748bd9e68256623c3bed6940` |
