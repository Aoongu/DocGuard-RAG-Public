# S7 Development Judge 对照

本目录冻结 2026-10-19 对 Development 固定人工样本中 15 道可评分题的
Judge 结果。`s4-dev-032` 为生成失败且不可评分，没有送入 Judge。

## 冻结身份与执行边界

- run ID：`s7-judge-development-deepseek-flash-20261019`
- Judge：`deepseek-flash`，`s7-judge-v1` 独立 Prompt，严格 JSON Schema
- Judge 与答案生成器使用同一模型别名，只能视为独立 Prompt 对照，不能称为独立模型验证
- 自动重试：0；已有成功行和失败行均不可再次调用
- 结果：15 success、0 failed；Test 未读取或执行
- `results.jsonl` SHA-256：`32a2eeabe131e32ed8536c7d083fdb599a14b450cc619bb16cece7b8a3aa1380`

## 用量与成本

- input/cached/output token：25,954 / 0 / 1,565
- 冻结价格估算下界/上界：0.00483210 / 0.00966420 美元
- 端到端 p50/p95：1071.727 / 1369.629 ms

## 人工一致性

人工样本为 15 道可评分题；Faithfulness 均值 1.9333，Answer Relevance
均值 1.4000。Judge 对两项指标的 15 道题全部打 2 分：

| 指标 | 准确率 | 加权一致度 | Cohen's kappa |
| --- | ---: | ---: | ---: |
| Faithfulness | 0.9333 | 0.9667 | 0.0000 |
| Answer Relevance | 0.6000 | 0.7000 | 0.0000 |

kappa 为 0 的直接原因是 Judge 输出没有方差。Answer Relevance 的 6 个不一致
全部为 Judge 高估，主要是把诚实但没有回答问题的弃答也评为 2 分。因此本次 Judge
只能用作失败诊断，不能替代人工评分，也不能据此声称自动评测器可靠。

`run.json` 保存 Judge 身份和输入哈希，`results.jsonl` 保存逐题原始判定、usage、
成本和时延，`agreement.json` 保存混淆矩阵与不一致清单，`summary.json` 保存完整汇总。
