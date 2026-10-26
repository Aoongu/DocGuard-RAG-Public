# S7 Development B2 原始运行说明

本目录保存 2026-10-18 对冻结 `s4-golden-v1` Development split 执行的唯一一次完整 DeepSeek B2 批次。运行固定为 Dense Top-10 + `deepseek-flash`，未启用 BM25、RRF、Cross-Encoder 或 S8 校准拒答；没有执行 Validation 或 Test，也没有重跑失败题。

## 冻结结果

- 输入题数：60，题号唯一且与 Development 数据集顺序一致。
- 成功：49。
- 失败：11，均为生成后本地 `GenerationValidationError: verified answer requires at least one citation`。
- 失败构成：2 条可回答题、9 条边界题。
- 成功项记录的 token：输入 155,286、缓存输入 13,312、输出 9,260。
- 成功项记录的费用区间：0.026892036～0.053784072 美元；这是 49 条成功响应的记录值，不是 60 次请求的完整账单。
- 成功项端到端延迟 p50/p95：1473.260/2452.753 ms。
- 可回答成功项 Citation Precision/Recall：0.363095/0.343367，`n=28`。
- 边界成功项 answer/citation rate：1.0/1.0，`n=21`。这只是 S7 无校准拒答基线的风险观察，不能解释为正确拒答。

失败题号：

`s4-dev-003`、`s4-dev-028`、`s4-dev-032`、`s4-dev-034`、`s4-dev-035`、`s4-dev-036`、`s4-dev-037`、`s4-dev-038`、`s4-dev-051`、`s4-dev-052`、`s4-dev-053`。

## 已知审计缺口

这 11 个失败发生在 API 已返回之后，但当时的运行器把失败行的 token、费用、请求身份、Evidence 和原始响应写成空值。因此 `summary.json` 的 usage/cost 只覆盖 49 个成功项；现有文件不能恢复 11 次失败调用的精确 token 或费用，不能把该汇总称为完整账单。

该缺口在批次完成后通过离线 TDD 修复：后续生成后校验失败会保存 Evidence、请求/模型身份、真实 usage/cost 和完整分阶段时延；失败原文不落盘，只保存原始 UTF-8 内容的 SHA-256、字符数和字节数。修复没有回写本目录，也没有触发模型调用。

`failure_reviews.jsonl` 是批次完成后的派生复核记录，不修改原始运行文件。它按冻结的多标签协议保存全部 11 个失败项；其中 `s4-dev-003` 的主因是 `retrieval_miss`，`s4-dev-028` 的主因是经协议所有者确认的 `dataset_question_ambiguity`。两题同时保留观察到的 `generation_contract_failure`，不把表面合同错误冒充唯一根因。

## 文件校验

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `run.json` | 1,284 | `842597b0b9ef8d9380e2a569fc74556cc3bba823bf30c16176ccc6d0d3d386b9` |
| `results.jsonl` | 1,141,177 | `54387a65b90c6fa720900e5bb465ca5c5f6406e8fe6795f4956d3c2824300cc5` |
| `failures.jsonl` | 1,485 | `60d98c8af44e38cd6b1b1da98b40b875f74ad934e97683c9161e64480900cc03` |
| `summary.json` | 1,198 | `3f8aeac461b5e8143ef63f0702a9a964ca14200ec087317c28d238c726dbc5d1` |

Test split 继续保持 `locked_until_s12`。本目录中的任何成功行或失败行都不得重跑；中断恢复只允许补齐从未产生记录的题号。复现实验必须创建新的目录和 `run_id`。
