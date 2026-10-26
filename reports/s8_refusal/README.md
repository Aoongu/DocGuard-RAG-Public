# S8 版本感知拒答校准结果

本目录保存 S8 透明规则的正式 Validation 选参与 Development 诊断复放。两次运行均只读取已有 S6 M2 结果和冻结 S2 chunks，没有重新检索、调用 LLM 或读取 Test。

## Validation 选参

- 目录：`validation_selected/`
- 输入：60 条冻结 Validation、S6 M2 rerank depth 50 Top-10。
- 候选：预注册的 1,680 组阈值，全部保存在 `selection.json`。
- 选择：`top1=0`、`top3_mean=0`、`coverage=0.75`、`margin=0.5`。
- 拒答 Precision/Recall/F1：0.7073 / 0.9667 / 0.8169。
- 四动作 macro-F1：0.3767。
- 错误接受 / 错误拒答：1 / 12。
- 可回答题覆盖率：0.60。
- ROC-AUC：0.7183。

这是 `selection_result`：同一 Validation 同时用于选择阈值和报告结果，不能作为未见数据泛化证据。结果冻结后不根据失败题修改规则或补跑。

## Development 诊断

- 目录：`development_diagnostic/`
- 输入：60 条冻结 Development、既有 S6 M2 rerank depth 20 Top-10。
- 阈值：直接读取 Validation `selection.json`，不重新搜索。
- 拒答 Precision/Recall/F1：0.7353 / 0.8333 / 0.7812。
- 四动作 macro-F1：0.5000。
- 错误接受 / 错误拒答：5 / 9。
- 可回答题覆盖率：0.70。
- ROC-AUC：0.9256。

Development 与 Validation 的上游 rerank depth 不同，因此这里只用于诊断透明规则在既有冻结结果上的行为，不将两个 split 的数值差异解释为模型泛化变化。

## 主要失败边界

- Validation 有 12 道可回答题被拒答或要求澄清，说明当前透明规则以覆盖率换取较高拒答召回。
- Validation 的版本歧义题中，仅一部分被识别为澄清；双版本同源冲突规则保持保守，不用数据集类别直接触发。
- 显式双版本比较若 Top-10 未同时提供两侧版本证据，会进入保守路径。这一行为已随正式结果冻结，本阶段不回看失败后修改。
- Test 继续锁定到 S12；只有 Test 能提供未见数据结论。

## 文件说明

- `run.json`：不可变运行身份与 Test 锁。
- `results.jsonl`：逐题特征、动作、原因码和触发规则。
- `summary.json`：完整指标、混淆矩阵和解释限制。
- `selection.json`：Validation 全部候选及最终阈值。
- `risk_coverage.jsonl`：逐覆盖率 selective risk。
- `failures.jsonl`：运行失败；本次两个目录均为空。
