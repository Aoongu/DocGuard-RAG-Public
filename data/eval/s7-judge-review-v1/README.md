# S7 Judge 人工复核

本目录保存 S7 Development 固定分层样本的人工复核记录。模板由程序按
`(category, answerable)` 分层、每层按 `question_id` 取前 2 题生成，并记录模板
SHA-256。模板必须同时读取冻结 Development 生成结果，不能在生成失败后换题。
人工必须独立填写 0～2 的 Faithfulness 与 Answer Relevance 分数及说明；
禁止使用生成模型代填并将其声明为人工标签。

可评分题必须有可见答案，两项分数只能取 0、1、2，且不得填写生成失败原因。
不可评分题的两项分数必须为 `null`，至少填写一个冻结失败原因码；多标签允许，
但 `primary_failure_reason` 必须属于原因列表。失败原因只能使用冻结协议中的
`retrieval_miss`、`dataset_question_ambiguity`、`generation_contract_failure`、
`upstream_failure` 或带说明的 `unknown`。

在所有固定题目完成填写前，汇总状态必须保持 `human_validation_status=incomplete`，
Judge 指标必须标记为 `unvalidated`。Test 数据在 S12 前不得进入本工作流。
