# S4 冻结集 Dense Smoke 回执

本目录由 `docguard.evaluation.cli` 在提交 `05a1ae4b18b60ab1074d1f42cd9520023951209e` 上，对冻结后的 30 条 Development Smoke 生成：15 条可回答、15 条边界题，30 条均成功。`run.json` 固定数据、配置、语料、模型 revision 和参数，`results.jsonl` 保存逐题事实，`summary.json` 只从逐题结果派生。

冻结 Smoke SHA-256 为 `f4e2fa3772378925f1a1b17415b9c85dc78b02c7f525522e1271f91a55ad2c72`。可回答题 Recall@1/5/10 为 0.3333/0.5333/0.5333，MRR@10 为 0.4333，nDCG@10 为 0.3972；检索延迟 p50/p95 约为 126.65/277.14 ms，本地 Dense 检索 token 与 API 成本均为 0。

立即以相同 run ID 恢复执行后没有重复检索，`summary.json` SHA-256 前后均为 `E609E165F818347631C8747700BFCECF240A0B10C09F28546655B05B4914DDD7`。本回执是 Development 纯 Dense 基线，不是 Test 结果，也不用于解除 `locked_until_s12`。
