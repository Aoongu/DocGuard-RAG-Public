# S4 Dense smoke 回执

本目录由 `docguard.evaluation.cli` 对 30 条 development smoke 一次性生成：15 条可回答、15 条边界题，30 条均成功。`results.jsonl` 是逐题事实记录，`summary.json` 只从该文件派生，`run.json` 固定代码提交、数据/配置/语料哈希、模型 revision 和 Top-K 参数。

可回答题指标为 Recall@1/5/10 = 0.2000/0.4667/0.4667、MRR@10 = 0.3333、nDCG@10 = 0.3459。检索和端到端延迟 p50 分别约为 78.87/78.87 ms，p95 分别约为 169.42/169.42 ms；本地 Dense 检索 token 与 API 成本均为 0。

这些数字只证明冻结前的 S4 评测链路可运行，并显示纯 Dense 在当时候选标签上的表现。该数据哈希已被正式冻结集取代；本目录只作为历史证据保留，当前 Development 基线见 `../s4_smoke_dense_frozen_v1/`。Test split 从未执行。
