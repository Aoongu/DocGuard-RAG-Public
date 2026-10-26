# configs

该目录从后续里程碑开始保存可版本控制、无密钥的运行配置，例如数据源清单、切分参数、检索方案、模型标识和评测协议。

真实密钥只放在被 Git 忽略的 `.env` 或外部密钥管理系统中。会影响实验结论的参数必须显式记录，不能依赖本机默认值。

`retrieval/s5-hybrid-v1.json` 固定 S5 技术词 tokenizer、BM25 常数、显式消融开关、RRF Validation 候选网格、选择指标顺序以及 S2 Chunk 哈希。它不包含由 Validation 选择出的最终参数；选择结果写入独立实验回执，Test 在 S12 前保持锁定。

`retrieval/s6-reranker-v1.json` 固定 S6 BGE Cross-Encoder 的模型 revision、模型 manifest、S2 tokenizer、512-token 截断、CPU 批处理、S5 已选 M1 参数和 rerank depth Validation 候选。原始 logit 只用于该固定模型内排序；Test 继续锁到 S12。

`generation/s7-generation-v1.json` 固定 S7 OpenAI-compatible Chat Completions 协议、模型快照、Prompt/schema 哈希、Evidence 预算、超时、价格和 S3/S4 identity。API Key 只从环境读取；模型、价格或服务根地址变化必须建立新配置版本。

`refusal/s8-refusal-v1.json` 固定 S8 生成前拒答的支持版本、透明阈值候选网格、选参排序、词面特征、版本冲突门槛、上游输入 SHA-256 与 Test 锁。Validation 只用于选择该配置的阈值，不能解释为未见数据泛化结果。

`evaluation/s12-final-v1.json` 是唯一允许解锁冻结 Test 的最终协议：它固定 Test 文件哈希、B0～M4 唯一 run ID、方法顺序、Validation 选中参数、六项消融映射、`max_retries=0` 和“任何已有记录均跳过”的恢复规则。该文件不包含 API Key；修改任一冻结字段都必须视为新的研究轮次，而不是覆盖本轮结果。
