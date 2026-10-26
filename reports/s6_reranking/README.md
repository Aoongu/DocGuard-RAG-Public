# S6 Cross-Encoder 重排实验

## 实验边界

- 方法：M2 = S5 选中 M1（`rrf_k=20`、Dense depth 50、BM25 depth 20）+ Cross-Encoder。
- Reranker：`BAAI/bge-reranker-base`，revision `2cfc18c9415c912f9d8155881c133215df768a70`。
- ONNX：1,112,459,588 bytes，SHA-256 `15b9a8c3da82eddf263df571281166e00e9308fe19d077084b642ebfcaf06d2b`。
- Tokenizer：S2 固定资产，SHA-256 `9eb652ac4e40cc093272bbbe0f55d521cf67570060227109b5cdc20945a4489e`。
- 推理：CPUExecutionProvider，最大长度 512，`only_second` 截断，batch size 8。
- 选择集：冻结 Validation 60 题，其中 30 题可回答并参与检索指标。
- Test：未执行，仍为 `locked_until_s12`。

权重位于 Git 忽略的 `data/models/`；仓库只提交固定 manifest、逐题结果和汇总证据。原始 logit 只用于该固定模型内部排序，不解释为概率。

## 真实模型 smoke

`real_model_smoke.json` 保存了两个独立进程对同一组三个 query-passage 对的结果。两次输出逐字一致、所有 logit 有限，第三个超长 passage 两次均记录 `truncated=true`。

## Development 诊断

Development 只运行预注册中间深度 20，共 60/60 成功：

| 方法 | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | rerank p50/p95 | end-to-end p50/p95 |
|---|---:|---:|---:|---:|---:|---:|
| M2 depth20 | 0.4008 | 0.4417 | 0.6667 | 0.7000 | 4.81/5.78 s | 5.24/6.74 s |

该结果只用于功能和延迟诊断，不参与最终 depth 选择。

## Validation 深度比较

三组均为 60/60 成功，使用完全相同的数据集、索引、模型和 M1 参数：

| rerank depth | nDCG@10 | MRR@10 | Recall@5 | Recall@10 | rerank p50/p95 | end-to-end p50/p95 |
|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0.5581 | 0.6667 | 0.7667 | 0.7667 | 2.83/3.50 s | 3.53/4.15 s |
| 20 | 0.5840 | 0.6559 | 0.8000 | 0.8333 | 5.74/6.32 s | 6.35/6.98 s |
| 50 | **0.6053** | **0.6789** | **0.8333** | **0.8333** | 14.61/16.83 s | 15.18/17.50 s |

预注册选择顺序为 nDCG@10、MRR@10、Recall@10、Recall@5；只有质量完全相同时才比较 reranking p95 和更小 depth。因此冻结选择为 **depth 50**。这个选择追求离线排序质量，并不代表它已满足线上延迟要求。

## 冻结选择与基线比较

M1 指标为 nDCG@10 0.4591、MRR@10 0.4676、Recall@5 0.6667、Recall@10 0.7667；端到端 p50/p95 为 0.410/0.711 s。B1 指标为 0.3773、0.4575、0.6333、0.7000；端到端 p50/p95 为 0.023/0.178 s。

| 对比 | 指标 | 均值差 | 配对 bootstrap 95% CI | 解释 |
|---|---|---:|---:|---|
| M2-M1 | nDCG@10 | +0.1462 | [0.0727, 0.2226] | Validation 上稳定为正 |
| M2-M1 | MRR@10 | +0.2113 | [0.0944, 0.3388] | Validation 上稳定为正 |
| M2-M1 | Recall@5 | +0.1667 | [0.0333, 0.3000] | Validation 上稳定为正 |
| M2-M1 | Recall@10 | +0.0667 | [0.0000, 0.1667] | CI 触及 0，不宣称稳定提升 |
| M2-B1 | nDCG@10 | +0.2280 | [0.0944, 0.3619] | Validation 上稳定为正 |
| M2-B1 | MRR@10 | +0.2213 | [0.0502, 0.3897] | Validation 上稳定为正 |
| M2-B1 | Recall@5 | +0.2000 | [0.0667, 0.3333] | Validation 上稳定为正 |
| M2-B1 | Recall@10 | +0.1333 | [0.0000, 0.3000] | CI 触及 0，不宣称稳定提升 |

所有 CI 使用固定 seed `20261014`、10,000 次配对 bootstrap 和相同 30 道可回答题。完整值见 `validation_selected/selection.json`。

## 结论边界

固定 BGE Cross-Encoder 在当前 Validation 上显著改善前排排序质量，但选择的 depth50 把 CPU 端到端 p95 从 M1 的约 0.71 秒提高到约 17.50 秒。S6 只证明固定 Validation 上的质量—延迟权衡，不证明未见 Test 泛化，也不证明该配置适合在线部署。后续可以在不改变本阶段结论的前提下研究量化、缓存或较小模型；任何模型/revision/tokenizer/max_length 变化都必须建立新配置并重新校准。

## 回执完整性

- 四个已发布运行均完成相同 metadata 的断点恢复检查，`summary.json` SHA-256 不变。
- `failures.jsonl` 均为空。
- 选择器只读取保存结果，不导入或调用检索器/模型。
- 未创建任何 Test 检索结果目录。
