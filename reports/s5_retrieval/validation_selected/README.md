# Validation 选择回执

按冻结选择规则选中 `validation_grid/k20_d50_b20`：

- `dense_depth=50`
- `bm25_depth=20`
- `rrf_k=20`
- nDCG@10 = 0.45910060374861744
- MRR@10 = 0.46757936507936504
- Recall@5 = 0.6666666666666666
- Recall@10 = 0.7666666666666667

完整参数选择、基线身份和 10,000 次配对 bootstrap 结果见 `selection.json`。这是 Validation 选择结果，不是未见 Test 上的泛化估计。
