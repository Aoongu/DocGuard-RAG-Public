# reports

该目录保存可审计的实验说明、人工复核记录和小型原始结果。由脚本生成的图表、临时日志和大体积报告写入 `reports/generated/`，默认不提交 Git。

报告必须区分原始测量、派生统计和文字结论；未运行的结果使用“待实验填写”，不以目标值冒充实际结果。

`s4_smoke_dense_v1/` 是冻结前候选集的历史 Development Smoke，仅证明早期评测链路可运行。`s4_smoke_dense_frozen_v1/` 是冻结黄金集对应的当前纯 Dense Development 基线。两者都不含 Test 运行结果，也不得用于解除 S12 前的 Test 执行锁。
