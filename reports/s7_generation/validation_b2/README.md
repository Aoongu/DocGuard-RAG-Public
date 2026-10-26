# S7 Validation B2 无效运行记录

本目录保存 2026-10-19 创建的 `s7-validation-b2-20261019`。它没有形成有效的模型实验：工作树缺少默认 `.qdrant/s3-dense-v1`，而当时的 CLI 在逐题检索时才检查索引，因此 60 题均在 API 调用前写为 `FileSafetyError`。

- 结果：0 success、60 failed。
- API 调用：0；input/cached/output token 均为 0。
- 费用：0 美元。
- 失败原因：本地 Dense 索引默认路径不存在，不是模型、网络或数据集质量失败。
- 协议处理：本目录永久保留，不覆盖、不删除、不恢复重跑其中任何失败行。
- 原因码处理：冻结 v1 原因码没有本地前置资产类别，因此该运行若进入统一失败复核，应使用排他的 `unknown`，并在说明中记录已知的本地索引缺失事实；不得伪记为 `upstream_failure`。

暴露的工程缺口已用测试复现：真实 B2 现在会在创建 `run.json` 或 `results.jsonl` 前检查 Qdrant 目录、E5 模型 manifest/文件和 tokenizer。若需要形成有效实验，只能使用新的输出目录和新的 `run_id`，并保留本目录作为失败证据。

| 文件 | 字节数 | SHA-256 |
|---|---:|---|
| `run.json` | 1,267 | `91a93ed6af11e57b62a23c28c3301ff6de1b65dfd0cba2169569dc6153cf2b9e` |
| `results.jsonl` | 46,745 | `d1d2d386519a30fe59431c66456b2b17b45361f8ab07deac8a6e9fa0fa28b059` |
| `failures.jsonl` | 8,160 | `a015243be5d08d17e178452e1161164d2ab2c16649c710fe77c2d0c5e796f7ff` |
| `summary.json` | 829 | `cc79a8e0a1614279d1ae84c062587d3ac5dbb9ea18c6dc7fc8488e56c018e1ec` |

Test split 未读取、未运行，继续保持 `locked_until_s12`。
