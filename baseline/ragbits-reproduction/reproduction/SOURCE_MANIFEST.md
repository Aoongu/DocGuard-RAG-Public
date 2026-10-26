# Source Manifest

## 上游来源

- 项目：Ragbits
- 官方仓库：https://github.com/deepsense-ai/ragbits
- 固定上游 commit：`32074a71324239ba677234a67737ed65fb678eeb`
- 许可证：MIT
- 许可证正文：仓库根目录 `LICENSE`
- 上游代码范围：`packages/**`、`examples/**` 及仓库中除下述复现新增文件外的固定提交内容

可分发 provenance 使用 sanitized/squashed 单提交快照：其 parent 直接为固定上游提交，tree 与原最终历史提交 `fc411730eb6bb95eaae02fcc7be5c5be34cceac0` 完全相同。该原历史提交仅作来源记录；由于其可达 DAG 包含早期未脱敏内容，bundle 不收录原 8 个中间提交。验证脚本枚举固定上游 commit 的全部原始跟踪路径，要求路径、blob 与 mode 一致，并扫描上游到快照间全部新增或修改的普通 blob；UTF-8 与带 BOM 的 UTF-16LE/BE 作可靠文本解码，其他对象使用 ASCII、UTF-16LE、UTF-16BE 原始字节视图兜底，未经显式 allowlist 不跳过。

本文件在外层归档中仅更新了上述 provenance 说明，因此不宣称其归档副本 blob 属于 sanitized 快照；快照内仍保留与原最终历史 tree 完全一致的版本。复现脚本、输入和结果文件未因该说明更新而改写。

## 本次复现新增文件

以下文件不是上游功能实现，而是复现输入、编排、测试和证据：

- `reproduction/environment.md`
- `reproduction/README_复现说明.md`
- `reproduction/复现结论.md`
- `reproduction/SOURCE_MANIFEST.md`
- `reproduction/verify_reproduction.ps1`
- `reproduction/inputs/evaluation.md`
- `reproduction/inputs/ragbits_overview.md`
- `reproduction/inputs/vector_store.md`
- `reproduction/run_document_search.py`
- `reproduction/smoke_offline_chat.py`
- `reproduction/tests/test_reproduction.py`
- `reproduction/tests/test_offline_chat_smoke.py`
- `reproduction/results/document_search.json`
- `reproduction/results/offline_chat_smoke.json`
- `reproduction/results/document_search_tests.txt`
- `reproduction/results/upstream_tests.txt`
- `reproduction/results/pip_freeze.txt`
- `reproduction/results/verification.txt`
- `docs/superpowers/plans/2026-10-11-ragbits-baseline-reproduction.md`

`reproduction/results/verification.txt` 是原复现历史产生的运行证据；sanitized/squashed provenance 只改写提交拓扑，不改写冻结结果。归档包完整性由归档包根目录的外置文件 `PACKAGE_VERIFICATION.txt` 记录，避免清单自引用循环。

运行时生成的 `.venv/`、`__pycache__/`、`.pytest_cache/`、模型缓存和机器临时文件不属于交付内容。用户交付目录中的 `README_交付说明.md`、`UPSTREAM_COMMIT.txt`、`UPSTREAM_TREE_SHA256.txt`、`SHA256SUMS.txt`、`VERIFY_PACKAGE.ps1` 与 `PACKAGE_VERIFICATION.txt` 是包装元数据，也不属于上游代码。
